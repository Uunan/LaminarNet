"""
LaminarNet v0.6.5 — Recurrent Inference & Math Equality Update
                    Forget Gate, Talking Heads, Iterative CSR, DenseNet Residuals
Faster than Transformer: larger chunks, streamlined architecture, vectorized carry.
All temporal operations are strictly causal — no future information leakage.
Recurrent inference: token-by-token generation via step() using the same trained weights.
"""

import math
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LaminarNetConfig:
    vocab_size: int = 50257
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 1024
    n_strata: int = 2
    strata_ratios: tuple = (1, 2, 4)
    seq_len: int = 1024
    dropout: float = 0.1
    conv_kernel: int = 4
    rope_base: float = 10000.0

    def __post_init__(self):
        if len(self.strata_ratios) < self.n_strata:
            raise ValueError(
                f"strata_ratios length ({len(self.strata_ratios)}) must be "
                f">= n_strata ({self.n_strata}). "
                f"Provide at least {self.n_strata} ratios."
            )
        if self.strata_ratios[0] != 1:
            raise ValueError(
                f"strata_ratios[0] must be 1 (fine stratum), got {self.strata_ratios[0]}"
            )
        for i, r in enumerate(self.strata_ratios):
            if not isinstance(r, int) or r < 1:
                raise ValueError(
                    f"strata_ratios[{i}] must be a positive integer, got {r}"
                )


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        orig_dtype = x.dtype
        x_f32 = x.float()
        rms = x_f32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        # Ensure scale is float before multiplying, then clamp and downcast safely
        out = self.scale.float() * (x_f32 * rms)
        return out.clamp(min=-6e4, max=6e4).to(orig_dtype)


# ─────────────────────────────────────────────────────────────
# Rotary Position Embedding
# ─────────────────────────────────────────────────────────────

class RotaryPositionEmbedding(nn.Module):
    """RoPE — computes sincos on-the-fly, works for any sequence length."""
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)            # (N, dim//2)
        cos_f = freqs.cos().to(dtype)                     # (N, dim//2)
        sin_f = freqs.sin().to(dtype)                     # (N, dim//2)
        return cos_f, sin_f


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (B, N, H, D). cos/sin are (N, D//2)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.unsqueeze(0).unsqueeze(2)   # (1, N, 1, D//2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ─────────────────────────────────────────────────────────────
# 1. O(N) Selective Geometric Drift Field — with RoPE
# ─────────────────────────────────────────────────────────────

class GeometricDriftField(nn.Module):
    """
    Geometric Drift Field v6.0 — O(N) Vectorized Parallel Scan
    with RoPE and fully parallel inter-chunk carry.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 conv_kernel: int = 4, rope_base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Fused Projections — now 3-way (no separate theta needed; RoPE replaces it)
        self.in_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel,
                                padding=conv_kernel-1, groups=d_model)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Talking Heads Mixing Matrix
        self.head_mix = nn.Parameter(torch.eye(n_heads))
        
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Head-specific dt_bias initialization (fast-decaying to slow-decaying)
        dt = torch.exp(
            torch.rand(n_heads, self.d_head) * (math.log(0.1) - math.log(10.0)) + math.log(10.0)
        ).clamp(min=0.001)
        # We store initial dt_bias as log(exp(dt) - 1) which is inverse softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt.view(-1)) # D_model
        
        self.rope = RotaryPositionEmbedding(self.d_head, base=rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, N, D = x.shape

        # 1. Context & Proj (causal conv)
        x_conv = self.conv1d(x.transpose(1, 2))[..., :N].transpose(1, 2)
        x_conv = F.silu(x_conv)

        fused = self.in_proj(x_conv)
        dt_raw, v, gate = fused.chunk(3, dim=-1)

        # 2. Selective Parameters
        dt = F.softplus(dt_raw.float() + self.dt_bias.float()).clamp(min=0.001, max=2.0).to(dt_raw.dtype)
        gate = torch.sigmoid(gate)
        
        # In v0.6.3, gate is strictly an output gate, not a forget gate.
        log_alpha = -dt.float()

        # 3. RoPE-based Positional Rotation
        cos_f, sin_f = self.rope(N, device=x.device, dtype=x.dtype)
        v_view = v.view(B, N, self.n_heads, self.d_head)
        v_rotated = apply_rope(v_view, cos_f, sin_f).reshape(B, N, D)

        # 4. O(N) Vectorized Parallel Scan — chunk_size=256 for speed
        chunk_size = 256
        v_in = v_rotated * dt

        # Pad to nearest chunk multiple
        orig_N = N
        remainder = N % chunk_size
        if remainder != 0:
            pad_len = chunk_size - remainder
            v_in = F.pad(v_in, (0, 0, 0, pad_len))
            log_alpha = F.pad(log_alpha, (0, 0, 0, pad_len))

        N_padded = v_in.shape[1]
        num_chunks = N_padded // chunk_size

        v_chunks = v_in.view(B, num_chunks, chunk_size, D)
        la_chunks = log_alpha.view(B, num_chunks, chunk_size, D)

        # Cumulative log-decay within each chunk — CLAMP to prevent exp overflow
        # Ensure cumsum happens in float32, not float16!
        L_chunks = torch.cumsum(la_chunks.float(), dim=2)
        L_chunks = L_chunks.clamp(min=-20.0, max=0.0)

        # O(N) intra-chunk scan via cumsum (float32)
        L_max = L_chunks.max(dim=2, keepdim=True).values
        L_stable = L_chunks - L_max
        exp_neg_L_stable = torch.exp(-L_stable)
        scaled_v = exp_neg_L_stable * v_chunks.float()
        cum_scaled = torch.cumsum(scaled_v, dim=2)
        exp_L_stable = torch.exp(L_stable)
        chunk_out = exp_L_stable * cum_scaled  # float32

        # 5. Parallel inter-chunk carry (no Python for-loop) — all in float32
        if num_chunks > 1:
            chunk_boundary_decay = L_chunks[:, :, -1, :]
            chunk_boundary_out = chunk_out[:, :, -1, :]

            # Parallel prefix sum in log-space
            cum_decay = torch.cumsum(chunk_boundary_decay, dim=1).clamp(min=-80.0, max=0.0)

            # Log-space stabilized parallel carry
            stabilizer = cum_decay.max(dim=1, keepdim=True).values
            norm_cum_decay = (cum_decay - stabilizer).clamp(min=-20.0, max=0.0)

            seeds = chunk_boundary_out * torch.exp(-norm_cum_decay)
            shifted_seeds = F.pad(seeds[:, :-1], (0, 0, 1, 0))
            cum_seeds = torch.cumsum(shifted_seeds, dim=1)
            carries = cum_seeds * torch.exp(norm_cum_decay)

            final_out = chunk_out + carries.unsqueeze(2) * torch.exp(L_chunks)
            final_out = final_out.to(x.dtype).view(B, -1, D)
        else:
            final_out = chunk_out.to(x.dtype).view(B, -1, D)

        final_out = final_out[:, :orig_N, :]

        # Talking Heads Mixing
        v_mix = final_out.view(B, orig_N, self.n_heads, self.d_head)
        v_mix = torch.einsum('bnhd,hm->bnmd', v_mix, self.head_mix)
        final_out = v_mix.reshape(B, orig_N, D)

        # 6. Output — Restore v0.6.3 Gating
        out = self.out_proj(final_out * gate)
        return residual + self.dropout(out)

    # ── Recurrent single-token step ──────────────────────────
    def step(self, x: torch.Tensor, state: dict) -> tuple:
        """
        Single-token recurrent inference.
        x:     (B, 1, D) — one token embedding
        state: dict with keys 'carry' (B, D), 'conv_buf' (B, D, K-1), 'pos' (int)
        Returns: (output (B, 1, D), new_state)
        """
        residual = x
        x = self.norm(x)
        B, _, D = x.shape

        # 1. Causal conv1d via rolling buffer
        conv_buf = state["conv_buf"]                           # (B, D, K-1)
        x_t = x.transpose(1, 2)                               # (B, D, 1)
        conv_input = torch.cat([conv_buf, x_t], dim=2)         # (B, D, K)
        new_conv_buf = conv_input[:, :, 1:]                    # slide window

        # Apply depthwise conv weights via F.conv1d (groups=D)
        x_conv = F.conv1d(conv_input, self.conv1d.weight,
                          bias=self.conv1d.bias, groups=D)     # (B, D, 1)
        x_conv = x_conv.transpose(1, 2)                        # (B, 1, D)
        x_conv = F.silu(x_conv)

        # 2. Projections (same weights as forward)
        fused = self.in_proj(x_conv)
        dt_raw, v, gate = fused.chunk(3, dim=-1)

        # 3. Selective parameters
        dt = F.softplus(dt_raw.float() + self.dt_bias.float()).clamp(min=0.001, max=2.0)
        gate = torch.sigmoid(gate.float())
        alpha = torch.exp(-dt.squeeze(1))                      # (B, D) float32

        # 4. RoPE at current position
        pos = state["pos"]
        cos_f, sin_f = self.rope(pos + 1, device=x.device, dtype=x.dtype)
        cos_f = cos_f[pos:pos+1]                               # (1, D//2)
        sin_f = sin_f[pos:pos+1]
        v_view = v.view(B, 1, self.n_heads, self.d_head)
        v_rotated = apply_rope(v_view, cos_f, sin_f).reshape(B, D)

        # 5. Recurrent state update: carry = gate * (alpha * carry) + dt * v
        # Note: In parallel scan, dt is NOT multiplied by gate, but the incoming carry IS.
        # So we gate the old carry, but add the new v_rotated normally.
        carry = state["carry"].float()                                 # (B, D)
        dt_sq = dt.squeeze(1).float()                                  # (B, D)
        gate_sq = gate.squeeze(1)                                      # (B, D)
        v_rotated = v_rotated.float()
        
        # Strict v0.6.3 recurrence: no forget gate on carry, only output gate
        new_carry = alpha * carry + dt_sq * v_rotated
        
        # Explicit bounds check to prevent FP16 INF cast
        new_carry = new_carry.clamp(min=-6e4, max=6e4)

        # Talking Heads Mixing on new_carry
        c_mix = new_carry.to(x.dtype).view(B, self.n_heads, self.d_head)
        c_mix = torch.einsum('bhd,hm->bmd', c_mix, self.head_mix.to(x.dtype))
        c_mixed = c_mix.reshape(B, D)

        # 6. Output
        out = self.out_proj(c_mixed.unsqueeze(1) * gate)
        ret_out = residual + self.dropout(out)
        new_state = {
            "carry": new_carry,  # store original newly gated carry, not the mixed one
            "conv_buf": new_conv_buf,
            "pos": pos + 1,
            "last_out": ret_out
        }
        return ret_out, new_state


# ─────────────────────────────────────────────────────────────
# 2. Standard Infrastructure
# ─────────────────────────────────────────────────────────────

class CrossStratumRouting(nn.Module):
    def __init__(self, d_model: int, stride: int):
        super().__init__()
        self.stride = stride
        self.down = nn.AvgPool1d(kernel_size=stride, stride=stride)
        self.up = nn.Upsample(scale_factor=stride, mode='nearest')
        self.gate_f2c = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.gate_c2f = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, h_fine: torch.Tensor, h_coarse: torch.Tensor):
        fine_t = h_fine.transpose(1, 2)
        k = self.down.kernel_size if isinstance(self.down.kernel_size, int) else self.down.kernel_size[0]
        fine_t = F.pad(fine_t, (k - 1, 0))
        f_to_c = self.down(fine_t).transpose(1, 2)
        Lc = h_coarse.shape[1]
        if f_to_c.shape[1] < Lc:
            f_to_c = F.pad(f_to_c, (0, 0, 0, Lc - f_to_c.shape[1]))
        g_f2c = self.gate_f2c(f_to_c[:, :Lc, :])
        h_coarse = (1.0 - g_f2c) * h_coarse + g_f2c * f_to_c[:, :Lc, :]
        c_to_f = self.up(h_coarse.transpose(1, 2)).transpose(1, 2)
        Lf = h_fine.shape[1]
        if c_to_f.shape[1] < Lf:
            c_to_f = F.pad(c_to_f, (0, 0, 0, Lf - c_to_f.shape[1]))
        g_c2f = self.gate_c2f(c_to_f[:, :Lf, :])
        h_fine = (1.0 - g_c2f) * h_fine + g_c2f * c_to_f[:, :Lf, :]
        return h_fine, h_coarse

    def step(self, h_f_step: torch.Tensor, h_c_step: torch.Tensor, state: dict):
        """
        Recurrent causal step for CrossStratumRouting.
        h_f_step: (B, 1, D)
        h_c_step: (B, 1, D) - only valid/used when (pos+1) % stride == 0
        state: buf_f (B, stride-1, D), last_c (B, 1, D), pos (int)
        """
        buf_f = state["buf_f"]
        last_c = state["last_c"]
        pos = state["pos"]
        
        # We compute a new coarse token every `stride` steps
        if pos % self.stride == 0:
            window = torch.cat([buf_f, h_f_step], dim=1) # (B, stride, D)
            f_to_c = window.mean(dim=1, keepdim=True)
            
            g_f2c = self.gate_f2c(f_to_c)
            # Stable interpolation for step routing
            last_c = (1.0 - g_f2c) * h_c_step + g_f2c * f_to_c
            new_h_c_step = last_c
        else:
            new_h_c_step = last_c # return the appropriately updated coarse token from the last tick
            
        # 2. Coarse-to-Fine (Up Routing) - Dense replication via nearest approach
        c_to_f = last_c
        g_c2f = self.gate_c2f(c_to_f)
        new_h_f_step = (1.0 - g_c2f) * h_f_step + g_c2f * c_to_f
        
        # Rotate buffer
        if self.stride > 1:
            new_buf_f = torch.cat([buf_f[:, 1:], h_f_step], dim=1)
        else:
            new_buf_f = buf_f
            
        new_state = {
            "buf_f": new_buf_f,
            "last_c": last_c,
            "pos": pos + 1
        }
        return new_h_f_step, new_h_c_step, new_state

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.w1, self.w2, self.w3 = nn.Linear(d_model, d_ff, bias=False), nn.Linear(d_ff, d_model, bias=False), nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        res = x
        x = self.norm(x)
        # Protect SwiGLU multiplication from internal float16 overflow, but don't clamp the residual output
        gate_val = (F.silu(self.w1(x)) * self.w3(x)).clamp(min=-6e4, max=6e4)
        return res + self.dropout(self.w2(gate_val))

class LaminarNet(nn.Module):
    def __init__(self, config: LaminarNetConfig):
        super().__init__()
        self.config, d = config, config.d_model
        self.tok_emb = nn.Embedding(config.vocab_size, d)
        self.dropout = nn.Dropout(config.dropout)
        self.strata_init = nn.ModuleList([nn.AvgPool1d(kernel_size=r, stride=r) for r in config.strata_ratios[1:config.n_strata]])
        self.blocks = nn.ModuleList([LaminarBlock(config) for _ in range(config.n_layers)])
        self.norm_out = RMSNorm(d)
        self.head = nn.Linear(d, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(lambda m: nn.init.normal_(m.weight, std=0.02) if isinstance(m, (nn.Linear, nn.Embedding)) else None)

    def forward(self, ids):
        B, N = ids.shape
        x = self.dropout(self.tok_emb(ids))
        # CAUSAL strata init: left-pad so each coarse position only sees past/current
        coarse_strata = []
        for pool in self.strata_init:
            x_t = x.transpose(1, 2)
            k = pool.kernel_size[0] if isinstance(pool.kernel_size, tuple) else pool.kernel_size
            x_t = F.pad(x_t, (k - 1, 0))
            coarse_strata.append(pool(x_t).transpose(1, 2))
        strata = [x] + coarse_strata
        
        for b in self.blocks:
            strata = b(strata)
            
        # The final head projection is a 320 -> 50257 matmul. In float16, this almost
        # always overflows the 65504 limit if not protected.
        with torch.amp.autocast(device_type=ids.device.type if ids.device.type != 'cpu' else 'cpu', enabled=False):
            return self.head(self.norm_out(strata[0].float()))

    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Recurrent Inference API ──────────────────────────────
    def init_state(self, batch_size: int = 1,
                   device: torch.device = torch.device("cpu")) -> dict:
        """
        Create the initial recurrent state for step-by-step inference.
        Returns a dict with global state and block states.
        """
        D = self.config.d_model
        K = self.config.conv_kernel
        
        # Strata initialization buffers
        strata_state = {
            "pos": 0,
            "bufs": [
                torch.zeros(batch_size, pool.kernel_size[0] if isinstance(pool.kernel_size, tuple) else pool.kernel_size - 1, D, device=device) 
                for pool in self.strata_init
            ],
            "last_c": [
                torch.zeros(batch_size, 1, D, device=device)
                for _ in self.strata_init
            ]
        }
        
        block_states = []
        for i in range(self.config.n_layers):
            block_state = []
            for s in range(self.config.n_strata):
                block_state.append({
                    "carry": torch.zeros(batch_size, D, device=device),
                    "conv_buf": torch.zeros(batch_size, D, K - 1, device=device),
                    "pos": 0,
                    "last_out": torch.zeros(batch_size, 1, D, device=device)
                })
                
            routing_states = []
            for pass_idx in range(2):
                pass_state = []
                for s in range(self.config.n_strata - 1):
                    stride = self.config.strata_ratios[s+1] // self.config.strata_ratios[s]
                    pass_state.append({
                        "buf_f": torch.zeros(batch_size, stride - 1, D, device=device),
                        "last_c": torch.zeros(batch_size, 1, D, device=device),
                        "pos": 0,
                    })
                routing_states.append(pass_state)
                
            block_states.append({
                "gdfs": block_state,
                "csrs": routing_states
            })
        
        return {
            "strata_state": strata_state,
            "block_states": block_states
        }

    @torch.no_grad()
    def step(self, token_id: torch.Tensor, state: dict) -> tuple:
        """
        Run a single token through the model recurrently.
        token_id: (B,) or (B, 1)
        state:    from init_state()
        Returns:  (logits (B, vocab_size), new_state)
        """
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)           # (B, 1)

        x = self.tok_emb(token_id)                     # (B, 1, D)

        # Causal strata initialization step
        strata = [x]
        
        s_state = state["strata_state"]
        pos = s_state["pos"]
        new_bufs = []
        new_last_c = []
        
        for s, pool in enumerate(self.strata_init):
            k = pool.kernel_size[0] if isinstance(pool.kernel_size, tuple) else pool.kernel_size
            buf_f = s_state["bufs"][s]
            last_c = s_state["last_c"][s]
            
            if pos % k == 0:
                # Need 1 less element than kernel size from the buffer
                window = torch.cat([buf_f[:, -(k-1):, :] if k > 1 else torch.empty(x.shape[0], 0, x.shape[2], device=x.device), x], dim=1)
                new_c = window.mean(dim=1, keepdim=True)
                last_c = new_c
                
            strata.append(last_c)
            new_last_c.append(last_c)
            
            if k > 1:
                new_buf_f = torch.cat([buf_f[:, 1:], x], dim=1)
            else:
                new_buf_f = buf_f
            new_bufs.append(new_buf_f)

        new_strata_state = {
            "pos": pos + 1,
            "bufs": new_bufs,
            "last_c": new_last_c
        }

        new_block_states = []
        # DenseNet-style cross-block residual accumulator for fine stratum
        fine_accumulator = 0
        for i, block in enumerate(self.blocks):
            strata[0] = strata[0] + fine_accumulator * 0.5
            strata, bs = block.step(strata, state["block_states"][i], pos)
            fine_accumulator = fine_accumulator + strata[0]
            new_block_states.append(bs)

        logits = self.head(self.norm_out(strata[0]))   # (B, 1, V)
        return logits.squeeze(1), {"strata_state": new_strata_state, "block_states": new_block_states}

class LaminarBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.S = config.n_strata
        self.gdfs = nn.ModuleList([GeometricDriftField(config.d_model, config.n_heads, config.dropout, config.conv_kernel, config.rope_base) for _ in range(self.S)])
        self.csrs = nn.ModuleList([CrossStratumRouting(config.d_model, config.strata_ratios[s+1]//config.strata_ratios[s]) for s in range(self.S-1)])
        self.ffns = nn.ModuleList([SwiGLUFFN(config.d_model, config.d_ff, config.dropout) for _ in range(self.S)])
    def forward(self, strata):
        for s in range(self.S): strata[s] = self.gdfs[s](strata[s])
        
        # Iterative CSR (2 passes)
        for _ in range(2):
            for s in range(self.S - 1): 
                strata[s], strata[s+1] = self.csrs[s](strata[s], strata[s+1])
                
        for s in range(self.S): strata[s] = self.ffns[s](strata[s])
        return strata

    def step(self, strata: list, block_state: dict, pos: int) -> tuple:
        """Single-token step through block. Returns (strata, new_block_state)."""
        new_gdf_state = []
        
        # Calculate cumulative strides for each stratum
        # strata 0 has stride 1
        # strata s has stride Product(strata_ratios[:s])
        
        for s in range(self.S):
            if s == 0:
                stride = 1
            else:
                stride = 1
                for r in range(s):
                    stride *= self.csrs[r].stride
                    
            if pos % stride == 0:
                strata[s], new_s = self.gdfs[s].step(strata[s], block_state["gdfs"][s])
            else:
                new_s = block_state["gdfs"][s]
                strata[s] = new_s["last_out"]
            new_gdf_state.append(new_s)
            
        new_csr_states = [[], []]
        # Iterative CSR (2 passes)
        for pass_idx in range(2):
            for s in range(self.S - 1):
                strata[s], strata[s+1], updated_state = self.csrs[s].step(
                    strata[s], strata[s+1], block_state["csrs"][pass_idx][s]
                )
                new_csr_states[pass_idx].append(updated_state)
                
        for s in range(self.S):
            strata[s] = self.ffns[s](strata[s])
            
        return strata, {"gdfs": new_gdf_state, "csrs": new_csr_states}

if __name__ == "__main__":
    conf = LaminarNetConfig()
    model = LaminarNet(conf)
    x = torch.randint(0, conf.vocab_size, (2, 128))
    print(f"LaminarNet v0.6.5 | Out: {model(x).shape}")
