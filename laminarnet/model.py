"""
LaminarNet v0.7.11 — RoPE, Dual-Gate CSR, Parallel Carry, Strata Validation
                    + Recurrent Inference (step / init_state)
Faster than Transformer: larger chunks, streamlined architecture, vectorized carry.
All temporal operations are strictly causal — no future information leakage.
Recurrent inference: token-by-token generation via step() using the same trained weights.
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

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
        return (self.scale * (x_f32 * rms)).to(orig_dtype)


# ─────────────────────────────────────────────────────────────
# Rotary Position Embedding
# ─────────────────────────────────────────────────────────────

class RotaryPositionEmbedding(nn.Module):
    """RoPE — computes sincos on-the-fly, works for any sequence length."""
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)            # (N, dim//2)
        cos_f = freqs.cos().to(dtype)                     # (N, dim//2)
        sin_f = freqs.sin().to(dtype)                     # (N, dim//2)
        return cos_f, sin_f

    def forward_single(self, pos: int, device: torch.device, dtype: torch.dtype):
        """Efficient RoPE for a single position."""
        t = torch.tensor([pos], device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (B, N, H, D). cos/sin are (N, D//2) or (1, D//2)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ─────────────────────────────────────────────────────────────
# 1. O(N) Selective Geometric Drift Field
# ─────────────────────────────────────────────────────────────

class GeometricDriftField(nn.Module):
    """
    Geometric Drift Field v7.10 — O(N) Vectorized Parallel Scan
    with RoPE and fully parallel inter-chunk carry.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 conv_kernel: int = 4, rope_base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.in_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel,
                                padding=conv_kernel-1, groups=d_model)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.dt_bias = nn.Parameter(torch.ones(d_model) * -3.0)
        self.rope = RotaryPositionEmbedding(self.d_head, base=rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, N, D = x.shape

        # 1. Context & Proj
        x_conv = self.conv1d(x.transpose(1, 2))[..., :N].transpose(1, 2)
        x_conv = F.silu(x_conv)

        fused = self.in_proj(x_conv)
        dt_raw, v, gate = fused.chunk(3, dim=-1)

        # 2. Selective Parameters
        dt = F.softplus(dt_raw.float() + self.dt_bias.float()).clamp(min=0.001, max=2.0).to(dt_raw.dtype)
        gate = torch.sigmoid(gate)
        log_alpha = -dt

        # 3. RoPE
        cos_f, sin_f = self.rope(N, device=x.device, dtype=x.dtype)
        v_view = v.view(B, N, self.n_heads, self.d_head)
        v_rotated = apply_rope(v_view, cos_f, sin_f).reshape(B, N, D)

        # 4. Scan
        chunk_size = 256
        v_in = v_rotated * dt
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

        L_chunks = torch.cumsum(la_chunks, dim=2).float().clamp(min=-20.0, max=0.0)
        L_max = L_chunks.max(dim=2, keepdim=True).values
        L_stable = L_chunks - L_max
        exp_neg_L_stable = torch.exp(-L_stable)
        scaled_v = exp_neg_L_stable * v_chunks.float()
        cum_scaled = torch.cumsum(scaled_v, dim=2)
        exp_L_stable = torch.exp(L_stable)
        chunk_out = exp_L_stable * cum_scaled

        if num_chunks > 1:
            chunk_boundary_decay = L_chunks[:, :, -1, :]
            chunk_boundary_out = chunk_out[:, :, -1, :]
            cum_decay = torch.cumsum(chunk_boundary_decay, dim=1).clamp(min=-80.0, max=0.0)
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
        gate = gate[:, :orig_N, :]

        # 6. Output
        out = self.out_proj(final_out * gate)
        return residual + self.dropout(out)

    def step(self, x: torch.Tensor, state: dict) -> tuple:
        residual = x
        x = self.norm(x)
        B, _, D = x.shape

        conv_buf = state["conv_buf"]
        x_t = x.transpose(1, 2)
        conv_input = torch.cat([conv_buf, x_t], dim=2)
        new_conv_buf = conv_input[:, :, 1:]

        x_conv = F.conv1d(conv_input, self.conv1d.weight, bias=self.conv1d.bias, groups=D)
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        fused = self.in_proj(x_conv)
        dt_raw, v, gate = fused.chunk(3, dim=-1)

        dt = F.softplus(dt_raw.float() + self.dt_bias.float()).clamp(min=0.001, max=2.0).to(dt_raw.dtype)
        gate = torch.sigmoid(gate)
        alpha = torch.exp(-dt.squeeze(1))

        pos = state["pos"]
        cos_f, sin_f = self.rope.forward_single(pos, device=x.device, dtype=x.dtype)
        v_view = v.view(B, 1, self.n_heads, self.d_head)
        v_rotated = apply_rope(v_view, cos_f, sin_f).reshape(B, D)

        carry = state["carry"]
        new_carry = alpha * carry + dt.squeeze(1) * v_rotated

        out = self.out_proj((new_carry.unsqueeze(1)) * gate)
        new_state = {"carry": new_carry, "conv_buf": new_conv_buf, "pos": pos + 1}
        return residual + self.dropout(out), new_state


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
        h_coarse = h_coarse + self.gate_f2c(f_to_c[:, :Lc, :]) * f_to_c[:, :Lc, :]
        c_to_f = self.up(h_coarse.transpose(1, 2)).transpose(1, 2)
        Lf = h_fine.shape[1]
        if c_to_f.shape[1] < Lf:
            c_to_f = F.pad(c_to_f, (0, 0, 0, Lf - c_to_f.shape[1]))
        h_fine = h_fine + self.gate_c2f(c_to_f[:, :Lf, :]) * c_to_f[:, :Lf, :]
        return h_fine, h_coarse

    def step(self, h_fine: torch.Tensor, h_coarse: torch.Tensor):
        """Efficient token-by-token routing."""
        h_coarse = h_coarse + self.gate_f2c(h_fine) * h_fine
        h_fine = h_fine + self.gate_c2f(h_coarse) * h_coarse
        return h_fine, h_coarse


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        res = x
        x = self.norm(x)
        return res + self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class LaminarNet(nn.Module):
    def __init__(self, config: LaminarNetConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.tok_emb = nn.Embedding(config.vocab_size, d)
        self.dropout = nn.Dropout(config.dropout)
        
        # Strata ratios are already validated in __post_init__
        self.strata_init = nn.ModuleList([
            nn.AvgPool1d(kernel_size=r, stride=r) 
            for r in config.strata_ratios[1:config.n_strata]
        ])
        
        self.blocks = nn.ModuleList([LaminarBlock(config) for _ in range(config.n_layers)])
        self.norm_out = RMSNorm(d)
        self.head = nn.Linear(d, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        
        self.apply(lambda m: nn.init.normal_(m.weight, std=0.02) if isinstance(m, (nn.Linear, nn.Embedding)) else None)

    def forward(self, ids):
        B, N = ids.shape
        x = self.dropout(self.tok_emb(ids))
        coarse_strata = []
        for pool in self.strata_init:
            x_t = x.transpose(1, 2)
            k = pool.kernel_size[0]
            x_t = F.pad(x_t, (k - 1, 0))
            coarse_strata.append(pool(x_t).transpose(1, 2))
        strata = [x] + coarse_strata
        for b in self.blocks: 
            strata = b(strata)
        return self.head(self.norm_out(strata[0]))

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}

    def init_state(self, batch_size: int = 1, device: torch.device = torch.device("cpu")) -> dict:
        D = self.config.d_model
        K = self.config.conv_kernel
        # Max pool size needed for recurrent strata emulation
        max_r = max(self.config.strata_ratios[:self.config.n_strata])
        
        state = {
            "blocks": [], 
            "emb_buffer": torch.zeros(batch_size, D, max_r, device=device),
            "step_count": 0
        }
        for _ in self.blocks:
            block_state = []
            for _ in range(self.config.n_strata):
                block_state.append({
                    "carry": torch.zeros(batch_size, D, device=device),
                    "conv_buf": torch.zeros(batch_size, D, K - 1, device=device),
                    "pos": 0,
                })
            state["blocks"].append(block_state)
        return state

    @torch.no_grad()
    def step(self, token_id: torch.Tensor, state: dict) -> tuple:
        """
        Run a single token recurrently. 
        Note: 'state' is mutated in-place for efficiency.
        """
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        x = self.tok_emb(token_id)
        
        # 1. Update embedding buffer for causal strata initialization
        # (B, D, max_r) -> slide and add new token
        state["emb_buffer"] = torch.cat([state["emb_buffer"][:, :, 1:], x.transpose(1, 2)], dim=2)
        
        # 2. Reconstruct current coarse strata inputs (mimics strata_init pooling)
        coarse_inputs = []
        for r in self.config.strata_ratios[1:self.config.n_strata]:
            # Exact causal mean of last 'r' embeddings matches AvgPool1d logic
            c_x = state["emb_buffer"][:, :, -r:].mean(dim=2, keepdim=True).transpose(1, 2)
            coarse_inputs.append(c_x)
        
        strata = [x] + coarse_inputs
        
        # 3. Process through blocks (internal GDF states handle temporal carry)
        for i, block in enumerate(self.blocks):
            strata, bs = block.step(strata, state["blocks"][i])
            state["blocks"][i] = bs

        state["step_count"] += 1
        logits = self.head(self.norm_out(strata[0]))
        return logits.squeeze(1), state


class LaminarBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.S = config.n_strata
        self.gdfs = nn.ModuleList([GeometricDriftField(config.d_model, config.n_heads, config.dropout, config.conv_kernel, config.rope_base) for _ in range(self.S)])
        self.csrs = nn.ModuleList([CrossStratumRouting(config.d_model, config.strata_ratios[s+1]//config.strata_ratios[s]) for s in range(self.S-1)])
        self.ffns = nn.ModuleList([SwiGLUFFN(config.d_model, config.d_ff, config.dropout) for _ in range(self.S)])

    def forward(self, strata):
        # 1. Temporal Processing (GDF)
        for s in range(self.S):
            strata[s] = self.gdfs[s](strata[s])
        
        # 2. Cross-Stratum Routing (CSR)
        for s in range(self.S - 1):
            strata[s], strata[s+1] = self.csrs[s](strata[s], strata[s+1])
        
        # 3. Channel Processing (FFN)
        for s in range(self.S):
            strata[s] = self.ffns[s](strata[s])
            
        return strata

    def step(self, strata: list, block_state: list) -> tuple:
        new_block_state = []
        # 1. GDF Step (Handles carry for each stratum)
        for s in range(self.S):
            strata[s], new_s = self.gdfs[s].step(strata[s], block_state[s])
            new_block_state.append(new_s)
            
        # 2. CSR Step (Optimized)
        for s in range(self.S - 1):
            strata[s], strata[s+1] = self.csrs[s].step(strata[s], strata[s+1])
            
        # 3. FFN Step
        for s in range(self.S):
            strata[s] = self.ffns[s](strata[s])
            
        return strata, new_block_state


if __name__ == "__main__":
    conf = LaminarNetConfig()
    model = LaminarNet(conf)
    x = torch.randint(0, conf.vocab_size, (2, 128))
    params = model.count_parameters()
    print(f"LaminarNet v0.7.11 | Params: {params['trainable']/1e6:.1f}M | Out: {model(x).shape}")
