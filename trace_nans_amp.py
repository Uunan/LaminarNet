import torch
import torch.nn.functional as F
from laminarnet import LaminarNet, LaminarNetConfig

config = LaminarNetConfig(
    vocab_size=50257, d_model=320, n_heads=5, n_layers=10,
    d_ff=1200, n_strata=2, strata_ratios=(1,2,4), seq_len=2048, dropout=0.1
)
model = LaminarNet(config)
model.cuda() if torch.cuda.is_available() else None
model.train()

x = torch.randint(0, config.vocab_size, (2, 2048)).to(next(model.parameters()).device)
y = torch.randint(0, config.vocab_size, (2, 2048)).to(next(model.parameters()).device)

scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
opt = torch.optim.AdamW(model.parameters(), lr=6e-4)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device_type == 'cuda' else torch.bfloat16

def check_nan(name, val):
    if torch.isnan(val).any():
        print(f"NaN DETECTED AT: {name}")
        exit(1)

for step in range(3):
    opt.zero_grad()
    with torch.amp.autocast(device_type, enabled=True, dtype=dtype):
        # We need to manually trace the forward pass to find the exact NaN source
        
        # tok_emb
        emb = model.dropout(model.tok_emb(x))
        check_nan("tok_emb", emb)
        
        # Strata init
        coarse_strata = []
        for pool in model.strata_init:
            x_t = emb.transpose(1, 2)
            k = pool.kernel_size[0] if isinstance(pool.kernel_size, tuple) else pool.kernel_size
            x_t = F.pad(x_t, (k - 1, 0))
            coarse_strata.append(pool(x_t).transpose(1, 2))
        strata = [emb] + coarse_strata
        check_nan("strata_init fine", strata[0])
        check_nan("strata_init coarse", strata[1])
        
        fine_acc = 0.0
        for i, b in enumerate(model.blocks):
            strata[0] = strata[0] + fine_acc * 0.5
            
            # GDF
            for s in range(b.S):
                # Manual GDF trace
                gdf = b.gdfs[s]
                x_in_gdf = strata[s]
                residual = x_in_gdf
                x_n = gdf.norm(x_in_gdf)
                B, N, D = x_n.shape
                
                x_conv = gdf.conv1d(x_n.transpose(1, 2))[..., :N].transpose(1, 2)
                x_conv = F.silu(x_conv)
                
                fused = gdf.in_proj(x_conv)
                dt_raw, v, gate = fused.chunk(3, dim=-1)
                
                dt = F.softplus(dt_raw.float() + gdf.dt_bias.float()).clamp(min=0.001, max=2.0).to(dt_raw.dtype)
                gate = torch.sigmoid(gate)
                log_gate = torch.log(gate.float() + 1e-6)
                log_alpha = -dt.float() + log_gate
                check_nan(f"Block {i} GDF {s} log_alpha", log_alpha)
                
                cos_f, sin_f = gdf.rope(N, device=x_n.device, dtype=x_n.dtype)
                v_view = v.view(B, N, gdf.n_heads, gdf.d_head)
                v_rotated = gdf.apply_rope(v_view, cos_f, sin_f).reshape(B, N, D)
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
                
                L_chunks = torch.cumsum(la_chunks.float(), dim=2)
                L_chunks = L_chunks.clamp(min=-20.0, max=0.0)
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
                    final_out = final_out.to(x_in_gdf.dtype).view(B, -1, D)
                else:
                    final_out = chunk_out.to(x_in_gdf.dtype).view(B, -1, D)
                
                final_out = final_out[:, :orig_N, :]
                check_nan(f"Block {i} GDF {s} final_out", final_out)
                
                v_mix = final_out.view(B, orig_N, gdf.n_heads, gdf.d_head)
                v_mix = torch.einsum('bnhd,hm->bnmd', v_mix, gdf.head_mix)
                final_out = v_mix.reshape(B, orig_N, D)
                out = gdf.out_proj(final_out)
                strata[s] = residual + gdf.dropout(out)
                check_nan(f"Block {i} GDF {s} output", strata[s])

            # CSR
            for s in range(b.S - 1):
                csr = b.csrs[s]
                h_fine, h_coarse = strata[s], strata[s+1]
                fine_t = h_fine.transpose(1, 2)
                k = csr.down.kernel_size if isinstance(csr.down.kernel_size, int) else csr.down.kernel_size[0]
                fine_t = F.pad(fine_t, (k - 1, 0))
                f_to_c = csr.down(fine_t).transpose(1, 2)
                Lc = h_coarse.shape[1]
                if f_to_c.shape[1] < Lc:
                    f_to_c = F.pad(f_to_c, (0, 0, 0, Lc - f_to_c.shape[1]))
                gate_f2c_out = csr.gate_f2c(f_to_c[:, :Lc, :])
                check_nan(f"Block {i} CSR {s} gate_f2c_out", gate_f2c_out)
                h_coarse = h_coarse + gate_f2c_out * f_to_c[:, :Lc, :]
                
                c_to_f = csr.up(h_coarse.transpose(1, 2)).transpose(1, 2)
                Lf = h_fine.shape[1]
                if c_to_f.shape[1] < Lf:
                    c_to_f = F.pad(c_to_f, (0, 0, 0, Lf - c_to_f.shape[1]))
                gate_c2f_out = csr.gate_c2f(c_to_f[:, :Lf, :])
                check_nan(f"Block {i} CSR {s} gate_c2f_out", gate_c2f_out)
                h_fine = h_fine + gate_c2f_out * c_to_f[:, :Lf, :]
                strata[s], strata[s+1] = h_fine, h_coarse
                
            # FFN
            for s in range(b.S):
                ffn = b.ffns[s]
                x_in_ffn = strata[s]
                res = x_in_ffn
                x_n_ffn = ffn.norm(x_in_ffn)
                gate_val = (F.silu(ffn.w1(x_n_ffn)) * ffn.w3(x_n_ffn))
                check_nan(f"Block {i} FFN {s} raw_gate_val", gate_val)
                gate_val = gate_val.clamp(min=-6e4, max=6e4)
                strata[s] = res + ffn.dropout(ffn.w2(gate_val))
                check_nan(f"Block {i} FFN {s} output", strata[s])
            
            fine_acc = (fine_acc + strata[0].float()).clamp(min=-6e4, max=6e4)
            check_nan(f"Block {i} fine_acc", fine_acc)

        # Head mapping
        with torch.amp.autocast(device_type=device_type, enabled=False):
            logits = model.head(model.norm_out(strata[0].float()))
            check_nan("logits", logits)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            check_nan("loss", loss)

    print(f'Step {step} FWD OK')
    scaler.scale(loss).backward()
    
    # Check grads
    for n, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            if not torch.isfinite(torch.tensor(gn)):
                print(f"NaN/Inf grad in {n}")
                exit(1)
                
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    print(f'Step {step} DONE')
print('ALL OK')
