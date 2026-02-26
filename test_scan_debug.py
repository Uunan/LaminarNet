"""
Diagnose GDF scan: compare intra-chunk cumsum scan output directly with 
step-by-step recurrence to find exactly where they diverge.
"""
import torch
import torch.nn.functional as F
from laminarnet.model import GeometricDriftField, apply_rope

torch.manual_seed(42)

gdf = GeometricDriftField(d_model=64, n_heads=4, dropout=0.0, conv_kernel=4)
gdf.eval()

B, N, D = 1, 16, 64

x_in = torch.randn(B, N, D)

with torch.no_grad():
    # Replicate forward() internals
    residual = x_in
    x = gdf.norm(x_in)
    
    x_conv = gdf.conv1d(x.transpose(1, 2))[..., :N].transpose(1, 2)
    x_conv = F.silu(x_conv)
    
    fused = gdf.in_proj(x_conv)
    dt_raw, v, gate = fused.chunk(3, dim=-1)
    
    dt = F.softplus(dt_raw.float() + gdf.dt_bias.float()).clamp(min=0.001, max=2.0).to(dt_raw.dtype)
    gate_val = torch.sigmoid(gate)
    log_alpha = -dt.float()
    
    cos_f, sin_f = gdf.rope(N, device=x.device, dtype=x.dtype)
    v_view = v.view(B, N, gdf.n_heads, gdf.d_head)
    v_rotated = apply_rope(v_view, cos_f, sin_f).reshape(B, N, D)
    
    v_in = v_rotated * dt  # (B, N, D)
    
    # ===== FORWARD SCAN (stabilized cumsum) =====
    la = log_alpha.view(B, 1, N, D)
    v_c = v_in.float().view(B, 1, N, D)
    
    L = torch.cumsum(la, dim=2).clamp(min=-20.0, max=0.0)
    L_max = L.max(dim=2, keepdim=True).values
    L_stable = L - L_max
    exp_neg_L_stable = torch.exp(-L_stable)
    scaled_v = exp_neg_L_stable * v_c
    cum_scaled = torch.cumsum(scaled_v, dim=2)
    exp_L_stable = torch.exp(L_stable)
    scan_out = (exp_L_stable * cum_scaled).squeeze(1)  # (B, N, D)
    
    # ===== STEP-BY-STEP RECURRENCE =====
    carry = torch.zeros(B, D)
    step_outs = []
    for t in range(N):
        alpha_t = torch.exp(log_alpha[:, t, :])  # (B, D)
        carry = alpha_t * carry + v_in[:, t, :].float()
        step_outs.append(carry.clone())
    step_out = torch.stack(step_outs, dim=1)  # (B, N, D)
    
    # Compare
    print("Scan vs Step recurrence (raw, before talking heads):")
    for t in range(N):
        diff = (scan_out[:, t, :] - step_out[:, t, :]).abs().max().item()
        scan_max = scan_out[:, t, :].abs().max().item()
        step_max = step_out[:, t, :].abs().max().item()
        marker = " ⚠️" if diff > 1e-5 else ""
        print(f"  t={t:2d}: diff={diff:.8f}  scan_max={scan_max:.4f}  step_max={step_max:.4f}{marker}")
    
    print(f"\nOverall max diff: {(scan_out - step_out).abs().max().item():.8f}")
    
    # Also check L_max and L_stable values
    print(f"\nL_max (entire chunk): {L_max.item():.6f}")
    print(f"L range: [{L.min().item():.6f}, {L.max().item():.6f}]")
    print(f"L_stable range: [{L_stable.min().item():.6f}, {L_stable.max().item():.6f}]")
