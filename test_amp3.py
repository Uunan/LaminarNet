import torch
import torch.nn.functional as F
from laminarnet.model import LaminarNet, LaminarNetConfig

def test_amp_fp16():
    config = LaminarNetConfig(d_model=64, n_heads=2, n_layers=1, d_ff=128, n_strata=2)
    model = LaminarNet(config)
    model.train()

    # Manual forward simulation injecting FP16 downcasts
    x_test = torch.randint(0, config.vocab_size, (2, 256))
    
    # Let's run forward pass manually and inspect max values
    x = model.tok_emb(x_test).half()
    
    # Pass through block 0
    b0 = model.blocks[0]
    
    # 1. Norm
    x_norm = b0.norm1(x.float()).half()
    
    # 2. GDF
    # we know GDF does a bunch of parallel scans. 
    # Let's inspect max values of fused projections.
    gdf = b0.gdfs[0]
    conv_out = gdf.conv1d(x_norm.transpose(1, 2).float())[..., :x_norm.shape[1]].transpose(1,2).half()
    conv_out = F.silu(conv_out.float()).half()
    
    fused = gdf.in_proj(conv_out.float()).half()
    dt_raw, v, gate = fused.chunk(3, dim=-1)
    
    dt = F.softplus(dt_raw.float() + gdf.dt_bias.float()).clamp(min=0.001, max=2.0)
    gate = torch.sigmoid(gate.float())
    log_gate = torch.log(gate + 1e-6)
    log_alpha = -dt + log_gate
    
    v_rotated = v.float()
    v_in = v_rotated * dt
    
    N = v_in.shape[1]
    
    # Now simulate the chunking logic
    chunk_size = 256
    L_chunks = torch.cumsum(log_alpha.double(), dim=2)
    L_max = L_chunks.max(dim=2, keepdim=True).values
    L_stable = L_chunks - L_max
    
    exp_neg_L_stable = torch.exp(-L_stable)
    scaled_v = exp_neg_L_stable * v_in.double()
    cum_scaled = torch.cumsum(scaled_v, dim=2)
    exp_L_stable = torch.exp(L_stable)
    
    chunk_out = (exp_L_stable * cum_scaled).half() # THIS is where fp16 fails?
    print(f"Max chunk_out value (fp16 range = 65500): {chunk_out.abs().max().item()}")
    
    print(f"Max cum_scaled value: {cum_scaled.abs().max().item()}")
    print(f"Max exp_L_stable value: {exp_L_stable.abs().max().item()}")

if __name__ == "__main__":
    test_amp_fp16()
