import torch
from laminarnet.model import LaminarNet, LaminarNetConfig

def trace_nans():
    config = LaminarNetConfig(d_model=128, n_heads=4, n_layers=2, d_ff=256, n_strata=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LaminarNet(config).to(device)
    model.eval()

    x = torch.randint(0, config.vocab_size, (2, 512)).to(device)

    print(f"Tracing NaNs on {device}...")
    
    with torch.amp.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.bfloat16):
        x_emb = model.dropout(model.tok_emb(x))
        if torch.isnan(x_emb).any(): print("NaN in embeddings!"); return

        coarse_strata = []
        for pool in model.strata_init:
            x_t = x_emb.transpose(1, 2)
            k = pool.kernel_size[0] if isinstance(pool.kernel_size, tuple) else pool.kernel_size
            x_t = torch.nn.functional.pad(x_t, (k - 1, 0))
            coarse = pool(x_t).transpose(1, 2)
            coarse_strata.append(coarse)
        strata = [x_emb] + coarse_strata

        fine_acc = 0
        for i, b in enumerate(model.blocks):
            strata[0] = strata[0] + fine_acc * 0.5
            if torch.isnan(strata[0]).any(): print(f"NaN in Fine Accumulator before Block {i}"); return
            
            # GDFs
            strata[0] = b.gdfs[0](strata[0])
            if torch.isnan(strata[0]).any(): print(f"NaN in GDF 0 Block {i}"); return
            strata[1] = b.gdfs[1](strata[1])
            if torch.isnan(strata[1]).any(): print(f"NaN in GDF 1 Block {i}"); return
            
            # CSRs 2 passes
            for _ in range(2):
                strata[0], strata[1] = b.csrs[0](strata[0], strata[1])
            if torch.isnan(strata[0]).any() or torch.isnan(strata[1]).any(): print(f"NaN in CSR Block {i}"); return
            
            # FFNs
            strata[0] = b.ffns[0](strata[0])
            if torch.isnan(strata[0]).any(): print(f"NaN in FFN 0 Block {i}"); return
            strata[1] = b.ffns[1](strata[1])
            if torch.isnan(strata[1]).any(): print(f"NaN in FFN 1 Block {i}"); return
            
            if isinstance(fine_acc, int):
                fine_acc = strata[0]
            else:
                fine_acc = (fine_acc + strata[0]).clamp(min=-6e4, max=6e4)
            
        out = model.norm_out(strata[0])
        if torch.isnan(out).any(): print(f"NaN in Head Norm"); return
        
        logits = model.head(out)
        if torch.isnan(logits).any(): print(f"NaN in Output Logits"); return
        
        print("Forward pass clean! No NaNs found directly in the trace.")

if __name__ == "__main__":
    trace_nans()
