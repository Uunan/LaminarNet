import torch
from laminarnet.model import LaminarNet, LaminarNetConfig

def test_nan_4096():
    print("Testing N=4096 for NaNs...")
    conf = LaminarNetConfig(
        vocab_size=1000, d_model=320, n_heads=5, n_layers=4,
        d_ff=1200, n_strata=2, strata_ratios=(1, 2), seq_len=8192, dropout=0.0
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LaminarNet(conf).to(device)
    model.eval()
    
    # 4096 token test
    x = torch.randint(0, conf.vocab_size, (2, 4096), device=device)
    
    with torch.no_grad():
        out = model(x)
        
    if torch.isnan(out).any():
        print("❌ FAIL: NaNs detected in output!")
        # Let's see where the NaNs are
        nan_indices = torch.nonzero(torch.isnan(out))
        print(f"NaNs found at indices: {nan_indices[:10]}")
    else:
        print("✅ PASS: No NaNs detected at N=4096!")
        print(f"Output shape: {out.shape}")
        
if __name__ == "__main__":
    test_nan_4096()
