import torch
import torch.nn.functional as F
from laminarnet.model import LaminarNet, LaminarNetConfig

def test_amp():
    config = LaminarNetConfig(d_model=64, n_heads=2, n_layers=2, d_ff=128, n_strata=2)
    model = LaminarNet(config)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for s in range(5):
        x = torch.randint(0, config.vocab_size, (2, 512))
        y = torch.randint(0, config.vocab_size, (2, 512))

        opt.zero_grad(set_to_none=True)
        # Using bfloat16 on CPU to simulate AMP range issues
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = model(x)
            loss = F.cross_entropy(out.view(-1, config.vocab_size), y.view(-1))
            
        print(f"Step {s} Loss: {loss.item()}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("FOUND NAN/INF LOSS!")
            break
            
        loss.backward()
        
        has_nan = False
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                print(f"NaN/Inf gradient in {name}")
                has_nan = True
                break
                
        if has_nan:
            print("Stopped due to grad NaNs/Infs.")
            break
        
        opt.step()

if __name__ == "__main__":
    test_amp()
