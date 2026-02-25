import torch
import torch.nn.functional as F
from laminarnet.model import LaminarNet, LaminarNetConfig

config = LaminarNetConfig(d_model=64, n_heads=2, n_layers=1, d_ff=128, n_strata=2)
model = LaminarNet(config).cuda()
model.train()

# 1. Float16 is the standard AMP type for T4/V100/A100 default
scaler = torch.amp.GradScaler('cuda')

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

steps = 20
for s in range(steps):
    x = torch.randint(0, config.vocab_size, (4, 1024)).cuda()
    y = torch.randint(0, config.vocab_size, (4, 1024)).cuda()

    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', dtype=torch.float16):
        out = model(x)
        loss = F.cross_entropy(out.view(-1, config.vocab_size), y.view(-1))
        
    print(f"Step {s} Loss: {loss.item()}")
    
    if torch.isnan(loss):
        print("FOUND NAN LOSS!")
        # Find which output caused it
        print("Out finite:", torch.isfinite(out).all().item())
        break
        
    scaler.scale(loss).backward()
    
    # Check gradients
    scaler.unscale_(opt)
    has_nan = False
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            print(f"NaN/Inf gradient in {name}")
            has_nan = True
            break
            
    if has_nan:
        print("Stopped due to grad NaNs/Infs.")
        break
    
    scaler.step(opt)
    scaler.update()

