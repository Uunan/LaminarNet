"""Final comprehensive forward vs step consistency test."""
import torch
from laminarnet.model import LaminarNet, LaminarNetConfig

def run_test(name, config, N):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    torch.manual_seed(42)
    model = LaminarNet(config)
    model.eval()
    B = 1
    ids = torch.randint(0, config.vocab_size, (B, N))
    
    with torch.no_grad():
        logits_fwd = model(ids)
    
    state = model.init_state(batch_size=B)
    step_list = []
    with torch.no_grad():
        for t in range(N):
            logit_t, state = model.step(ids[:, t], state)
            step_list.append(logit_t.unsqueeze(1))
    logits_step = torch.cat(step_list, dim=1)
    
    max_diff = (logits_fwd - logits_step).abs().max().item()
    mean_diff = (logits_fwd - logits_step).abs().mean().item()
    print(f"Max diff:  {max_diff:.8f}")
    print(f"Mean diff: {mean_diff:.8f}")
    ok = max_diff < 1e-3
    print(f"{'✅ PASS' if ok else '❌ FAIL'}")
    return ok

if __name__ == "__main__":
    results = []
    
    # Test 1: Single stratum, 1 layer
    results.append(run_test("1-layer, 1-stratum, N=16",
        LaminarNetConfig(d_model=64, n_heads=4, n_layers=1, d_ff=128,
                         n_strata=1, strata_ratios=(1, 2, 4), dropout=0.0), N=16))
    
    # Test 2: 2 strata, 2 layers (DenseNet + CSR)
    results.append(run_test("2-layer, 2-strata, N=16",
        LaminarNetConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128,
                         n_strata=2, strata_ratios=(1, 2, 4), dropout=0.0), N=16))
    
    # Test 3: Longer sequence
    results.append(run_test("1-layer, 1-stratum, N=64",
        LaminarNetConfig(d_model=64, n_heads=4, n_layers=1, d_ff=128,
                         n_strata=1, strata_ratios=(1, 2, 4), dropout=0.0), N=64))
    
    # Test 4: Larger model
    results.append(run_test("4-layer, 2-strata, N=32",
        LaminarNetConfig(d_model=128, n_heads=8, n_layers=4, d_ff=256,
                         n_strata=2, strata_ratios=(1, 2, 4), dropout=0.0), N=32))
    
    print(f"\n{'='*60}")
    print(f"OVERALL: {'✅ ALL PASS' if all(results) else '❌ SOME FAILED'}")
    print(f"{'='*60}")
