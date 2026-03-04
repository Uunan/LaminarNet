# LaminarNet

<div align="center">

**A Structured Orthogonal State Space Sequence Model**

[![PyPI version](https://badge.fury.io/py/laminarnet.svg)](https://badge.fury.io/py/laminarnet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

</div>

---

## Overview

**LaminarNet** is a novel deep learning architecture designed for long-context sequence modeling. It achieves **O(N) inference complexity** by replacing the quadratic attention mechanism of Transformers with a structured, parallelizable state-space evolution called the **Geometric Drift Field (GDF)**.

The model processes information through multiple hierarchical resolution levels called **Strata**, which communicate via **Cross-Stratum Routing (CSR)**. Position is encoded using **Rotary Position Embeddings (RoPE)**, applied directly inside the GDF heads.

The entire architecture supports two execution modes on the **same trained weights**:
- **Parallel forward pass** — efficient training via chunked prefix scan
- **Recurrent step pass** — O(1) per-token inference via `step()` + `init_state()`

---

## Architecture

### High-Level Structure

```
Input Tokens
     │
     ▼
Token Embedding
     │
     ├──► Stratum 0 (Fine, ratio=1)    ──┐
     ├──► Stratum 1 (ratio=2)           │  Causal AvgPool
     └──► Stratum 2 (ratio=4, ...)      │
                                         │
     ┌────────────────────────────────── ┘
     │
     ▼  ×n_layers
┌─────────────────────────────────────────────┐
│               LaminarBlock                  │
│                                             │
│  1. GDF  — per-stratum temporal scan        │
│  2. CSR  — cross-resolution gated routing   │
│  3. FFN  — SwiGLU channel mixing            │
└─────────────────────────────────────────────┘
     │
     ▼
RMSNorm → LM Head (weight-tied to embedding)
     │
     ▼
  Logits
```

### Component 1 — Geometric Drift Field (GDF)

GDF is the core temporal operator of LaminarNet. It replaces attention with a selective, rotation-based state evolution that runs in **O(N)** via a chunked parallel prefix scan.

**Forward pass (parallel, training):**

```
x  →  Conv1D (depthwise, causal)  →  SiLU
   →  in_proj  →  [Δt | v | gate]

Δt  =  softplus(Δt_raw + dt_bias)        # selective time step
α   =  exp(−Δt)                          # decay per position
v   =  RoPE(v)                           # rotary position encoding

state[n] = α[n] * state[n−1] + Δt[n] * v[n]   # (parallelized via prefix scan)

out = out_proj(state * sigmoid(gate))
```

The parallelization is achieved via a **two-level chunked scan**:
1. Intra-chunk: parallel cumulative sum with log-domain stabilization
2. Inter-chunk: vectorized carry propagation across chunk boundaries

All operations are strictly causal — no future token information leaks into state.

**Recurrent pass (inference):**

The `step()` method uses the identical weights with an explicit `carry` buffer, executing in O(1) per token:

```python
alpha     = exp(−dt)
new_carry = alpha * carry + dt * RoPE(v, pos)
out       = out_proj(new_carry * gate)
```

### Component 2 — Cross-Stratum Routing (CSR)

CSR enables bidirectional information exchange between adjacent strata (resolution levels). It is applied once per LaminarBlock, after GDF and before the FFN.

```
Fine (Stratum s)   ────► causal AvgPool ────► gate_f2c ────► Coarse (Stratum s+1)
                                                                      │
Fine (Stratum s)   ◄─── Upsample (nearest) ◄─── gate_c2f ◄──────────┘
```

Both the downsampling and upsampling paths are gated with learned sigmoid projections:

```
h_coarse = h_coarse + σ(W_f2c · f_to_c) ⊙ f_to_c
h_fine   = h_fine   + σ(W_c2f · c_to_f) ⊙ c_to_f
```

The downsampling uses causal `AvgPool1d` (left-padded to prevent future leakage). In `step()` mode, routing degenerates to a simple gated residual since each stratum holds a single time step.

### Component 3 — SwiGLU FFN

Each stratum has its own feed-forward block using the SwiGLU activation:

```
FFN(x) = x + Dropout( W2 · (SiLU(W1 · x̂) ⊙ W3 · x̂) )
```
where `x̂ = RMSNorm(x)`. This is applied independently per stratum after CSR.

### Position Encoding — RoPE

Rotary Position Embeddings are applied inside GDF on the value vectors `v`, per head. The implementation supports both parallel mode (full sequence) and single-step mode (recurrent inference):

```python
# Parallel
cos, sin = rope(seq_len, device, dtype)    # (N, d_head//2)

# Recurrent
cos, sin = rope.forward_single(pos, ...)   # (1, d_head//2)
```

---

## Configuration

```python
@dataclass
class LaminarNetConfig:
    vocab_size:    int   = 50257   # vocabulary size
    d_model:       int   = 256     # model dimension
    n_heads:       int   = 8       # GDF heads (d_head = d_model // n_heads)
    n_layers:      int   = 8       # number of LaminarBlocks
    d_ff:          int   = 1024    # FFN hidden dimension
    n_strata:      int   = 2       # number of resolution levels
    strata_ratios: tuple = (1,2,4) # temporal compression ratios (must start with 1)
    seq_len:       int   = 1024    # training sequence length
    dropout:       float = 0.1     # dropout rate
    conv_kernel:   int   = 4       # depthwise conv kernel size inside GDF
    rope_base:     float = 10000.0 # RoPE frequency base
```

**Rules:**
- `strata_ratios[0]` must always be `1` (fine stratum, no compression)
- `len(strata_ratios) >= n_strata`
- All ratios must be positive integers

---

## Installation

```bash
pip install laminarnet
```

---

## Usage

### Parallel Forward (Training)

```python
import torch
from laminarnet import LaminarNet, LaminarNetConfig

config = LaminarNetConfig(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_layers=12,
    d_ff=2048,
    n_strata=2,
    strata_ratios=(1, 4),
)

model = LaminarNet(config)

x = torch.randint(0, 32000, (2, 1024))  # (batch, seq_len)
logits = model(x)                        # (2, 1024, 32000)
```

### Recurrent Inference (Generation)

```python
model.eval()
state = model.init_state(batch_size=1, device="cuda")

token = torch.tensor([1], device="cuda")  # BOS token

generated = [token.item()]
for _ in range(200):
    logits, state = model.step(token, state)  # O(1) per step
    token = logits.argmax(dim=-1)
    generated.append(token.item())
```

> `step()` uses the exact same weights as `forward()`. No retraining or fine-tuning needed.

### Training Loop

```python
import torch.nn as nn, torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

inputs  = torch.randint(0, 32000, (8, 512))
targets = torch.randint(0, 32000, (8, 512))

model.train()
optimizer.zero_grad()
logits = model(inputs)
loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))
loss.backward()
optimizer.step()
```

### Parameter Count

```python
info = model.count_parameters()
print(f"Trainable: {info['trainable']/1e6:.1f}M")
```

---

## Complexity

| Operation       | Transformer     | LaminarNet      |
|-----------------|-----------------|-----------------|
| Training FLOPs  | O(N² · D)       | O(N · D · log N)|
| Inference step  | O(N · D)        | **O(D)**        |
| KV Cache memory | O(N · D)        | **O(D)**        |
| Positional enc. | RoPE / ALiBi    | RoPE inside GDF |

---

## Module Reference

| Class | File | Description |
|---|---|---|
| `LaminarNetConfig` | `model.py` | Dataclass for all hyperparameters |
| `LaminarNet` | `model.py` | Top-level model: embedding, strata init, blocks, head |
| `LaminarBlock` | `model.py` | Single layer: GDF → CSR → FFN per stratum |
| `GeometricDriftField` | `model.py` | O(N) selective state-space operator with RoPE |
| `CrossStratumRouting` | `model.py` | Bidirectional gated cross-resolution routing |
| `SwiGLUFFN` | `model.py` | Per-stratum feed-forward block |
| `RotaryPositionEmbedding` | `model.py` | RoPE for parallel and single-step modes |
| `RMSNorm` | `model.py` | Root-mean-square normalization (FP32 stable) |

---

## Design Principles

- **Strict causality**: all temporal operations (conv, pooling, scan) are left-padded. No future token information leaks at any point.
- **Numerical stability**: log-domain stabilization in the parallel scan; RMSNorm computed in FP32 regardless of model dtype; dt clamped to `[0.001, 2.0]`.
- **Weight tying**: the LM head shares weights with the token embedding matrix, reducing parameter count.
- **AMP-safe**: all components are compatible with `torch.autocast` for mixed-precision training.

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
Developed by <a href="https://github.com/Uunan">Unan</a>
</div>
