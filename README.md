# LaminarNet: Structured Orthogonal State Space Sequence Model

<div align="center">

![LaminarNet Logo](https://via.placeholder.com/800x200.png?text=LaminarNet)

**A Next-Generation Neural Architecture for Long-Context Sequence Modeling**

[![PyPI version](https://badge.fury.io/py/laminarnet.svg)](https://badge.fury.io/py/laminarnet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🌊 Overview

**LaminarNet** is a novel deep learning architecture designed to overcome the limitations of traditional Transformers in handling long sequences. By fusing principles from State Space Models (SSMs), Recurrent Neural Networks (RNNs), and Hierarchical Processing, LaminarNet achieves $O(N)$ inference complexity while maintaining the parallel training benefits of Transformers.

At its core, LaminarNet introduces three groundbreaking mechanisms:

1.  **Geometric Drift Fields (GDF)**: Parallelizable rotation-based state evolution that replaces standard RNN loops with prefix sums.
2.  **Cross-Stratum Routing (CSR)**: A bidirectional, multi-resolution information exchange system that allows different layers of abstraction to communicate effectively.
3.  **Phase Mesh Encoding (PME)**: A stabilized coupled oscillator system for position encoding, offering superior extrapolation capabilities compared to Rotary Embeddings (RoPE).

## 🚀 Key Features

*   **Linear Complexity**: Scales linearly with sequence length during inference, making it ideal for long-context applications.
*   **Hierarchical Structure**: Processes information at multiple resolutions (strata) simultaneously, capturing both local details and global context.
*   **Parallel Training**: Utilizes prefix scan algorithms to enable efficient parallel training on GPUs.
*   **Stabilized Dynamics**: Incorporates gated retention and phase clipping to ensure stable training dynamics even at depth.


## 💡 Motivation: Why LaminarNet?

The core limitation of the Transformer architecture is its quadratic **$O(N^2)$** complexity with respect to sequence length $N$. This makes scaling to context windows of 1M+ tokens computationally prohibitive.

LaminarNet introduces a **Structured Orthogonal State Space** mechanism that evolves states via **Geometric Drift Fields (GDF)**. This allows the model to:

1.  Maintain **$O(N)$ linear inference complexity**.
2.  Preserve long-range dependencies without the "forgetting" typical of LSTMs.
3.  Train in parallel like a Transformer using prefix-scan algorithms.

This positions LaminarNet as a bridge between the parallelizability of Transformers and the efficiency of RNNs/SSMs.

## 📐 Architecture

LaminarNet processes data through multiple hierarchical layers called **Strata**. Information flows both sequentially (time) and vertically (depth/resolution) via **Cross-Stratum Routing (CSR)**.

```mermaid
graph TD
    Input[Input Sequence] --> Embed[Embedding + Phase Mesh]
    Embed --> S1[Stratum 1 (Fine / Detail)]
    Embed --> S2[Stratum 2 (Coarse / Global)]
    
    subgraph "Laminar Block"
        S1 -- CSR --> S2
        S2 -- CSR --> S1
        
        S1 -- GDF (Time) --> S1
        S2 -- GDF (Time) --> S2
        
        S1 -- Local Mixing --> S1
        S2 -- Local Mixing --> S2
    end
    
    S1 --> Output[Output Head]
```

## 📦 Installation

Install LaminarNet directly from PyPI:

```bash
pip install laminarnet
```

## 🛠️ Usage

### Basic Inference

```python
import torch
from laminarnet import LaminarNet, LaminarNetConfig

# Initialize configuration
config = LaminarNetConfig(
    vocab_size=32000,
    d_model=256,
    n_layers=6,
    seq_len=2048,
    n_strata=2
)

# Create model
model = LaminarNet(config)

# Forward pass
inputs = torch.randint(0, 32000, (1, 128))  # (Batch, SeqLen)
logits = model(inputs)

print(f"Output shape: {logits.shape}")
# Output: torch.Size([1, 128, 32000])
```

### ⚡ Recurrent Inference (New in v0.6.3)

Use `step()` for fast, token-by-token autoregressive generation — no need to reprocess the full sequence at every step.

```python
import torch
from laminarnet import LaminarNet, LaminarNetConfig

config = LaminarNetConfig(vocab_size=32000, d_model=256, n_layers=6, n_strata=2)
model = LaminarNet(config)
model.eval()

# Initialize recurrent state
state = model.init_state(batch_size=1, device="cpu")

# Start with a prompt token
token = torch.tensor([1])  # BOS token

# Generate 50 tokens autoregressively
generated = [token.item()]
for _ in range(50):
    logits, state = model.step(token, state)  # (B, vocab_size)
    token = logits.argmax(dim=-1)              # greedy sampling
    generated.append(token.item())

print("Generated token IDs:", generated)
```

> **Note:** `step()` uses the exact same trained weights as `forward()`. No retraining needed!

### ✨ Training Loop Example

LaminarNet is a standard PyTorch `nn.Module`. You can train it with any optimizer and loss function.

```python
import torch.nn as nn
import torch.optim as optim

# Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Dummy Data
inputs = torch.randint(0, 32000, (4, 128)) # Batch of 4
targets = torch.randint(0, 32000, (4, 128))

# Training Step
model.train()
optimizer.zero_grad()

# Forward
logits = model(inputs) # (B, N, Vocab)

# Calculate Loss (Flatten for CrossEntropy)
loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))

# Backward & Step
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
```


### Advanced Configuration

You can customize the depth, width, and hierarchical structure of the network:

```python
config = LaminarNetConfig(
    d_model=512,
    n_heads=8,
    n_layers=12,
    d_ff=2048,
    n_strata=3,            # 3 levels of hierarchy
    strata_ratios=(1, 4, 16), # Compression ratios for each stratum
    n_oscillators=32,      # Number of phase oscillators
    dropout=0.1
)
```

## 🔬 Architecture Deep Dive

### 1. Geometric Drift Fields (GDF)
Unlike traditional RNNs that rely on matrix multiplications for state updates, GDF employs a rotation-based mechanism in a high-dimensional vector space. By using parallel prefix sums (cumulative rotations), GDF achieves the sequential modeling power of an RNN with the parallel efficiency of a Transformer.

```python
# Conceptual GDF Update
theta = tanh(Project(x))
state = state * rotation(theta) + input
```

### 2. Cross-Stratum Routing (CSR)
CSR enables information to flow between "fast" strata (processing high-frequency details) and "slow" strata (processing global context). This mimics the biological plausibility of neural oscillation coupling in the brain.

### 3. Phase Mesh Encoding (PME)
Replacing static positional embeddings, PME uses a system of coupled oscillators to encode position as a dynamic state. This allows the model to generalize to sequence lengths far beyond those seen during training.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Developed with ❤️ by Unan
</div>
