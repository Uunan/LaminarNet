"""
LaminarNet vs Transformer — Inference Hız Benchmark Grafiği
150 tokenden 8196 tokene kadar 150'şer artırarak her iki modelin
inference hızını (tok/s) ölçer ve karşılaştırma grafiği oluşturur.
LaminarNet: Recurrent step() — O(1) per token
Transformer: forward() — O(N) per token
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

torch.set_float32_matmul_precision('high')

try:
    from laminarnet import LaminarNet, LaminarNetConfig
except ImportError:
    os.system("pip install laminarnet==0.6.5")
    from laminarnet import LaminarNet, LaminarNetConfig

# -----------------------------------------------------------------------------
# Ayarlar
# -----------------------------------------------------------------------------
VOCAB_SIZE = 50257
SEQ_LEN    = 2048
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR   = "/content/drive/MyDrive/LaminarNet_Bench"
SAVE_DIR   = "/content/drive/MyDrive/LaminarNet_Bench"

TOKEN_STEPS = list(range(150, 8197, 150))  # 150, 300, 450, ..., 8100, (8196'yi da ekle)
if TOKEN_STEPS[-1] < 8196:
    TOKEN_STEPS.append(8196)

WARMUP_TOKENS = 10  # GPU warmup

# -----------------------------------------------------------------------------
# Transformer Baseline
# -----------------------------------------------------------------------------
@dataclass
class TransformerConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 416
    n_heads: int = 8
    n_layers: int = 10
    d_ff: int = 416 * 4
    seq_len: int = SEQ_LEN
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head  = config.d_model // config.n_heads
        self.qkv      = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout  = nn.Dropout(config.dropout)
        mask = torch.triu(torch.ones(config.seq_len, config.seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.causal_mask[:N, :N], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y   = (att @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.out_proj(y)

class SwiGLUFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.d_model)
        self.ffn  = SwiGLUFFN(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config  = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.blocks  = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f    = nn.LayerNorm(config.d_model)
        self.head    = nn.Linear(config.vocab_size, config.d_model, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x    = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

# -----------------------------------------------------------------------------
# Benchmark: Transformer forward() — her adımda tüm diziyi işler
# -----------------------------------------------------------------------------
@torch.no_grad()
def bench_transformer(model, num_tokens):
    """num_tokens kadar token üret, her adımda forward() çağır."""
    model.eval()
    ids = torch.randint(0, VOCAB_SIZE, (1, 1), device=DEVICE)

    # Warmup
    for _ in range(WARMUP_TOKENS):
        logits = model(ids[:, -SEQ_LEN:])
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)

    # Reset
    ids = torch.randint(0, VOCAB_SIZE, (1, 1), device=DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(num_tokens):
        idx_cond = ids[:, -SEQ_LEN:]
        logits = model(idx_cond)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return num_tokens / dt  # tok/s

# -----------------------------------------------------------------------------
# Benchmark: LaminarNet step() — O(1) per token
# -----------------------------------------------------------------------------
@torch.no_grad()
def bench_laminarnet(model, num_tokens):
    """num_tokens kadar token üret, her adımda step() çağır."""
    model.eval()
    state = model.init_state(batch_size=1, device=DEVICE)
    token = torch.randint(0, VOCAB_SIZE, (1,), device=DEVICE)

    # Warmup
    for _ in range(WARMUP_TOKENS):
        logits, state = model.step(token, state)
        token = logits.argmax(dim=-1)

    # Reset
    state = model.init_state(batch_size=1, device=DEVICE)
    token = torch.randint(0, VOCAB_SIZE, (1,), device=DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(num_tokens):
        logits, state = model.step(token, state)
        token = logits.argmax(dim=-1)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return num_tokens / dt  # tok/s

# -----------------------------------------------------------------------------
# Grafik Oluşturma
# -----------------------------------------------------------------------------
def create_plot(token_counts, trans_speeds, laminar_speeds, save_path):
    """Karşılaştırma grafiği oluştur ve kaydet."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("LaminarNet vs Transformer — Inference Speed Comparison",
                 fontsize=16, fontweight='bold', y=0.98)

    # ── Sol: tok/s karşılaştırma ─────────────────────────────────────────
    ax1.plot(token_counts, trans_speeds, 'b-o', linewidth=2, markersize=4,
             label='Transformer (forward)', alpha=0.9)
    ax1.plot(token_counts, laminar_speeds, 'r-s', linewidth=2, markersize=4,
             label='LaminarNet (step)', alpha=0.9)

    # Kesişim noktasını bul
    trans_arr = np.array(trans_speeds)
    laminar_arr = np.array(laminar_speeds)
    diff = laminar_arr - trans_arr
    crossover_indices = np.where(np.diff(np.sign(diff)))[0]
    if len(crossover_indices) > 0:
        ci = crossover_indices[0]
        cross_token = token_counts[ci]
        ax1.axvline(x=cross_token, color='green', linestyle='--', alpha=0.7,
                    label=f'Kesişim ~{cross_token} token')

    ax1.set_xlabel("Üretilen Token Sayısı", fontsize=13)
    ax1.set_ylabel("Hız (token/s)", fontsize=13)
    ax1.set_title("Inference Hızı", fontsize=14)
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(token_counts[0], token_counts[-1])

    # ── Sağ: Hız oranı ──────────────────────────────────────────────────
    ratios = []
    for ts, ls in zip(trans_speeds, laminar_speeds):
        if ts > 0:
            ratios.append(ls / ts)
        else:
            ratios.append(0)

    colors = ['green' if r >= 1.0 else 'red' for r in ratios]
    ax2.bar(range(len(ratios)), ratios, color=colors, alpha=0.7, width=0.8)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Eşit hız (1.0x)')
    ax2.set_xlabel("Test Noktası", fontsize=13)
    ax2.set_ylabel("LaminarNet / Transformer Hız Oranı", fontsize=13)
    ax2.set_title("Hız Oranı (>1 = LaminarNet daha hızlı)", fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # X-axis labels (her 5. noktayı göster)
    step = max(1, len(token_counts) // 10)
    tick_positions = list(range(0, len(token_counts), step))
    tick_labels = [str(token_counts[i]) for i in tick_positions]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n📊 Grafik kaydedildi: {save_path}")
    print(f"📊 PDF kaydedildi:    {save_path.replace('.png', '.pdf')}")
    plt.show()

# -----------------------------------------------------------------------------
# Ana Benchmark
# -----------------------------------------------------------------------------
def run_benchmark():
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount('/content/drive')
    except ImportError:
        pass # Not in colab

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 70)
    print("  LaminarNet vs Transformer — Inference Hız Benchmark")
    print(f"  Device: {DEVICE} | 150 → 8196 token (150 artış)")
    print("=" * 70)

    # ── Modelleri Yükle ──────────────────────────────────────────────────
    print("\n🔵 Transformer yükleniyor...")
    t_conf = TransformerConfig()
    transformer = TransformerBaseline(t_conf).to(DEVICE)
    trans_path = os.path.join(BASE_DIR, "transformer_latest.pt")
    if os.path.exists(trans_path):
        transformer.load_state_dict(torch.load(trans_path, map_location=DEVICE, weights_only=True))
        print("   -> Transformer weights loaded.")
    else:
        print("   -> Transformer weights not found, using random initialization for speed test.")
    transformer.eval()
    t_params = sum(p.numel() for p in transformer.parameters()) / 1e6
    print(f"   ✅ Transformer ({t_params:.1f}M parametre)")

    print("\n🔴 LaminarNet yükleniyor...")
    l_conf = LaminarNetConfig(
        vocab_size=VOCAB_SIZE, d_model=320, n_heads=5, n_layers=10,
        d_ff=1200, n_strata=2, strata_ratios=(1, 2, 4), seq_len=SEQ_LEN, dropout=0.1
    )
    laminarnet = LaminarNet(l_conf).to(DEVICE)
    lam_path = os.path.join(BASE_DIR, "laminarnet_latest.pt")
    if os.path.exists(lam_path):
        laminarnet.load_state_dict(torch.load(lam_path, map_location=DEVICE, weights_only=True))
        print("   -> LaminarNet weights loaded.")
    else:
        print("   -> LaminarNet weights not found, using random initialization for speed test.")
    laminarnet.eval()
    l_params = sum(p.numel() for p in laminarnet.parameters()) / 1e6
    print(f"   ✅ LaminarNet  ({l_params:.1f}M parametre)")

    # ── Benchmark Döngüsü ────────────────────────────────────────────────
    trans_speeds = []
    laminar_speeds = []
    total_tests = len(TOKEN_STEPS)

    print(f"\n{'=' * 70}")
    print(f"  🧪 {total_tests} nokta test ediliyor...")
    print(f"{'=' * 70}\n")

    print(f"  {'Token':>8}  {'Transformer':>14}  {'LaminarNet':>14}  {'Oran':>8}  {'Durum':>10}")
    print(f"  {'─' * 60}")

    for idx, n_tokens in enumerate(TOKEN_STEPS):
        # Transformer: seq_len aşarsa OOM olabilir, güvenli kontrol
        try:
            t_speed = bench_transformer(transformer, n_tokens)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            t_speed = 0.0
            torch.cuda.empty_cache()

        # LaminarNet: step() ile — bellek sabit, hiç OOM olmaz
        l_speed = bench_laminarnet(laminarnet, n_tokens)

        trans_speeds.append(t_speed)
        laminar_speeds.append(l_speed)

        ratio = l_speed / t_speed if t_speed > 0 else float('inf')
        winner = "🔴 Laminar" if ratio >= 1.0 else "🔵 Trans"

        print(f"  {n_tokens:>8}  {t_speed:>11.1f} t/s  {l_speed:>11.1f} t/s  {ratio:>6.2f}x  {winner}")

        # Her 10 adımda GPU belleği temizle
        if idx % 10 == 0 and DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ── Sonuçları Kaydet ─────────────────────────────────────────────────
    # CSV kaydet
    csv_path = os.path.join(SAVE_DIR, "inference_benchmark.csv")
    with open(csv_path, 'w') as f:
        f.write("tokens,transformer_tok_s,laminarnet_tok_s,ratio\n")
        for t, ts, ls in zip(TOKEN_STEPS, trans_speeds, laminar_speeds):
            ratio = ls / ts if ts > 0 else 0
            f.write(f"{t},{ts:.2f},{ls:.2f},{ratio:.4f}\n")
    print(f"\n📄 CSV kaydedildi: {csv_path}")

    # Grafik oluştur ve kaydet
    plot_path = os.path.join(SAVE_DIR, "inference_benchmark.png")
    create_plot(TOKEN_STEPS, trans_speeds, laminar_speeds, plot_path)

    # ── Özet ─────────────────────────────────────────────────────────────
    avg_t = np.mean(trans_speeds) if trans_speeds else 0
    avg_l = np.mean(laminar_speeds) if laminar_speeds else 0
    laminar_wins = sum(1 for ls, ts in zip(laminar_speeds, trans_speeds) if ls > ts)

    print(f"\n{'=' * 70}")
    print(f"  🏆 BENCHMARK SONUCU")
    print(f"{'=' * 70}")
    print(f"  🔵 Transformer  Ortalama: {avg_t:.1f} tok/s")
    print(f"  🔴 LaminarNet   Ortalama: {avg_l:.1f} tok/s")
    print(f"  📈 LaminarNet {laminar_wins}/{total_tests} noktada daha hızlı")

    if laminar_wins > 0:
        # İlk kesişim noktasını bul
        for i, (ls, ts) in enumerate(zip(laminar_speeds, trans_speeds)):
            if ls > ts:
                print(f"  ⚡ Kesişim noktası: ~{TOKEN_STEPS[i]} token")
                break
    print(f"{'=' * 70}")

if __name__ == "__main__":
    run_benchmark()
