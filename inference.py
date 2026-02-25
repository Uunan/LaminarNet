"""
LaminarNet vs Transformer — Metin Üretim Karşılaştırma Testi
LaminarNet: Recurrent inference (step) kullanır — O(1) per token
Transformer: Standart forward kullanır — O(N) per token
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from dataclasses import dataclass
from transformers import AutoTokenizer

torch.set_float32_matmul_precision('high')

try:
    from laminarnet import LaminarNet, LaminarNetConfig
except ImportError:
    os.system("pip install laminarnet==0.6.3")
    from laminarnet import LaminarNet, LaminarNetConfig

# -----------------------------------------------------------------------------
# Ayarlar
# -----------------------------------------------------------------------------
VOCAB_SIZE = 50257
SEQ_LEN    = 2048
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR   = "/content/drive/MyDrive/LaminarNet_Bench"

# Test promptları
PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time in a distant land,",
    "Science has proven that the universe",
    "In the year 2050, humanity will",
    "The most important discovery in history was",
]

MAX_NEW_TOKENS = 150
TEMPERATURE    = 0.8
TOP_K          = 50

# -----------------------------------------------------------------------------
# Transformer Baseline (Benchmark ile aynı mimari)
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
# Transformer generate — forward() ile (standart)
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_forward(model, tokenizer, prompt, max_new_tokens=150,
                     temperature=0.8, top_k=50):
    """Her adımda tüm diziyi forward() ile işler — Transformer için."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors=None)
    ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    t0 = time.time()
    for _ in range(max_new_tokens):
        idx_cond = ids[:, -SEQ_LEN:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

        if next_id.item() == tokenizer.eos_token_id:
            break

    dt = time.time() - t0
    generated_tokens = ids.shape[1] - len(input_ids)
    speed = generated_tokens / dt
    text = tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)
    return text, generated_tokens, dt, speed

# -----------------------------------------------------------------------------
# LaminarNet generate — step() ile (recurrent, O(1) per token)
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_recurrent(model, tokenizer, prompt, max_new_tokens=150,
                       temperature=0.8, top_k=50):
    """
    Recurrent inference: init_state() + step() kullanır.
    Her yeni token O(1) — tüm diziyi tekrar işlemez.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors=None)

    # State başlat
    state = model.init_state(batch_size=1, device=DEVICE)

    t0 = time.time()

    # 1) Prompt tokenlarını stateʼe besle (prefill)
    for tok_id in input_ids:
        token = torch.tensor([tok_id], dtype=torch.long, device=DEVICE)
        logits, state = model.step(token, state)

    # 2) Autoregressive üretim — her adım O(1)
    all_ids = list(input_ids)
    for _ in range(max_new_tokens):
        scaled_logits = logits / temperature

        if top_k > 0:
            v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
            scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(scaled_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (1,)
        all_ids.append(next_id.item())

        if next_id.item() == tokenizer.eos_token_id:
            break

        # Sonraki step — sadece 1 token işleniyor
        logits, state = model.step(next_id, state)

    dt = time.time() - t0
    generated_tokens = len(all_ids) - len(input_ids)
    speed = generated_tokens / dt
    text = tokenizer.decode(all_ids, skip_special_tokens=True)
    return text, generated_tokens, dt, speed

# -----------------------------------------------------------------------------
# Ana Test Fonksiyonu
# -----------------------------------------------------------------------------
def run_test():
    from google.colab import drive
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount('/content/drive')

    print("=" * 70)
    print("  LaminarNet vs Transformer — Metin Üretim Testi")
    print("  LaminarNet: Recurrent step() | Transformer: forward()")
    print("=" * 70)

    # Tokenizer
    print("\n📦 Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    # ── Transformer Yükleme ──────────────────────────────────────────────
    trans_path = os.path.join(BASE_DIR, "transformer_latest.pt")
    print(f"\n🔵 Transformer yükleniyor: {trans_path}")
    t_conf = TransformerConfig()
    transformer = TransformerBaseline(t_conf).to(DEVICE)
    transformer.load_state_dict(torch.load(trans_path, map_location=DEVICE, weights_only=True))
    transformer.eval()
    t_params = sum(p.numel() for p in transformer.parameters()) / 1e6
    print(f"   ✅ Yüklendi ({t_params:.1f}M parametre)")

    # ── LaminarNet Yükleme ───────────────────────────────────────────────
    lam_path = os.path.join(BASE_DIR, "laminarnet_latest.pt")
    print(f"\n🔴 LaminarNet yükleniyor: {lam_path}")
    l_conf = LaminarNetConfig(
        vocab_size=VOCAB_SIZE, d_model=320, n_heads=5, n_layers=10,
        d_ff=1200, n_strata=2, strata_ratios=(1, 2, 4), seq_len=SEQ_LEN, dropout=0.1
    )
    laminarnet = LaminarNet(l_conf).to(DEVICE)
    laminarnet.load_state_dict(torch.load(lam_path, map_location=DEVICE, weights_only=True))
    laminarnet.eval()
    l_params = sum(p.numel() for p in laminarnet.parameters()) / 1e6
    print(f"   ✅ Yüklendi ({l_params:.1f}M parametre)")

    # ── Her Prompt İçin Test ─────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  🧪 {len(PROMPTS)} Prompt Test Ediliyor")
    print(f"  Temperature={TEMPERATURE} | Top-K={TOP_K} | Max Tokens={MAX_NEW_TOKENS}")
    print(f"{'=' * 70}")

    total_speed_t, total_speed_l = 0, 0

    for i, prompt in enumerate(PROMPTS):
        print(f"\n{'─' * 70}")
        print(f"  📝 Prompt {i+1}: \"{prompt}\"")
        print(f"{'─' * 70}")

        # Transformer üretimi (forward)
        text_t, tokens_t, time_t, speed_t = generate_forward(
            transformer, tokenizer, prompt, MAX_NEW_TOKENS, TEMPERATURE, TOP_K
        )
        total_speed_t += speed_t

        # LaminarNet üretimi (recurrent step)
        text_l, tokens_l, time_l, speed_l = generate_recurrent(
            laminarnet, tokenizer, prompt, MAX_NEW_TOKENS, TEMPERATURE, TOP_K
        )
        total_speed_l += speed_l

        print(f"\n  🔵 Transformer ({tokens_t} token, {time_t:.2f}s, {speed_t:.1f} tok/s):")
        print(f"  {text_t}")
        print(f"\n  🔴 LaminarNet  ({tokens_l} token, {time_l:.2f}s, {speed_l:.1f} tok/s):")
        print(f"  {text_l}")

    # ── Özet ─────────────────────────────────────────────────────────────
    avg_speed_t = total_speed_t / len(PROMPTS)
    avg_speed_l = total_speed_l / len(PROMPTS)

    print(f"\n\n{'=' * 70}")
    print(f"  🏆 SONUÇ ÖZETİ")
    print(f"{'=' * 70}")
    print(f"  🔵 Transformer  Avg Speed: {avg_speed_t:.1f} tok/s (forward — O(N) per token)")
    print(f"  🔴 LaminarNet   Avg Speed: {avg_speed_l:.1f} tok/s (step   — O(1) per token)")
    if avg_speed_l > avg_speed_t:
        ratio = avg_speed_l / avg_speed_t
        print(f"\n  ⚡ LaminarNet recurrent inference {ratio:.1f}x daha hızlı!")
    else:
        ratio = avg_speed_t / avg_speed_l
        print(f"\n  ⚡ Transformer {ratio:.1f}x daha hızlı (kısa dizilerde normal)")
        print(f"  💡 Dizi uzadıkça LaminarNet avantajı artar (O(1) vs O(N²))")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    run_test()
