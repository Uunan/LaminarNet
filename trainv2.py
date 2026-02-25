"""
LaminarNet vs Transformer — State-of-the-Art Benchmark (Time-Aware)
- Packed token tensor kullanır (Padding yok, okuma hızı max)
- AMP (Mixed Precision) ve GradScaler ile hız & VRAM optimize edilmiştir
- İki modeli eşzamanlı eğitip doğruluğunu, hızını ve bellek tüketimini karşılaştırır
- Zamana dayalı asıl yarış: Transformer vs O(N) LaminarNet
- LaminarNet için x2 kat Learning Rate ölçeklemesi ayarlanmıştır
"""

import os
import sys
import time
import json
import math
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from dataclasses import dataclass

# Maksimum GPU tensor çekirdeği optimizasyonu (TF32)
torch.set_float32_matmul_precision('high')

# Try to import LaminarNet from the package
try:
    from laminarnet import LaminarNet, LaminarNetConfig
except ImportError:
    print("LaminarNet package not found. Installing...")
    os.system("pip install laminarnet==0.7.1")
    from laminarnet import LaminarNet, LaminarNetConfig

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
VOCAB_SIZE    = 50257
SEQ_LEN       = 2048
BATCH_SIZE    = 8
EPOCHS        = 1
LR_TRANS      = 3e-4       # Transformer Base LR
LR_LAM        = 6e-4       # LaminarNet (O(N) mimariler x2 veya x3 kat fazla LR sever)
VAL_INTERVAL  = 200
LOG_INTERVAL  = 10
SAVE_INTERVAL = 500
BASE_LOG_DIR  = "/content/drive/MyDrive/LaminarNet_Bench"
DATASET_PATH  = "/content/drive/MyDrive/FineWeb_Data/fineweb_10gb.jsonl"
PACKED_PATH   = "/content/drive/MyDrive/FineWeb_Data/packed_tokens_1b.pt"

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP       = torch.cuda.is_available()

# -----------------------------------------------------------------------------
# O(N^2) Transformer Baseline Mimari Özeti
# -----------------------------------------------------------------------------
@dataclass
class TransformerConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 256 * 4
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
        self.head    = nn.Linear(config.d_model, config.vocab_size, bias=False)
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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# -----------------------------------------------------------------------------
# Veriseti: Packed Tokens (Hızlı)
# -----------------------------------------------------------------------------
class PackedDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.data    = token_ids
        self.seq_len = seq_len
        self.n_seqs  = (len(token_ids) - 1) // seq_len

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]

def build_packed_tokens(file_path, tokenizer, max_tokens=1_000_000_000):
    EOS = tokenizer.eos_token_id
    all_tokens, total = [], 0
    print(f"📂 Tokenize ediliyor: {file_path}")
    with open(file_path, "rb") as f:
        for i, line in enumerate(f):
            if total >= max_tokens:
                break
            if i % 100000 == 0:
                print(f"   {i:,} satır | {total/1e6:.1f}M token")
            try:
                text = json.loads(line.decode("utf-8", errors="ignore")).get("text", "")
            except Exception:
                continue
            if not text.strip():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(ids)
            all_tokens.append(EOS)
            total += len(ids) + 1
    print(f"✅ {total/1e6:.1f}M token toplandı")
    return torch.tensor(all_tokens, dtype=torch.long)

def format_tokens(n):
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return str(n)

# -----------------------------------------------------------------------------
# Değerlendirme & Metrikler
# -----------------------------------------------------------------------------
def evaluate(model, dataloader, device, max_batches=50):
    model.eval()
    total_loss, total_tokens = 0, 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=USE_AMP):
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))  # Overflow shield
    model.train()
    return avg_loss, ppl

def save_csv(log_file, row_dict):
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def plot_results(log_file, out_dir):
    """CSV'den okur ve her iki modelin kıyaslamalı grafiklerini oluşturur (Wall-Clock Time eklentili)."""

    # Veri yapıları
    data = {'Transformer': {'steps': [], 'time': [], 'train_loss': [], 'val_loss': [], 'ppl': [], 'spd': [], 'mem': []},
            'LaminarNet':  {'steps': [], 'time': [], 'train_loss': [], 'val_loss': [], 'ppl': [], 'spd': [], 'mem': []}}

    with open(log_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['Model']
            data[model]['steps'].append(int(row['GlobalStep']))
            data[model]['time'].append(float(row['Wall_Clock_Sec']) / 3600.0) # Saate (Hour) çeviriyoruz
            data[model]['train_loss'].append(float(row['Train_Loss']))
            data[model]['val_loss'].append(float(row['Val_Loss']))
            data[model]['ppl'].append(float(row['Perplexity']))
            data[model]['spd'].append(float(row['Speed_tok_s']))
            data[model]['mem'].append(float(row['VRAM_GB']))

    t = data['Transformer']
    l = data['LaminarNet']

    fig = plt.figure(figsize=(24, 18))
    fig.suptitle('LaminarNet vs Transformer — State-of-the-Art Benchmark\n(Time & Step Scaled)', fontsize=18, fontweight='bold', y=0.95)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    def plot_panel(ax, x1, y1, x2, y2, title, xlabel, ylabel, log_scale=False):
        ax.plot(x1, y1, label='Transformer', color='#2196F3', linewidth=2)
        ax.plot(x2, y2, label='LaminarNet',  color='#FF5722', linewidth=2)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale('log')

    # SATIR 1: Adım (Step) Odaklı Klasik Grafikler
    plot_panel(fig.add_subplot(gs[0, 0]), t['steps'], t['train_loss'], l['steps'], l['train_loss'], 'Train Loss vs Steps', 'Global Step', 'Loss')
    plot_panel(fig.add_subplot(gs[0, 1]), t['steps'], t['val_loss'],   l['steps'], l['val_loss'],   'Val Loss vs Steps', 'Global Step', 'Loss')
    plot_panel(fig.add_subplot(gs[0, 2]), t['steps'], t['ppl'],        l['steps'], l['ppl'],        'Perplexity vs Steps', 'Global Step', 'PPL', log_scale=True)

    # SATIR 2: Zaman (Wall-Clock) Odaklı Gerçek Yarış Grafikleri (MAKALE İÇİN KRİTİK)
    plot_panel(fig.add_subplot(gs[1, 0]), t['time'], t['train_loss'], l['time'], l['train_loss'], 'Train Loss vs Wall-Clock Time', 'Time (Hours)', 'Loss')
    plot_panel(fig.add_subplot(gs[1, 1]), t['time'], t['val_loss'],   l['time'], l['val_loss'],   'Val Loss vs Wall-Clock Time', 'Time (Hours)', 'Loss')
    plot_panel(fig.add_subplot(gs[1, 2]), t['time'], t['ppl'],        l['time'], l['ppl'],        'Perplexity vs Wall-Clock Time', 'Time (Hours)', 'PPL', log_scale=True)

    # SATIR 3: Sistem Performansı ve Özet
    plot_panel(fig.add_subplot(gs[2, 0]), t['steps'], t['spd'], l['steps'], l['spd'], 'Throughput', 'Global Step', 'Tokens/sec')
    plot_panel(fig.add_subplot(gs[2, 1]), t['steps'], t['mem'], l['steps'], l['mem'], 'VRAM Usage', 'Global Step', 'GB')

    ax_tbl = fig.add_subplot(gs[2, 2])
    ax_tbl.axis('off')
    if t['train_loss'] and l['train_loss']:
        summary = [
            ['Metric', 'Transformer', 'LaminarNet'],
            ['Final Val Loss', f"{t['val_loss'][-1]:.3f}", f"{l['val_loss'][-1]:.3f}"],
            ['Final PPL',      f"{t['ppl'][-1]:.1f}",   f"{l['ppl'][-1]:.1f}"],
            ['Avg Speed (t/s)',f"{sum(t['spd'])/len(t['spd']):.0f}", f"{sum(l['spd'])/len(l['spd']):.0f}"],
            ['Avg VRAM (GB)',  f"{sum(t['mem'])/len(t['mem']):.2f}", f"{sum(l['mem'])/len(l['mem']):.2f}"],
            ['Total Time (hr)',f"{t['time'][-1]:.2f}", f"{l['time'][-1]:.2f}"]
        ]
        tbl = ax_tbl.table(cellText=summary[1:], colLabels=summary[0], loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1, 2)
        ax_tbl.set_title('Final Summary Report', fontweight='bold')

    out_path = os.path.join(out_dir, 'benchmark_results_time_aware.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# Ana Benchmark Döngüsü
# -----------------------------------------------------------------------------
def train_colab():
    # Drive mount
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount('/content/drive')
    except ImportError:
        pass

    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    log_file = os.path.join(BASE_LOG_DIR, f"benchmark_log_{int(time.time())}.csv")
    print(f"📁 Log dosyası oluşturuldu: {log_file}")

    # ── Token Tensor (Paketlenmiş Veri Yükleme) ──────────────────────────────
    if os.path.exists(PACKED_PATH):
        print(f"📦 Packed tensor yükleniyor: {PACKED_PATH}")
        all_tokens = torch.load(PACKED_PATH, map_location='cpu', weights_only=True)
        print(f"✅ {format_tokens(len(all_tokens))} token yüklendi")
    else:
        print("📦 Packed tensor bulunamadı, tokenize ediliyor (Bu biraz sürebilir)...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        all_tokens = build_packed_tokens(DATASET_PATH, tokenizer, max_tokens=25_000_000)
        torch.save(all_tokens, PACKED_PATH)
        print(f"💾 Paketlenmiş Olarak Kaydedildi: {PACKED_PATH}")

    # ── Train / Val Split ───────────────────────────────────────────────────
    split_idx    = int(0.95 * len(all_tokens))
    train_tokens = all_tokens[:split_idx]
    val_tokens   = all_tokens[split_idx:]

    train_ds = PackedDataset(train_tokens, SEQ_LEN)
    val_ds   = PackedDataset(val_tokens,   SEQ_LEN)

    # Stabilite ve Hız için Num_Workers=2 (Colab CPU için tatlı nokta)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2, drop_last=False)

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * EPOCHS
    tokens_per_step = BATCH_SIZE * SEQ_LEN
    print(f"\n📊 Toplam step: {total_steps} ({steps_per_epoch} step/epoch × {EPOCHS} epoch)")

    # ── Modellerin Kurulumu ─────────────────────────────────────────────────
    # Her ikisi de %100 eşit olacak (Yaklaşık 49.4M Parametre)
    t_conf = TransformerConfig(vocab_size=VOCAB_SIZE, d_model=416, n_layers=10, d_ff=416*4)
    l_conf = LaminarNetConfig(
        vocab_size=VOCAB_SIZE, d_model=320, n_heads=5, n_layers=10,
        d_ff=1200, n_strata=2, strata_ratios=(1, 2, 4), seq_len=SEQ_LEN, dropout=0.1
    )

    transformer = TransformerBaseline(t_conf).to(DEVICE)
    laminarnet  = LaminarNet(l_conf).to(DEVICE)

    print(f"\n🔢 Model Parametreleri Kapışması:")
    print(f"   Transformer Baseline : {transformer.count_parameters()/1e6:.1f} Milyon")
    print(f"   LaminarNet Model     : {laminarnet.count_parameters()/1e6:.1f} Milyon")

    # Mamba / RWKV Kuralı: O(N) modeller için Transformer'ın 2-3 katı LR kullan!
    print(f"\n🧠 Learning Rate Ayarları:")
    print(f"   Transformer LR : {LR_TRANS}")
    print(f"   LaminarNet LR  : {LR_LAM} (O(N) Mimarisi için x2 Artırıldı)")

    opt_t = torch.optim.AdamW(transformer.parameters(), lr=LR_TRANS, weight_decay=0.01)
    opt_l = torch.optim.AdamW(laminarnet.parameters(),  lr=LR_LAM, weight_decay=0.01) # Laminar daha yüksek LR ile eğitiliyor!

    scaler_t = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    scaler_l = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    start_time_t = 0.0 # Transformer Timer
    start_time_l = 0.0 # LaminarNet Timer

    print("\n🚀 Kıyasıya Benchmark Eğitimi Başlıyor...\n")

    for epoch in range(EPOCHS):
        transformer.train()
        laminarnet.train()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            # -------------------------------------------------------------
            # [1] Transformer Baseline Forward & Backward
            # -------------------------------------------------------------
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()

            opt_t.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                logits_t = transformer(x)
                loss_t   = criterion(logits_t.view(-1, VOCAB_SIZE), y.view(-1))

            scaler_t.scale(loss_t).backward()
            scaler_t.unscale_(opt_t)
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            scaler_t.step(opt_t)
            scaler_t.update()

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            dt_t  = time.time() - t0
            start_time_t += dt_t # SADECE Transformer'ın harcadığı süreyi biriktir

            spd_t = tokens_per_step / dt_t
            mem_t = torch.cuda.max_memory_allocated() / 1024**3 if DEVICE.type == "cuda" else 0.0

            # -------------------------------------------------------------
            # [2] LaminarNet Forward & Backward
            # -------------------------------------------------------------
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()

            opt_l.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                logits_l = laminarnet(x)
                loss_l   = criterion(logits_l.view(-1, VOCAB_SIZE), y.view(-1))

            scaler_l.scale(loss_l).backward()
            scaler_l.unscale_(opt_l)
            torch.nn.utils.clip_grad_norm_(laminarnet.parameters(), 1.0)
            scaler_l.step(opt_l)
            scaler_l.update()

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            dt_l  = time.time() - t0
            start_time_l += dt_l # SADECE LaminarNet'in harcadığı süreyi biriktir

            spd_l = tokens_per_step / dt_l
            mem_l = torch.cuda.max_memory_allocated() / 1024**3 if DEVICE.type == "cuda" else 0.0

            global_step += 1

            # ── Log Ekrana Yansıması ─────────────────────────────────────
            if global_step % LOG_INTERVAL == 0:
                progress = global_step / total_steps * 100
                print(f"St {global_step:5d} [{progress:4.1f}%] | "
                      f"Trans: L={loss_t.item():.2f} Spd={spd_t:.0f}t/s Mem={mem_t:.2f}GB || "
                      f"Laminar: L={loss_l.item():.2f} Spd={spd_l:.0f}t/s Mem={mem_l:.2f}GB")

            # ── Doğrulama & CSV Kayıt ─────────────────────────────────────
            if global_step % VAL_INTERVAL == 0:
                print("\n  🔍 Validation Checkpoint...")
                val_loss_t, ppl_t = evaluate(transformer, val_loader, DEVICE)
                val_loss_l, ppl_l = evaluate(laminarnet,  val_loader, DEVICE)

                # Transformer Kaydı (Ayrı Timer ile)
                save_csv(log_file, {
                    'Epoch': epoch + 1, 'GlobalStep': global_step, 'Model': 'Transformer',
                    'Train_Loss': f'{loss_t.item():.4f}', 'Val_Loss': f'{val_loss_t:.4f}',
                    'Perplexity': f'{ppl_t:.2f}', 'Speed_tok_s': f'{spd_t:.1f}', 'VRAM_GB': f'{mem_t:.3f}',
                    'Wall_Clock_Sec': f'{start_time_t:.2f}'  # Makale için Kritik Süre
                })
                # LaminarNet Kaydı (Ayrı Timer ile)
                save_csv(log_file, {
                    'Epoch': epoch + 1, 'GlobalStep': global_step, 'Model': 'LaminarNet',
                    'Train_Loss': f'{loss_l.item():.4f}', 'Val_Loss': f'{val_loss_l:.4f}',
                    'Perplexity': f'{ppl_l:.2f}', 'Speed_tok_s': f'{spd_l:.1f}', 'VRAM_GB': f'{mem_l:.3f}',
                    'Wall_Clock_Sec': f'{start_time_l:.2f}' # Makale için Kritik Süre
                })

                print(f"  📈 Trans   → Val Loss: {val_loss_t:.3f} | PPL: {ppl_t:.1f} | Time: {start_time_t/60:.1f}m")
                print(f"  📈 Laminar → Val Loss: {val_loss_l:.3f} | PPL: {ppl_l:.1f} | Time: {start_time_l/60:.1f}m")

                plot_results(log_file, BASE_LOG_DIR)
                print(f"  📊 Benchmark Time-Aware Grafiği Güncellendi! \n")

            if global_step % SAVE_INTERVAL == 0:
                torch.save(transformer.state_dict(), os.path.join(BASE_LOG_DIR, "transformer_latest.pt"))
                torch.save(laminarnet.state_dict(),  os.path.join(BASE_LOG_DIR, "laminarnet_latest.pt"))

    # Eğitim tam bitince Final Validation
    print("\n🏁 Eğitim Tamamlandı. Final Validation yapılıyor...")
    val_loss_t, ppl_t = evaluate(transformer, val_loader, DEVICE, max_batches=200)
    val_loss_l, ppl_l = evaluate(laminarnet,  val_loader, DEVICE, max_batches=200)

    print(f"\n🏆 FİNAL SONUÇLAR:")
    print(f"   Transformer → Val Loss: {val_loss_t:.4f} | PPL: {ppl_t:.2f}")
    print(f"   LaminarNet  → Val Loss: {val_loss_l:.4f} | PPL: {ppl_l:.2f}")

    plot_results(log_file, BASE_LOG_DIR)
    print("\n✅ Benchmark Tamamen Kuruldu ve Sonuçlandı!")
    print(f"   📊 Grafik: {os.path.join(BASE_LOG_DIR, 'benchmark_results_time_aware.png')}")

if __name__ == "__main__":
    train_colab()
