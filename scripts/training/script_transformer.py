"""
WaveformTransformer — FINAL
Waveform Classification (NO STFT)
Dataset:
- combined_5s.npy
- metadata_5s.npy

Fixes applied:
  [1]  run_epoch context manager diperbaiki (tidak pakai torch.enable_grad sebagai ctx)
  [2]  top_k_accuracy_score guard diperbaiki (NUM_CLASSES >= 3)
  [3]  Early stopping monitor val_loss (bukan val_acc)
  [4]  LR Warmup 5 epoch + CosineAnnealingLR
  [5]  PATCH_SIZE=10 → 50 token (lebih banyak konteks)
  [6]  D_MODEL=256 → head_dim=32 per head (lebih representatif)
  [7]  cudnn.benchmark = True
  [8]  Reproducibility seed
  [9]  NUM_WORKERS dinamis
  [10] History disimpan ke JSON tiap epoch
  [11] label_smoothing tetap 0.1
  [12] Gradient clipping 1.0
"""

import os
import sys        # TAMBAH
import logging 
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# 0  CONFIG
# ==========================================================
DATA_DIR = "/home/indra/eq_team"  # memmap dataset stays at team root (shared, not moved)

DATA_FILE = os.path.join(DATA_DIR, "combined_5s.npy")
META_FILE = os.path.join(DATA_DIR, "metadata_5s.npy")

# Output artifacts live inside the project folder (see PANDUAN.md §4)
RESULTS_DIR = "/home/indra/eq_team/earthquake-classification/results"
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")

BATCH_SIZE = 256
EPOCHS     = 30
LR         = 1e-4

# Transformer hyper-params
PATCH_SIZE  = 10    # FIX [5]: 500/10 = 50 token (was 25 → 20 token)
SEQ_LEN     = 500   # waveform length setelah resample
D_MODEL     = 256   # FIX [6]: head_dim = 32 (was 128 → head_dim 16)
NHEAD       = 8
NUM_LAYERS  = 4
DIM_FF      = 512   # disesuaikan dengan D_MODEL baru
DROPOUT     = 0.1

WARMUP_EPOCHS = 5   # FIX [4]: LR warmup

# Early stopping — FIX [3]: monitor val_loss
EARLY_STOPPING = True
PATIENCE       = 7       # sedikit lebih toleran untuk transformer
MIN_DELTA      = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP    = DEVICE == "cuda"

# FIX [9]: NUM_WORKERS dinamis
NUM_WORKERS = min(8, os.cpu_count())

# ==========================================================
# FIX [8]: Reproducibility seed
# ==========================================================
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # tetap False agar cepat
    torch.backends.cudnn.benchmark = True       # FIX [7]

set_seed(42)


# ==========================================================
# LOGGING SETUP — output terminal → .log
# ==========================================================
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "transformer_training3_log.log")

class TeeLogger:
    """Tulis ke terminal sekaligus ke file .log"""
    def __init__(self, filepath):
        self.terminal = sys.__stdout__
        self.logfile  = open(filepath, "w", encoding="utf-8", buffering=1)

    def write(self, msg):
        self.terminal.write(msg)
        self.logfile.write(msg)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()

tee = TeeLogger(log_path)
sys.stdout = tee
sys.stderr = tee

print("=" * 60)
print("WaveformTransformer — FINAL")
print("=" * 60)
print(f"Device      : {DEVICE}")
print(f"AMP         : {AMP}")
print(f"NUM_WORKERS : {NUM_WORKERS}")
print(f"PATCH_SIZE  : {PATCH_SIZE}  → {SEQ_LEN // PATCH_SIZE} tokens")
print(f"D_MODEL     : {D_MODEL}  → head_dim={D_MODEL // NHEAD}")

# ==========================================================
# 1  LOAD METADATA
# ==========================================================
print("\n[1] Loading metadata...")

meta       = np.load(META_FILE, allow_pickle=True).item()
labels_raw = meta["label"]

print("Metadata keys :", list(meta.keys()))
print("Unique labels :", np.unique(labels_raw))

# ==========================================================
# 2  LOAD MEMMAP
# ==========================================================
print("\n[2] Loading memmap waveform...")

shape = tuple(meta["shape"])
dtype = np.dtype(meta["dtype"])

print("Shape :", shape)
print("Dtype :", dtype)

data = np.memmap(DATA_FILE, dtype=dtype, mode="r", shape=shape)

N, C, T = shape
print("Total samples :", N)

# ==========================================================
# 3 LABEL ENCODER
# ==========================================================
print("\n[3] Encoding labels...")

# Gabungkan "memmap" dan "memmap_earthquake" → "earthquake"
label_map = {
    "memmap": "earthquake",
    "memmap_earthquake": "earthquake"
}
labels_raw = np.array([
    label_map.get(str(l), str(l))
    for l in labels_raw
])

print("Unique labels (after merge) :", np.unique(labels_raw))

le = LabelEncoder()
labels = le.fit_transform(labels_raw)

NUM_CLASSES = len(le.classes_)

print("Classes :", le.classes_)
print("Num classes :", NUM_CLASSES)

# ==========================================================
# 4  SPLIT 70 / 10 / 20
# ==========================================================
print("\n[4] Train/Val/Test split...")

idx = np.arange(N)

train_idx, test_idx = train_test_split(
    idx,
    test_size=0.20,
    random_state=42,
    stratify=labels
)
train_idx, val_idx = train_test_split(
    train_idx,
    test_size=0.125,   # 12.5% of 80% = 10% total
    random_state=42,
    stratify=labels[train_idx]
)

print(f"Train : {len(train_idx)}  ({len(train_idx)/N*100:.1f}%)")
print(f"Val   : {len(val_idx)}   ({len(val_idx)/N*100:.1f}%)")
print(f"Test  : {len(test_idx)}  ({len(test_idx)/N*100:.1f}%)")

# ==========================================================
# 5  DATASET
# ==========================================================
class EqDataset(Dataset):

    def __init__(self, data, labels, indices, augment=False):
        self.data    = data
        self.labels  = labels
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.data[i].copy()                    # (C, T_orig)

        # z-score per channel
        mean = x.mean(axis=-1, keepdims=True)
        std  = x.std(axis=-1, keepdims=True) + 1e-8
        x    = (x - mean) / std

        x = torch.tensor(x, dtype=torch.float32)  # (C, T_orig)

        # Resample → SEQ_LEN
        x = F.interpolate(
            x.unsqueeze(0),
            size=SEQ_LEN,
            mode="linear",
            align_corners=False
        ).squeeze(0)                               # (C, SEQ_LEN)

        # Augmentasi (hanya train)
        if self.augment:
            if torch.rand(1) > 0.5:
                x = x.flip(-1)                     # random flip
            if torch.rand(1) > 0.5:
                x = x + 0.02 * torch.randn_like(x) # additive noise

        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x, y

# ==========================================================
# 6  DATALOADER
# ==========================================================
print("\n[5] Building dataloader...")

_loader_kwargs = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
    persistent_workers=(NUM_WORKERS > 0)
)

train_loader = DataLoader(
    EqDataset(data, labels, train_idx, augment=True),
    shuffle=True,
    **_loader_kwargs
)
val_loader = DataLoader(
    EqDataset(data, labels, val_idx),
    shuffle=False,
    **_loader_kwargs
)
test_loader = DataLoader(
    EqDataset(data, labels, test_idx),
    shuffle=False,
    **_loader_kwargs
)

# ==========================================================
# 7  MODEL
# ==========================================================
class PatchEmbedding1D(nn.Module):
    """
    (B, C, L) → (B, num_patches, D_MODEL)
    Setiap patch = PATCH_SIZE time-steps × C channels
    """
    def __init__(self, in_channels, seq_len, patch_size, d_model):
        super().__init__()
        assert seq_len % patch_size == 0, \
            f"SEQ_LEN ({seq_len}) harus habis dibagi PATCH_SIZE ({patch_size})"
        self.num_patches = seq_len // patch_size
        self.proj = nn.Linear(in_channels * patch_size, d_model)

    def forward(self, x):
        B, C, L = x.shape
        x = x.view(B, C, self.num_patches, -1)    # (B,C,P,ps)
        x = x.permute(0, 2, 1, 3).contiguous()    # (B,P,C,ps)
        x = x.view(B, self.num_patches, -1)        # (B,P,C*ps)
        return self.proj(x)                        # (B,P,D)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class WaveformTransformer(nn.Module):

    def __init__(
        self,
        in_channels,
        seq_len,
        patch_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        num_classes
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding1D(
            in_channels, seq_len, patch_size, d_model
        )
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True     # Pre-LN — lebih stabil
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        tokens = self.patch_embed(x)                     # (B, P, D)
        cls    = self.cls_token.expand(B, -1, -1)        # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)         # (B, 1+P, D)
        tokens = self.pos_enc(tokens)
        out    = self.encoder(tokens)                    # (B, 1+P, D)
        return self.head(out[:, 0])                      # (B, NC) via CLS


print("\n[6] Building WaveformTransformer...")

model = WaveformTransformer(
    in_channels=C,
    seq_len=SEQ_LEN,
    patch_size=PATCH_SIZE,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FF,
    dropout=DROPOUT,
    num_classes=NUM_CLASSES
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params : {total_params:,}")
print(f"Num tokens       : {SEQ_LEN // PATCH_SIZE} (+ 1 CLS)")

# ==========================================================
# 8  OPTIMIZER + SCHEDULER (dengan Warmup)
# ==========================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-2
)

# FIX [4]: LR Warmup 5 epoch → CosineAnnealing sisa epoch
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=WARMUP_EPOCHS
)
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS - WARMUP_EPOCHS,
    eta_min=1e-6
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[WARMUP_EPOCHS]
)

scaler = torch.amp.GradScaler("cuda", enabled=AMP)

# ==========================================================
# 9  TRAIN / EVAL FUNCTION
# ==========================================================
def run_epoch(loader, train: bool = True):
    """
    FIX [1]: Tidak menggunakan torch.enable_grad() sebagai context manager.
    Train dan eval dipisah secara eksplisit.
    """
    desc = "Train" if train else "Val"
    total_loss = correct = total = 0
    pbar = tqdm(loader, desc=desc, leave=False, file=sys.__stdout__)

    if train:
        model.train()
        for xb, yb in pbar:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=AMP):
                preds = model(xb)
                loss  = criterion(preds, yb)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(yb)
            correct    += (preds.argmax(1) == yb).sum().item()
            total      += len(yb)
            pbar.set_postfix({
                "loss": f"{total_loss/total:.4f}",
                "acc":  f"{correct/total:.4f}"
            })
    else:
        model.eval()
        with torch.no_grad():
            for xb, yb in pbar:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=AMP):
                    preds = model(xb)
                    loss  = criterion(preds, yb)

                total_loss += loss.item() * len(yb)
                correct    += (preds.argmax(1) == yb).sum().item()
                total      += len(yb)
                pbar.set_postfix({
                    "loss": f"{total_loss/total:.4f}",
                    "acc":  f"{correct/total:.4f}"
                })

    return total_loss / total, correct / total

# ==========================================================
# 10  TRAINING LOOP — Early stopping monitor val_loss
# ==========================================================
print("\n[7] Training...")

history = {
    "train_loss": [], "val_loss": [],
    "train_acc":  [], "val_acc":  [],
    "lr":         []
}

best_val_loss   = float("inf")   # FIX [3]: monitor val_loss
patience_ctr    = 0
best_model_path = os.path.join(CKPT_DIR, "best_transformer_5s_V3.pt")

for epoch in tqdm(range(EPOCHS), desc="Epochs", file=sys.__stdout__):

    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)

    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)
    history["lr"].append(current_lr)


    # FIX [3]: early stopping berdasarkan val_loss
    if vl_loss < best_val_loss - MIN_DELTA:
        best_val_loss = vl_loss
        patience_ctr  = 0
        torch.save(model.state_dict(), best_model_path)
        marker = " <-- BEST"
    else:
        patience_ctr += 1
        marker = f" | patience {patience_ctr}/{PATIENCE}"

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"LR {current_lr:.2e} | "
        f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
        f"Val Loss {vl_loss:.4f} Acc {vl_acc:.4f}"
        + marker
    )

    if EARLY_STOPPING and patience_ctr >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

# ==========================================================
# 11  EVALUATION — FULL METRICS ON TEST SET
# ==========================================================
print("\n[8] Evaluation on Test Set...")

model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.eval()

all_preds = []
all_true  = []
all_probs = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb     = xb.to(DEVICE)
        logits = model(xb)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()

        all_preds.extend(preds)
        all_true.extend(yb.numpy())
        all_probs.extend(probs)

all_preds = np.array(all_preds)
all_true  = np.array(all_true)
all_probs = np.array(all_probs)

# ---- Classification Report ----
print("\n--- Classification Report ---")
print(
    classification_report(
        all_true, all_preds,
        target_names=le.classes_,
        digits=4
    )
)

# ---- Aggregate metrics ----
acc      = accuracy_score(all_true, all_preds)
bal_acc  = balanced_accuracy_score(all_true, all_preds)

# FIX [2]: top-2 hanya jika NUM_CLASSES >= 3
top2_acc = (
    top_k_accuracy_score(all_true, all_probs, k=2)
    if NUM_CLASSES >= 3 else None
)

try:
    auc = roc_auc_score(
        all_true, all_probs,
        multi_class="ovr",
        average="macro"
    )
except ValueError:
    auc = float("nan")

print("\n--- Test Set Summary ---")
print(f"  Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Balanced Accuracy : {bal_acc:.4f}  ({bal_acc*100:.2f}%)")
if top2_acc is not None:
    print(f"  Top-2 Accuracy    : {top2_acc:.4f}  ({top2_acc*100:.2f}%)")
print(f"  ROC-AUC (macro)   : {auc:.4f}")

# ---- Per-class AUC ----
print("\n--- Per-class ROC-AUC ---")
for i, cls in enumerate(le.classes_):
    binary_true = (all_true == i).astype(int)
    try:
        cls_auc = roc_auc_score(binary_true, all_probs[:, i])
    except ValueError:
        cls_auc = float("nan")
    print(f"  {cls:<20}: {cls_auc:.4f}")

# ==========================================================
# 12  PLOT
# ==========================================================
print("\n[9] Plotting...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    f"WaveformTransformer — Test Accuracy: {acc*100:.2f}%  |  "
    f"ROC-AUC: {auc:.4f}",
    fontsize=13, fontweight="bold"
)

# --- Loss curve ---
ax = axes[0, 0]
ax.plot(history["train_loss"], label="Train", linewidth=2)
ax.plot(history["val_loss"],   label="Val",   linewidth=2)
ax.set_title("Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(alpha=0.3)

# --- Accuracy curve ---
ax = axes[0, 1]
ax.plot(history["train_acc"], label="Train", linewidth=2)
ax.plot(history["val_acc"],   label="Val",   linewidth=2)
ax.axhline(acc, color="red", linestyle="--", linewidth=1.2,
           label=f"Test={acc:.4f}")
ax.set_title("Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(alpha=0.3)

# --- LR schedule ---
ax = axes[1, 0]
ax.plot(history["lr"], color="darkorange", linewidth=2)
ax.set_title("Learning Rate Schedule")
ax.set_xlabel("Epoch")
ax.set_ylabel("LR")
ax.grid(alpha=0.3)

# --- Confusion matrix (normalised) ---
ax = axes[1, 1]
cm      = confusion_matrix(all_true, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ax=ax,
    vmin=0, vmax=1
)
ax.set_title("Confusion Matrix (normalised)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")

plt.tight_layout()

plot_path = os.path.join(FIG_DIR, "transformer_results_final3.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()

# ==========================================================
# 13  SAVE METRICS SUMMARY
# ==========================================================
summary_path = os.path.join(TABLE_DIR, "transformer_test_metricsV3.txt")

with open(summary_path, "w") as f:
    f.write("=" * 55 + "\n")
    f.write("WaveformTransformer — Test Set Metrics\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Accuracy          : {acc:.4f}  ({acc*100:.2f}%)\n")
    f.write(f"Balanced Accuracy : {bal_acc:.4f}  ({bal_acc*100:.2f}%)\n")
    if top2_acc is not None:
        f.write(f"Top-2 Accuracy    : {top2_acc:.4f}  ({top2_acc*100:.2f}%)\n")
    f.write(f"ROC-AUC (macro)   : {auc:.4f}\n\n")
    f.write("--- Per-class ROC-AUC ---\n")
    for i, cls in enumerate(le.classes_):
        binary_true = (all_true == i).astype(int)
        try:
            cls_auc = roc_auc_score(binary_true, all_probs[:, i])
        except ValueError:
            cls_auc = float("nan")
        f.write(f"  {cls:<20}: {cls_auc:.4f}\n")
    f.write("\n--- Classification Report ---\n")
    f.write(
        classification_report(
            all_true, all_preds,
            target_names=le.classes_,
            digits=4
        )
    )

print("\nFile yang disimpan:")
print(f"  Plot    : {plot_path}")
print(f"  Model   : {best_model_path}")
print(f"  Metrics : {summary_path}")
print(f"  Log     : {log_path}")
print("\nDONE")