"""
CNN Final Full - Revised
Waveform Classification (NO STFT)
Dataset:
- combined_5s.npy
- metadata_5s.npy
- data/splits_5s.npz

Revisi utama:
- Tidak menggunakan train_test_split().
- Split train/val/test di-load dari data/splits_5s.npz.
- Menggunakan LABEL_MAP kanonik.
- Normalisasi z-score per-trace gabungan 3 channel.
- Augmentasi time series valid:
  1) noise injection
  2) non-circular time shift dengan zero padding
  3) channel dropout hanya channel horizontal E/N
- Output training disimpan ke results/runs/<tanggal>_<nama_model>/.
- Menambahkan Post-Training Dynamic Quantization untuk model CNN.

Catatan:
- Path input dataset tetap menggunakan /home/indra/eq_team.
- Arsitektur CNN, hyperparameter, optimizer, scheduler, dan loss function dipertahankan.
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# ==========================================================
# 0 CONFIG
# ==========================================================
DATA_DIR = "/home/indra/eq_team"
PROJECT_DIR = "/home/indra/eq_team/earthquake-classification"

DATA_FILE = os.path.join(DATA_DIR, "combined_5s.npy")
META_FILE = os.path.join(DATA_DIR, "metadata_5s.npy")
SPLIT_FILE = os.path.join(PROJECT_DIR, "data", "splits_5s.npz")

MODEL_NAME = "CNN5s"
RUN_DATE = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(PROJECT_DIR, "results", "runs", f"{RUN_DATE}_{MODEL_NAME}")
CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
FIG_DIR = os.path.join(RUN_DIR, "figures")
REPORT_DIR = os.path.join(RUN_DIR, "reports")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-4
IMG_SIZE = 224

EARLY_STOPPING = True
PATIENCE = 5
MIN_DELTA = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEVICE == "cuda"

LABEL_MAP = {
    "memmap": "earthquake",
    "memmap_earthquake": "earthquake",
    "memmap_noise": "noise",
    "memmap_no_event": "no_event",
    "memmap_explosion": "explosion",
    "memmap_sonic": "sonic",
    "memmap_thunder": "thunder",
    "memmap_surface_event": "surface_event",
}

print("=" * 60)
print("CNN FINAL REVISED - 5s Waveform Classification")
print("=" * 60)
print("Device  :", DEVICE)
print("AMP     :", AMP)
print("Run dir :", RUN_DIR)


# ==========================================================
# 1 LOAD METADATA
# ==========================================================
print("\n[1] Loading metadata...")

meta = np.load(META_FILE, allow_pickle=True).item()
labels_raw = np.array(meta["label"])

print("Metadata keys      :", meta.keys())
print("Unique raw labels  :", np.unique(labels_raw))


# ==========================================================
# 2 LOAD MEMMAP
# ==========================================================
print("\n[2] Loading memmap waveform...")

shape = tuple(meta["shape"])
dtype = np.dtype(meta["dtype"])

print("Shape :", shape)
print("Dtype :", dtype)

data = np.memmap(
    DATA_FILE,
    dtype=dtype,
    mode="r",
    shape=shape,
)

N, C, T = shape
print("Total samples :", N)


# ==========================================================
# 3 LOAD SPLIT FROM NPZ
# ==========================================================
print("\n[3] Loading train/val/test split...")

split = np.load(SPLIT_FILE, allow_pickle=True)

train_idx = split["train"]
val_idx = split["val"]
test_idx = split["test"]
classes = list(split["classes"])

print("Split classes from file :", classes)
print("Train :", len(train_idx))
print("Val   :", len(val_idx))
print("Test  :", len(test_idx))


# ==========================================================
# 4 LABEL ENCODING WITH CANONICAL LABEL_MAP
# ==========================================================
print("\n[4] Encoding labels with canonical LABEL_MAP...")

labels_canonical = np.array([
    LABEL_MAP.get(str(label), str(label))
    for label in labels_raw
])

# Classes dari split tetap dibaca dari file, lalu dikanonisasi agar sesuai LABEL_MAP.
classes_canonical = []
for cls in classes:
    mapped_cls = LABEL_MAP.get(str(cls), str(cls))
    if mapped_cls not in classes_canonical:
        classes_canonical.append(mapped_cls)

# Jika split classes tidak lengkap karena ada penggabungan label, fallback ke label aktual.
for cls in np.unique(labels_canonical):
    if cls not in classes_canonical:
        classes_canonical.append(cls)

class_to_idx = {
    cls_name: idx
    for idx, cls_name in enumerate(classes_canonical)
}

labels = np.array([
    class_to_idx[label]
    for label in labels_canonical
], dtype=np.int64)

NUM_CLASSES = len(classes_canonical)

print("Unique labels after map :", np.unique(labels_canonical))
print("Final classes          :", classes_canonical)
print("Num classes            :", NUM_CLASSES)


# ==========================================================
# 5 AUGMENTATION HELPERS
# ==========================================================
def add_noise(x, noise_std=0.01):
    """Noise injection ringan untuk sinyal time series."""
    noise = torch.randn_like(x) * noise_std
    return x + noise


def non_circular_time_shift(x, max_shift=25):
    """
    Time shift non-circular menggunakan zero padding.
    Tidak memakai torch.roll karena roll membuat sinyal berputar ke sisi lain.
    Input x: (3, T)
    """
    shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())

    if shift == 0:
        return x

    shifted = torch.zeros_like(x)

    if shift > 0:
        shifted[:, shift:] = x[:, :-shift]
    else:
        k = abs(shift)
        shifted[:, :-k] = x[:, k:]

    return shifted


def horizontal_channel_dropout(x, dropout_prob=0.10):
    """
    Channel dropout hanya untuk channel horizontal.
    Asumsi urutan channel: E, N, Z sehingga channel 0 dan 1 adalah horizontal.
    Input x: (3, T)
    """
    if torch.rand(1).item() < dropout_prob:
        ch = int(torch.randint(0, 2, (1,)).item())
        x = x.clone()
        x[ch, :] = 0.0
    return x


def apply_time_series_augmentation(x):
    """Augmentasi valid untuk waveform time series."""
    if torch.rand(1).item() < 0.50:
        x = add_noise(x, noise_std=0.01)

    if torch.rand(1).item() < 0.50:
        x = non_circular_time_shift(x, max_shift=25)

    x = horizontal_channel_dropout(x, dropout_prob=0.10)
    return x


# ==========================================================
# 6 DATASET
# ==========================================================
class EqDataset(Dataset):

    def __init__(self, data, labels, indices, augment=False):
        self.data = data
        self.labels = labels
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        x = self.data[i].copy()

        # Z-score normalization per-trace gabungan 3 channel.
        mean = x.mean()
        std = x.std() + 1e-8
        x = (x - mean) / std

        x = torch.tensor(x, dtype=torch.float32)

        # Augmentasi dilakukan pada bentuk waveform asli (3, 500).
        if self.augment:
            x = apply_time_series_augmentation(x)

        # (3, 500) -> (1, 3, 500)
        x = x.unsqueeze(0)

        # Resize 1D: (1, 3, 500) -> (1, 3, 224)
        x = F.interpolate(
            x,
            size=IMG_SIZE,
            mode="linear",
            align_corners=False,
        )

        # (1, 3, 224) -> (3, 224)
        x = x.squeeze(0)

        # Pseudo-image 2D: (3, 224) -> (3, 224, 224)
        x = x.unsqueeze(-1).repeat(1, 1, IMG_SIZE)

        y = torch.tensor(
            self.labels[i],
            dtype=torch.long,
        )

        return x, y


# ==========================================================
# 7 DATALOADER
# ==========================================================
print("\n[5] Building dataloader...")

train_loader = DataLoader(
    EqDataset(data, labels, train_idx, augment=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    EqDataset(data, labels, val_idx, augment=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

test_loader = DataLoader(
    EqDataset(data, labels, test_idx, augment=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)


# ==========================================================
# 8 MODEL - CNN
# ==========================================================
print("\n[6] Building CNN...")


class CNN5s(nn.Module):
    def __init__(self, num_classes):
        super(CNN5s, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 56 -> 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 28 -> 14

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CNN5s(NUM_CLASSES).to(DEVICE)

total_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Trainable params :", f"{total_params:,}")


# ==========================================================
# 9 TRAIN SETUP
# ==========================================================
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=AMP,
)

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
}

best_acc = 0.0
patience_counter = 0

best_model_path = os.path.join(CKPT_DIR, "best_cnn_5s.pt")
last_model_path = os.path.join(CKPT_DIR, "last_cnn_5s.pt")
quantized_model_path = os.path.join(CKPT_DIR, "best_cnn_5s_dynamic_quantized.pt")
report_path = os.path.join(REPORT_DIR, "classification_report.txt")
plot_path = os.path.join(FIG_DIR, "cnn_5s_results.png")


def run_epoch(loader, train=True):
    if train:
        model.train()
        desc = "Train"
    else:
        model.eval()
        desc = "Val"

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=desc, leave=False)

    for xb, yb in pbar:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=AMP):
            preds = model(xb)
            loss = criterion(preds, yb)

        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * len(yb)
        pred_class = preds.argmax(1)
        correct += (pred_class == yb).sum().item()
        total += len(yb)

        pbar.set_postfix({
            "loss": f"{total_loss / total:.4f}",
            "acc": f"{correct / total:.4f}",
        })

    return total_loss / total, correct / total


# ==========================================================
# 10 TRAIN LOOP + EARLY STOPPING
# ==========================================================
print("\n[7] Training...")

for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)

    with torch.no_grad():
        vl_loss, vl_acc = run_epoch(val_loader, train=False)

    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)

    torch.save(model.state_dict(), last_model_path)

    if vl_acc > best_acc + MIN_DELTA:
        best_acc = vl_acc
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        marker = " <-- BEST"
    else:
        patience_counter += 1
        marker = f" | patience {patience_counter}/{PATIENCE}"

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
        f"Val Loss {vl_loss:.4f} Acc {vl_acc:.4f}"
        + marker
    )

    if EARLY_STOPPING and patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
        break


# ==========================================================
# 11 EVALUATION
# ==========================================================
print("\n[8] Evaluation...")

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=DEVICE,
    )
)

model.eval()

all_preds = []
all_true = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        pred = model(xb)
        pred = pred.argmax(1).cpu().numpy()

        all_preds.extend(pred)
        all_true.extend(yb.numpy())

all_preds = np.array(all_preds)
all_true = np.array(all_true)

report = classification_report(
    all_true,
    all_preds,
    target_names=classes_canonical,
)

print(report)

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)


# ==========================================================
# 12 POST-TRAINING DYNAMIC QUANTIZATION
# ==========================================================
print("\n[9] Applying Post-Training Dynamic Quantization...")

# Dynamic quantization PyTorch terutama diterapkan pada layer Linear.
# Pada CNN ini, bagian classifier memiliki Linear sehingga dapat dikompresi
# tanpa mengubah arsitektur training dan tanpa retraining.
model_cpu = CNN5s(NUM_CLASSES)
model_cpu.load_state_dict(torch.load(best_model_path, map_location="cpu"))
model_cpu.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model_cpu,
    {nn.Linear},
    dtype=torch.qint8,
)

torch.save(
    {
        "model_state_dict": quantized_model.state_dict(),
        "classes": classes_canonical,
        "label_map": LABEL_MAP,
        "num_classes": NUM_CLASSES,
        "img_size": IMG_SIZE,
        "note": "Post-Training Dynamic Quantization applied to nn.Linear layers.",
    },
    quantized_model_path,
)

print("Saved quantized model :", quantized_model_path)


# ==========================================================
# 13 PLOT
# ==========================================================
print("\n[10] Plotting...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history["train_loss"])
axes[0].plot(history["val_loss"])
axes[0].set_title("Loss")
axes[0].legend(["Train", "Val"])

axes[1].plot(history["train_acc"])
axes[1].plot(history["val_acc"])
axes[1].set_title("Accuracy")
axes[1].legend(["Train", "Val"])

cm = confusion_matrix(all_true, all_preds)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes_canonical,
    yticklabels=classes_canonical,
    ax=axes[2],
)

axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Pred")
axes[2].set_ylabel("True")

plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.show()


# ==========================================================
# 14 SAVE RUN CONFIG SUMMARY
# ==========================================================
summary_path = os.path.join(RUN_DIR, "run_summary.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("CNN5s Revised Training Summary\n")
    f.write("=" * 40 + "\n")
    f.write(f"Run dir: {RUN_DIR}\n")
    f.write(f"Data file: {DATA_FILE}\n")
    f.write(f"Meta file: {META_FILE}\n")
    f.write(f"Split file: {SPLIT_FILE}\n")
    f.write(f"Classes: {classes_canonical}\n")
    f.write(f"Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}\n")
    f.write(f"Best val acc: {best_acc:.6f}\n")
    f.write(f"Best model: {best_model_path}\n")
    f.write(f"Quantized model: {quantized_model_path}\n")
    f.write(f"Plot: {plot_path}\n")
    f.write(f"Report: {report_path}\n")

print("\nSaved plot       :", plot_path)
print("Saved report     :", report_path)
print("Saved summary    :", summary_path)
print("Saved best model :", best_model_path)
print("Saved last model :", last_model_path)
print("Saved quantized  :", quantized_model_path)
print("DONE")
