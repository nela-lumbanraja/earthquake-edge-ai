"""
MobileNetV2 Final Full
Waveform Classification (NO STFT)
Dataset:
- combined_5s.npy
- metadata_5s.npy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)

# ==========================================================
# 0 CONFIG
# ==========================================================
DATA_DIR = "/home/indra/eq_team"  # memmap dataset stays at team root (shared, not moved)

DATA_FILE = os.path.join(DATA_DIR, "combined_5s.npy")
META_FILE = os.path.join(DATA_DIR, "metadata_5s.npy")

# Output artifacts live inside the project folder (see PANDUAN.md §4)
RESULTS_DIR = "/home/indra/eq_team/earthquake-classification/results"
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-4
IMG_SIZE = 224

EARLY_STOPPING = True
PATIENCE = 5
MIN_DELTA = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AMP = DEVICE == "cuda"

print("="*60)
print("MobileNetV2 FINAL")
print("="*60)
print("Device :", DEVICE)
print("AMP    :", AMP)

# ==========================================================
# 1 LOAD METADATA
# ==========================================================
print("\n[1] Loading metadata...")

meta = np.load(META_FILE, allow_pickle=True).item()

labels_raw = meta["label"]

print("Metadata keys :", meta.keys())
print("Unique labels :", np.unique(labels_raw))

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
    shape=shape
)

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
# 4 SPLIT 70/10/20
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
    test_size=0.125,  # 10% total
    random_state=42,
    stratify=labels[train_idx]
)

print("Train :", len(train_idx))
print("Val   :", len(val_idx))
print("Test  :", len(test_idx))

# ==========================================================
# 5 DATASET
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

        # z-score
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-8
        x = (x - mean) / std

        x = torch.tensor(x, dtype=torch.float32)

        # (3,500) → (1,3,500)
        x = x.unsqueeze(0)

        # resize 1D dulu: (1,3,500) → (1,3,224)
        x = F.interpolate(
            x,
            size=IMG_SIZE,
            mode="linear",
            align_corners=False
        )

        # (1,3,224) → (3,224)
        x = x.squeeze(0)

        # ubah menjadi pseudo-image 2D
        # (3,224) → (3,224,224)
        x = x.unsqueeze(-1).repeat(1, 1, IMG_SIZE)

        # augment ringan
        if self.augment:
            if torch.rand(1) > 0.5:
                x = x.flip(-1)

        y = torch.tensor(
            self.labels[i],
            dtype=torch.long
        )

        return x, y

# ==========================================================
# 6 DATALOADER
# ==========================================================
print("\n[5] Building dataloader...")

train_loader = DataLoader(
    EqDataset(data, labels, train_idx, augment=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    EqDataset(data, labels, val_idx),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

test_loader = DataLoader(
    EqDataset(data, labels, test_idx),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

# ==========================================================
# 7 MODEL — MobileNetV2
# ==========================================================
print("\n[6] Building MobileNetV2...")

weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)

# MobileNetV2 classifier: Sequential(Dropout, Linear(1280, 1000))
# Replace the final Linear layer
in_features = model.classifier[1].in_features  # 1280
model.classifier[1] = nn.Linear(
    in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

total_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Trainable params :", f"{total_params:,}")

# ==========================================================
# 8 TRAIN SETUP
# ==========================================================
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=AMP
)

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

best_acc = 0
patience_counter = 0

best_model_path = os.path.join(
    CKPT_DIR,
    "best_mobilenetv2_5s.pt"
)

def run_epoch(loader, train=True):

    if train:
        model.train()
        desc = "Train"
    else:
        model.eval()
        desc = "Val"

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(
        loader,
        desc=desc,
        leave=False
    )

    for xb, yb in pbar:

        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(
            "cuda",
            enabled=AMP
        ):
            preds = model(xb)
            loss = criterion(preds, yb)

        if train:
            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * len(yb)

        pred_class = preds.argmax(1)

        correct += (
            pred_class == yb
        ).sum().item()

        total += len(yb)

        # update progress bar
        pbar.set_postfix({
            "loss": f"{total_loss/total:.4f}",
            "acc": f"{correct/total:.4f}"
        })

    return total_loss / total, correct / total

# ==========================================================
# 9 TRAIN LOOP + EARLY STOPPING
# ==========================================================
print("\n[7] Training...")

for epoch in tqdm(
    range(EPOCHS),
    desc="Epochs"
):

    tr_loss, tr_acc = run_epoch(
        train_loader,
        True
    )

    with torch.no_grad():
        vl_loss, vl_acc = run_epoch(
            val_loader,
            False
        )

    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)

    # =========================
    # SAVE BEST + EARLY STOP
    # =========================
    if vl_acc > best_acc + MIN_DELTA:

        best_acc = vl_acc
        patience_counter = 0

        torch.save(
            model.state_dict(),
            best_model_path
        )

        marker = " <-- BEST"

    else:

        patience_counter += 1

        marker = (
            f" | patience "
            f"{patience_counter}/{PATIENCE}"
        )

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Train Loss {tr_loss:.4f} "
        f"Acc {tr_acc:.4f} | "
        f"Val Loss {vl_loss:.4f} "
        f"Acc {vl_acc:.4f}"
        + marker
    )

    # =========================
    # EARLY STOPPING
    # =========================
    if (
        EARLY_STOPPING and
        patience_counter >= PATIENCE
    ):
        print(
            f"\nEarly stopping "
            f"triggered at "
            f"epoch {epoch+1}"
        )
        break

# ==========================================================
# 10 EVALUATION
# ==========================================================
print("\n[8] Evaluation...")

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=DEVICE
    )
)

model.eval()

all_preds = []
all_true = []

with torch.no_grad():

    for xb, yb in test_loader:

        xb = xb.to(DEVICE)

        pred = model(xb)
        pred = pred.argmax(1).cpu().numpy()

        all_preds.extend(pred)
        all_true.extend(yb.numpy())

all_preds = np.array(all_preds)
all_true = np.array(all_true)

print(
    classification_report(
        all_true,
        all_preds,
        target_names=le.classes_
    )
)

# ==========================================================
# 11 PLOT
# ==========================================================
print("\n[9] Plotting...")

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

axes[0].plot(history["train_loss"])
axes[0].plot(history["val_loss"])
axes[0].set_title("Loss")
axes[0].legend(["Train", "Val"])

axes[1].plot(history["train_acc"])
axes[1].plot(history["val_acc"])
axes[1].set_title("Accuracy")
axes[1].legend(["Train", "Val"])

cm = confusion_matrix(
    all_true,
    all_preds
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ax=axes[2]
)

axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Pred")
axes[2].set_ylabel("True")

plt.tight_layout()

plot_path = os.path.join(
    FIG_DIR,
    "mobilenetv2_results_v2.png"
)

plt.savefig(
    plot_path,
    dpi=150
)

plt.show()

print("\nSaved plot :", plot_path)
print("Saved best :", best_model_path)
print("DONE")