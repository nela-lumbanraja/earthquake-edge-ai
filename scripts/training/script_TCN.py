"""
TCN Final Full (Revised)
Waveform Classification (NO STFT)

Dataset:
- combined_5s.npy
- metadata_5s.npy
- data/splits_5s.npz  (fixed train/val/test indices + canonical class list)

Ringkasan perubahan terhadap versi sebelumnya (lihat penjelasan lengkap
di luar file ini untuk justifikasi metodologis setiap perubahan):

1. Pembagian data TIDAK lagi menggunakan train_test_split(). Indeks
   train/val/test dimuat langsung dari data/splits_5s.npz sehingga
   pembagian bersifat tetap (reproducible) dan tidak dibuat ulang.
2. Label mentah dipetakan ke label kanonik melalui LABEL_MAP, kemudian
   di-encode sesuai urutan `classes` dari file split (bukan LabelEncoder
   yang mengurutkan label secara alfabetis).
3. Augmentasi flip (x.flip(-1)) dan circular shift (torch.roll) dihapus
   karena tidak valid untuk sinyal time-series seismik (flip membalik
   arah waktu/fisika gelombang, roll membuat sinyal "membungkus" dari
   ujung ke awal trace sehingga menimbulkan diskontinuitas artifisial).
   Diganti dengan: noise injection, time-shift non-circular (zero
   padding), dan channel dropout khusus channel horizontal (N, E) --
   channel vertikal (Z) tidak pernah di-dropout.
4. Normalisasi z-score per-channel diganti menjadi z-score per-trace:
   mean & std dihitung dari gabungan ketiga channel dalam satu trace,
   lalu diterapkan ke seluruh channel pada trace yang sama. Ini menjaga
   rasio amplitudo relatif antar-channel (Z vs N vs E), yang membawa
   informasi penting untuk membedakan jenis event seismik.
5. Seluruh output (model terbaik, model terakhir, riwayat training,
   metrik, confusion matrix, classification report, konfigurasi, dan
   model hasil kompresi) disimpan ke results/runs/<tanggal>_TCN/.
6. Ditambahkan Post-Training Dynamic Quantization (torch.nn.Linear) dan
   evaluasi perbandingan model asli vs model terkuantisasi (akurasi,
   precision, recall, F1, ukuran file, waktu inferensi).

Arsitektur TCN (TemporalBlock, dilation, kernel size, jumlah filter),
hyperparameter training, optimizer, scheduler, dan loss function TIDAK
diubah dari versi asli.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# ==========================================================
# 0 CONFIG
# ==========================================================
DATA_DIR = "/home/indra/eq_team"

DATA_FILE = os.path.join(DATA_DIR, "combined_5s.npy")
META_FILE = os.path.join(DATA_DIR, "metadata_5s.npy")

# Pembagian data tetap (bukan random split ulang)
SPLIT_FILE = "data/splits_5s.npz"

# Pemetaan label mentah -> label kanonik
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

# Asumsi urutan channel pada data waveform: Z (vertikal), N, E (horizontal),
# konvensi ZNE yang umum dipakai pada data seismik 3-komponen.
# PENTING: sesuaikan urutan ini jika urutan channel pada data Anda berbeda
# (misalnya jika metadata menyimpan urutan channel secara eksplisit).
CHANNEL_ORDER = ["Z", "N", "E"]
VERTICAL_CHANNELS = ["Z"]
HORIZONTAL_CHANNELS = ["N", "E"]

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-4
SEQ_LEN = 500

# TCN hyper-params (TIDAK diubah dari versi asli)
TCN_CHANNELS = [64, 128, 128, 256]
KERNEL_SIZE = 7
DROPOUT = 0.2

WARMUP_EPOCHS = 5
EARLY_STOPPING = True
PATIENCE = 7
MIN_DELTA = 1e-4

# Parameter augmentasi (hanya diterapkan pada data train)
AUG_NOISE_STD = 0.02
AUG_NOISE_PROB = 0.5
AUG_SHIFT_RANGE = 20            # dalam sample, pada SEQ_LEN
AUG_SHIFT_PROB = 0.5
AUG_CHANNEL_DROPOUT_BLOCK_PROB = 0.5   # prob. augmentasi channel dropout dijalankan
AUG_CHANNEL_DROPOUT_RATE = 0.3         # prob. tiap channel horizontal individual di-drop

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEVICE == "cuda"
NUM_WORKERS = min(8, os.cpu_count() or 1)

# Direktori output run: results/runs/<tanggal>_<nama_model>/
MODEL_NAME = "TCN"
RUN_DIR = os.path.join(
    "results", "runs", f"{datetime.now().strftime('%Y%m%d')}_{MODEL_NAME}"
)
os.makedirs(RUN_DIR, exist_ok=True)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(42)

print("=" * 60)
print("TCN FINAL (REVISED) - 5s Waveform Classification")
print("=" * 60)
print("Device      :", DEVICE)
print("AMP         :", AMP)
print("NUM_WORKERS :", NUM_WORKERS)
print("SEQ_LEN     :", SEQ_LEN)
print("RUN_DIR     :", RUN_DIR)


# ==========================================================
# 1 LOAD METADATA
# ==========================================================
print("\n[1] Loading metadata...")

meta = np.load(META_FILE, allow_pickle=True).item()
labels_raw = np.asarray(meta["label"])

print("Metadata keys      :", list(meta.keys()))
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
    shape=shape
)

N, C, T = shape
print("Total samples :", N)


# ==========================================================
# 3 LOAD FIXED SPLIT (train/val/test) + CANONICAL CLASS LIST
# ==========================================================
print("\n[3] Loading fixed split dari", SPLIT_FILE, "...")

split = np.load(SPLIT_FILE, allow_pickle=True)

train_idx = np.asarray(split["train"])
val_idx = np.asarray(split["val"])
test_idx = np.asarray(split["test"])
classes = list(split["classes"])

NUM_CLASSES = len(classes)

print("Train idx  :", len(train_idx))
print("Val idx    :", len(val_idx))
print("Test idx   :", len(test_idx))
print("Classes    :", classes)
print("NumClasses :", NUM_CLASSES)

# --- Sanity check: tidak ada indeks yang tumpang tindih antar split ---
set_train = set(train_idx.tolist())
set_val = set(val_idx.tolist())
set_test = set(test_idx.tolist())

overlap_train_val = set_train & set_val
overlap_train_test = set_train & set_test
overlap_val_test = set_val & set_test

assert not overlap_train_val, f"Overlap train/val terdeteksi: {len(overlap_train_val)} indeks"
assert not overlap_train_test, f"Overlap train/test terdeteksi: {len(overlap_train_test)} indeks"
assert not overlap_val_test, f"Overlap val/test terdeteksi: {len(overlap_val_test)} indeks"

# --- Sanity check: seluruh indeks berada dalam rentang data ---
all_idx = np.concatenate([train_idx, val_idx, test_idx])
assert all_idx.min() >= 0 and all_idx.max() < N, "Indeks split berada di luar rentang data!"

print("Sanity check split : OK (tidak ada overlap, seluruh indeks valid)")


# ==========================================================
# 4 LABEL MAPPING (LABEL_MAP) + ENCODING SESUAI URUTAN `classes`
# ==========================================================
print("\n[4] Memetakan label mentah dengan LABEL_MAP...")

unique_raw = np.unique(labels_raw)
unmapped = [lbl for lbl in unique_raw if lbl not in LABEL_MAP]
assert not unmapped, f"Label berikut tidak terdapat pada LABEL_MAP: {unmapped}"

canonical_labels = np.array([LABEL_MAP[lbl] for lbl in labels_raw])

unknown_classes = sorted(set(canonical_labels.tolist()) - set(classes))
assert not unknown_classes, (
    f"Label kanonik berikut tidak terdapat pada daftar classes split: {unknown_classes}"
)

# Encoding label MENGIKUTI urutan classes dari file split (tidak diurutkan ulang)
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
labels = np.array([class_to_idx[c] for c in canonical_labels], dtype=np.int64)

print("Seluruh label berhasil dipetakan via LABEL_MAP -> classes split. OK")


# ==========================================================
# 5 DATASET
# ==========================================================
horizontal_ch_idx = [
    CHANNEL_ORDER.index(ch) for ch in HORIZONTAL_CHANNELS if ch in CHANNEL_ORDER
]


def zero_pad_shift(x, shift):
    """Time shift NON-CIRCULAR dengan zero padding (bukan torch.roll).

    x: tensor (C, T). Bagian yang "keluar" dari batas trace dibuang,
    bagian yang kosong akibat shift diisi nol -- tidak ada informasi
    dari ujung lain trace yang "membungkus" masuk seperti pada roll.
    """
    if shift == 0:
        return x
    c_, t_ = x.shape
    out = torch.zeros_like(x)
    if shift > 0:
        out[:, shift:] = x[:, : t_ - shift]
    else:
        out[:, : t_ + shift] = x[:, -shift:]
    return out


class EqDataset(Dataset):
    def __init__(self, data, labels, indices, augment=False):
        self.data = data
        self.labels = labels
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def _augment(self, x):
        # 1) Noise injection
        if torch.rand(1).item() < AUG_NOISE_PROB:
            x = x + AUG_NOISE_STD * torch.randn_like(x)

        # 2) Non-circular time shift (zero padding) -- BUKAN circular roll
        if torch.rand(1).item() < AUG_SHIFT_PROB:
            shift = torch.randint(-AUG_SHIFT_RANGE, AUG_SHIFT_RANGE + 1, (1,)).item()
            x = zero_pad_shift(x, shift)

        # 3) Channel dropout HANYA untuk channel horizontal (N, E).
        #    Channel vertikal (Z) tidak pernah di-dropout.
        if torch.rand(1).item() < AUG_CHANNEL_DROPOUT_BLOCK_PROB:
            x = x.clone()
            for ch_idx in horizontal_ch_idx:
                if ch_idx < x.shape[0] and torch.rand(1).item() < AUG_CHANNEL_DROPOUT_RATE:
                    x[ch_idx, :] = 0.0

        return x

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.data[i].copy()  # (C, T_orig)

        # Normalisasi z-score PER-TRACE: mean & std dihitung dari gabungan
        # seluruh nilai pada ketiga channel trace ini (bukan per-channel),
        # lalu diterapkan secara identik ke setiap channel pada trace yang
        # sama -- menjaga rasio amplitudo relatif antar-channel.
        mean = x.mean()
        std = x.std() + 1e-8
        x = (x - mean) / std

        x = torch.tensor(x, dtype=torch.float32)  # (C, T_orig)

        # Resample ke panjang tetap untuk TCN
        x = F.interpolate(
            x.unsqueeze(0),
            size=SEQ_LEN,
            mode="linear",
            align_corners=False
        ).squeeze(0)  # (C, SEQ_LEN)

        # Augmentasi HANYA untuk data train -> mencegah data leakage
        if self.augment:
            x = self._augment(x)

        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x, y


# ==========================================================
# 6 DATALOADER (memakai train_idx / val_idx / test_idx dari file split)
# ==========================================================
print("\n[5] Building dataloader...")

train_loader = DataLoader(
    EqDataset(data, labels, train_idx, augment=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
)

val_loader = DataLoader(
    EqDataset(data, labels, val_idx, augment=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
)

test_loader = DataLoader(
    EqDataset(data, labels, test_idx, augment=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
)


# ==========================================================
# 7 MODEL - TEMPORAL CONVOLUTIONAL NETWORK (ARSITEKTUR TIDAK DIUBAH)
# ==========================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else None
        )
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        residual = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + residual)


class TCNClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, channels, kernel_size=7, dropout=0.2):
        super().__init__()

        layers = []
        prev_channels = in_channels

        for i, out_channels in enumerate(channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    prev_channels, out_channels,
                    kernel_size=kernel_size, dilation=dilation, dropout=dropout
                )
            )
            prev_channels = out_channels

        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.tcn(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


print("\n[6] Building TCN...")

model = TCNClassifier(
    in_channels=C,
    num_classes=NUM_CLASSES,
    channels=TCN_CHANNELS,
    kernel_size=KERNEL_SIZE,
    dropout=DROPOUT
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable params :", f"{total_params:,}")

# --- Sanity check: jumlah output layer klasifikasi terakhir == jumlah kelas ---
final_linear = model.classifier[-1]
assert isinstance(final_linear, nn.Linear)
assert final_linear.out_features == NUM_CLASSES, (
    f"Output layer ({final_linear.out_features}) tidak sesuai jumlah kelas ({NUM_CLASSES})"
)
print("Sanity check output layer : OK ->", final_linear.out_features, "kelas")


# ==========================================================
# 8 TRAIN SETUP (hyperparameter TIDAK diubah dari versi asli)
# ==========================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-3
)

warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=WARMUP_EPOCHS
)

cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=max(1, EPOCHS - WARMUP_EPOCHS),
    eta_min=1e-6
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[WARMUP_EPOCHS]
)

scaler = torch.cuda.amp.GradScaler(enabled=AMP)

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "lr": []
}

best_val_loss = float("inf")
patience_counter = 0

best_model_path = os.path.join(RUN_DIR, "best_model.pt")
last_model_path = os.path.join(RUN_DIR, "last_model.pt")
log_path = os.path.join(RUN_DIR, "training_history.json")
config_path = os.path.join(RUN_DIR, "training_config.json")

training_config = {
    "data_file": DATA_FILE,
    "meta_file": META_FILE,
    "split_file": SPLIT_FILE,
    "classes": classes,
    "label_map": LABEL_MAP,
    "channel_order_assumed": CHANNEL_ORDER,
    "horizontal_channels": HORIZONTAL_CHANNELS,
    "vertical_channels": VERTICAL_CHANNELS,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "lr": LR,
    "seq_len": SEQ_LEN,
    "tcn_channels": TCN_CHANNELS,
    "kernel_size": KERNEL_SIZE,
    "dropout": DROPOUT,
    "warmup_epochs": WARMUP_EPOCHS,
    "early_stopping": EARLY_STOPPING,
    "patience": PATIENCE,
    "min_delta": MIN_DELTA,
    "augmentation": {
        "noise_std": AUG_NOISE_STD,
        "noise_prob": AUG_NOISE_PROB,
        "shift_range_samples": AUG_SHIFT_RANGE,
        "shift_prob": AUG_SHIFT_PROB,
        "channel_dropout_block_prob": AUG_CHANNEL_DROPOUT_BLOCK_PROB,
        "channel_dropout_rate_per_channel": AUG_CHANNEL_DROPOUT_RATE,
        "applies_to": "train only",
    },
    "optimizer": "AdamW",
    "weight_decay": 1e-3,
    "scheduler": "LinearLR warmup -> CosineAnnealingLR",
    "loss_function": "CrossEntropyLoss(label_smoothing=0.05)",
    "device": DEVICE,
    "amp": AMP,
    "num_train": int(len(train_idx)),
    "num_val": int(len(val_idx)),
    "num_test": int(len(test_idx)),
    "trainable_params": int(total_params),
}

with open(config_path, "w") as f:
    json.dump(training_config, f, indent=2)

print(f"Konfigurasi training disimpan di: {config_path}")


def run_epoch(loader, train=True):
    desc = "Train" if train else "Val"
    total_loss = 0
    correct = 0
    total = 0

    if train:
        model.train()
    else:
        model.eval()

    pbar = tqdm(loader, desc=desc, leave=False)

    for xb, yb in pbar:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=AMP):
                preds = model(xb)
                loss = criterion(preds, yb)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * len(yb)
        correct += (preds.argmax(1) == yb).sum().item()
        total += len(yb)

        pbar.set_postfix({
            "loss": f"{total_loss / total:.4f}",
            "acc": f"{correct / total:.4f}"
        })

    return total_loss / total, correct / total


# ==========================================================
# 9 TRAIN LOOP + EARLY STOPPING
# ==========================================================
print("\n[7] Training...")

last_epoch_run = 0

for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)

    with torch.no_grad():
        vl_loss, vl_acc = run_epoch(val_loader, train=False)

    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)
    history["lr"].append(current_lr)
    last_epoch_run = epoch + 1

    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    # Model terbaik tetap dipilih berdasarkan val_loss, sama seperti versi awal
    if vl_loss < best_val_loss - MIN_DELTA:
        best_val_loss = vl_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        marker = " <-- BEST"
    else:
        patience_counter += 1
        marker = f" | patience {patience_counter}/{PATIENCE}"

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"LR {current_lr:.2e} | "
        f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
        f"Val Loss {vl_loss:.4f} Acc {vl_acc:.4f}"
        + marker
    )

    if EARLY_STOPPING and patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
        break

# Simpan model pada akhir training (epoch terakhir yang benar-benar dijalankan)
torch.save(model.state_dict(), last_model_path)
print(f"Model terakhir (epoch {last_epoch_run}) disimpan di: {last_model_path}")


# ==========================================================
# 10 EVALUASI MODEL ASLI (FP32) PADA TEST SET
# ==========================================================
print("\n[8] Evaluation on Test Set (Original / FP32 Model)...")

model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.eval()


def run_inference(model_, loader, device):
    """Jalankan inferensi penuh atas satu loader.

    Mengembalikan: all_true, all_preds, all_probs, waktu_inferensi_detik, n_sampel.
    Satu batch warm-up dijalankan terlebih dahulu dan TIDAK dihitung dalam
    pengukuran waktu, untuk menghindari bias akibat overhead inisialisasi.
    """
    model_.eval()

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            _ = model_(xb)
            break

    all_preds, all_true, all_probs = [], [], []
    n_samples = 0

    t0 = time.time()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model_(xb)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()

            all_preds.extend(preds)
            all_true.extend(yb.numpy())
            all_probs.extend(probs)
            n_samples += len(yb)
    elapsed = time.time() - t0

    return np.array(all_true), np.array(all_preds), np.array(all_probs), elapsed, n_samples


all_true, all_preds, all_probs, infer_time_orig_device, n_test = run_inference(
    model, test_loader, DEVICE
)

report_orig_text = str(
    classification_report(
        all_true,
        all_preds,
        target_names=classes,
        digits=4,
        zero_division=0,
    )
)
print("\n--- Classification Report (Original Model) ---")
print(report_orig_text)

acc = accuracy_score(all_true, all_preds)
bal_acc = balanced_accuracy_score(all_true, all_preds)
prec_orig, rec_orig, f1_orig, _ = precision_recall_fscore_support(
    all_true, all_preds, average="macro", zero_division=0
)

try:
    auc = roc_auc_score(all_true, all_probs, multi_class="ovr", average="macro")
except ValueError:
    auc = float("nan")

print("\n--- Test Set Summary (Original Model, device =", DEVICE, ") ---")
print(f"Accuracy          : {acc:.4f} ({acc * 100:.2f}%)")
print(f"Balanced Accuracy : {bal_acc:.4f} ({bal_acc * 100:.2f}%)")
print(f"Precision (macro) : {prec_orig:.4f}")
print(f"Recall (macro)    : {rec_orig:.4f}")
print(f"F1 (macro)        : {f1_orig:.4f}")
print(f"ROC-AUC macro     : {auc:.4f}")

# --- Simpan classification report ---
classification_report_path = os.path.join(RUN_DIR, "classification_report.txt")
with open(classification_report_path, "w") as f:
    f.write("Classification Report - Original (FP32) Model\n")
    f.write("=" * 55 + "\n")
    f.write(report_orig_text)

# --- Simpan confusion matrix (array + plot) ---
cm = confusion_matrix(all_true, all_preds)
np.save(os.path.join(RUN_DIR, "confusion_matrix.npy"), cm)

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
    vmin=0,
    vmax=1
)
plt.title("Confusion Matrix (normalised) - Original Model")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
confusion_matrix_path = os.path.join(RUN_DIR, "confusion_matrix.png")
plt.savefig(confusion_matrix_path, dpi=150, bbox_inches="tight")
plt.close()


# ==========================================================
# 11 PLOT TRAINING CURVES
# ==========================================================
print("\n[9] Plotting training curves...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"TCN Training — Test Acc: {acc * 100:.2f}% | ROC-AUC: {auc:.4f}",
    fontsize=13,
    fontweight="bold"
)

axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["val_loss"], label="Val")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history["train_acc"], label="Train")
axes[1].plot(history["val_acc"], label="Val")
axes[1].axhline(acc, color="red", linestyle="--", linewidth=1.2, label=f"Test={acc:.4f}")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(history["lr"])
axes[2].set_title("Learning Rate Schedule")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("LR")
axes[2].grid(alpha=0.3)

plt.tight_layout()
training_curves_path = os.path.join(RUN_DIR, "training_curves.png")
plt.savefig(training_curves_path, dpi=150, bbox_inches="tight")
plt.close()


# ==========================================================
# 12 SIMPAN METRIK TEST SET (MODEL ASLI)
# ==========================================================
test_metrics_path = os.path.join(RUN_DIR, "test_metrics.json")
test_metrics_orig = {
    "device_used_for_eval": DEVICE,
    "accuracy": float(acc),
    "balanced_accuracy": float(bal_acc),
    "precision_macro": float(prec_orig),
    "recall_macro": float(rec_orig),
    "f1_macro": float(f1_orig),
    "roc_auc_macro": None if np.isnan(auc) else float(auc),
    "inference_time_sec_total": float(infer_time_orig_device),
    "inference_time_ms_per_sample": float((infer_time_orig_device / n_test) * 1000),
    "n_test_samples": int(n_test),
}
with open(test_metrics_path, "w") as f:
    json.dump(test_metrics_orig, f, indent=2)

print(f"Metrik test set (model asli) disimpan di: {test_metrics_path}")


# ==========================================================
# 13 POST-TRAINING DYNAMIC QUANTIZATION
# ==========================================================
print("\n[10] Post-Training Dynamic Quantization...")

try:
    torch.backends.quantized.engine = "fbgemm"
except RuntimeError:
    torch.backends.quantized.engine = "qnnpack"

# Quantization dilakukan SETELAH training & evaluasi model asli selesai.
model_cpu = model.to("cpu")
model_cpu.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model_cpu,
    {torch.nn.Linear},   # hanya layer Linear; Conv1d TIDAK diganti
    dtype=torch.qint8
)
quantized_model.eval()

# Simpan model hasil quantization secara terpisah dari model asli
quantized_model_path = os.path.join(RUN_DIR, "tcn_quantized_full_model.pt")
torch.save(quantized_model, quantized_model_path)

quantized_state_dict_path = os.path.join(RUN_DIR, "tcn_quantized_state_dict.pt")
torch.save(quantized_model.state_dict(), quantized_state_dict_path)

print(f"Model quantized (full object) disimpan di : {quantized_model_path}")
print(f"Model quantized (state_dict)  disimpan di : {quantized_state_dict_path}")


# ==========================================================
# 14 EVALUASI ULANG: MODEL ASLI (CPU) vs MODEL QUANTIZED (CPU)
# ==========================================================
# Dynamic quantization int8 hanya didukung di CPU, sehingga perbandingan
# yang adil (apple-to-apple) untuk akurasi & waktu inferensi dilakukan
# dengan menjalankan KEDUA model di CPU.
print("\n[11] Evaluation on Test Set (Original CPU vs Quantized CPU)...")

all_true_cpu, all_preds_cpu, all_probs_cpu, infer_time_orig_cpu, n_test_cpu = run_inference(
    model_cpu, test_loader, "cpu"
)
acc_orig_cpu = accuracy_score(all_true_cpu, all_preds_cpu)
prec_orig_cpu, rec_orig_cpu, f1_orig_cpu, _ = precision_recall_fscore_support(
    all_true_cpu, all_preds_cpu, average="macro", zero_division=0
)

all_true_q, all_preds_q, all_probs_q, infer_time_quant, n_test_q = run_inference(
    quantized_model, test_loader, "cpu"
)
acc_q = accuracy_score(all_true_q, all_preds_q)
bal_acc_q = balanced_accuracy_score(all_true_q, all_preds_q)
prec_q, rec_q, f1_q, _ = precision_recall_fscore_support(
    all_true_q, all_preds_q, average="macro", zero_division=0
)
try:
    auc_q = roc_auc_score(all_true_q, all_probs_q, multi_class="ovr", average="macro")
except ValueError:
    auc_q = float("nan")

report_quant_text = str(
    classification_report(
        all_true_q,
        all_preds_q,
        target_names=classes,
        digits=4,
        zero_division=0,
    )
)
print("\n--- Classification Report (Quantized Model) ---")
print(report_quant_text)

with open(os.path.join(RUN_DIR, "classification_report_quantized.txt"), "w") as f:
    f.write("Classification Report - Quantized (INT8, Dynamic) Model\n")
    f.write("=" * 55 + "\n")
    f.write(report_quant_text)

print("\n--- Test Set Summary (CPU vs CPU) ---")
print(f"[Original-CPU ] Acc {acc_orig_cpu:.4f} | F1 {f1_orig_cpu:.4f} | "
      f"Time {infer_time_orig_cpu:.4f}s ({(infer_time_orig_cpu / n_test_cpu) * 1000:.4f} ms/sampel)")
print(f"[Quantized-CPU] Acc {acc_q:.4f} | F1 {f1_q:.4f} | "
      f"Time {infer_time_quant:.4f}s ({(infer_time_quant / n_test_q) * 1000:.4f} ms/sampel)")


# ==========================================================
# 15 RINGKASAN PERBANDINGAN: ORIGINAL vs QUANTIZED
# ==========================================================
size_orig_mb = os.path.getsize(best_model_path) / (1024 ** 2)
size_quant_mb = os.path.getsize(quantized_state_dict_path) / (1024 ** 2)

comparison = {
    "note": (
        "Metrik dan waktu inferensi pada bagian ini dihitung dengan KEDUA "
        "model berjalan di CPU agar perbandingan adil, karena dynamic "
        "quantization (int8) hanya didukung di CPU. Ukuran file dibandingkan "
        "menggunakan format state_dict untuk kedua model."
    ),
    "original_model_cpu": {
        "accuracy": float(acc_orig_cpu),
        "precision_macro": float(prec_orig_cpu),
        "recall_macro": float(rec_orig_cpu),
        "f1_macro": float(f1_orig_cpu),
        "file_size_mb": float(size_orig_mb),
        "inference_time_sec_total": float(infer_time_orig_cpu),
        "inference_time_ms_per_sample": float((infer_time_orig_cpu / n_test_cpu) * 1000),
        "n_test_samples": int(n_test_cpu),
    },
    "quantized_model_cpu": {
        "accuracy": float(acc_q),
        "balanced_accuracy": float(bal_acc_q),
        "precision_macro": float(prec_q),
        "recall_macro": float(rec_q),
        "f1_macro": float(f1_q),
        "roc_auc_macro": None if np.isnan(auc_q) else float(auc_q),
        "file_size_mb": float(size_quant_mb),
        "inference_time_sec_total": float(infer_time_quant),
        "inference_time_ms_per_sample": float((infer_time_quant / n_test_q) * 1000),
        "n_test_samples": int(n_test_q),
    },
    "compression_ratio_size": (
        float(size_orig_mb / size_quant_mb) if size_quant_mb > 0 else None
    ),
    "speedup_ratio_cpu_inference": (
        float(infer_time_orig_cpu / infer_time_quant) if infer_time_quant > 0 else None
    ),
}

comparison_path = os.path.join(RUN_DIR, "quantization_comparison.json")
with open(comparison_path, "w") as f:
    json.dump(comparison, f, indent=2)

print("\n--- Perbandingan Original (CPU) vs Quantized (CPU) ---")
print(f"File size  : {size_orig_mb:.4f} MB -> {size_quant_mb:.4f} MB "
      f"(rasio kompresi: {comparison['compression_ratio_size']:.2f}x)"
      if comparison["compression_ratio_size"] else "")
print(f"Accuracy   : {acc_orig_cpu:.4f} -> {acc_q:.4f}")
print(f"F1 (macro) : {f1_orig_cpu:.4f} -> {f1_q:.4f}")
print(f"Inference  : {infer_time_orig_cpu:.4f}s -> {infer_time_quant:.4f}s "
      f"(CPU, {n_test_cpu} sampel)")


# ==========================================================
# 16 RINGKASAN FILE YANG DISIMPAN
# ==========================================================
print("\n" + "=" * 60)
print("DONE — Seluruh output disimpan di:", RUN_DIR)
print("=" * 60)
print(f"  Model terbaik (best_model.pt)              : {best_model_path}")
print(f"  Model terakhir (last_model.pt)              : {last_model_path}")
print(f"  Riwayat training (training_history.json)    : {log_path}")
print(f"  Konfigurasi training (training_config.json) : {config_path}")
print(f"  Metrik test set asli (test_metrics.json)     : {test_metrics_path}")
print(f"  Classification report (asli)                 : {classification_report_path}")
print(f"  Confusion matrix (.npy & .png)                : {confusion_matrix_path}")
print(f"  Training curves (.png)                        : {training_curves_path}")
print(f"  Model quantized (full object)                 : {quantized_model_path}")
print(f"  Model quantized (state_dict)                  : {quantized_state_dict_path}")
print(f"  Classification report (quantized)             : "
      f"{os.path.join(RUN_DIR, 'classification_report_quantized.txt')}")
print(f"  Perbandingan original vs quantized (.json)    : {comparison_path}")
