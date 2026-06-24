"""
MobileNetV2 v2
Waveform Classification 5s - NO STFT

Revisi:
1. Menghapus train_test_split(); split dimuat dari data/splits_5s.npz.
2. Menggunakan LABEL_MAP kanonik 7 kelas.
3. Menghapus augmentasi tidak valid (flip, roll).
4. Menggunakan augmentasi sah:
   - noise injection
   - non-circular time shift (zero padding)
   - channel dropout horizontal E/N
5. Menggunakan z-score per-trace gabungan 3 channel (bukan per-channel).
6. Output disimpan ke results/runs/<tanggal>_<nama_model>/.
7. Fix torch.load PyTorch 2.6 dengan weights_only=False.
8. Menambahkan Dynamic Quantization sebagai teknik kompresi.
9. Arsitektur model, hyperparameter, dan loss function TIDAK diubah
   (kecuali penyesuaian wajib agar kompatibel dengan split & label baru).
"""

import os
import json
import copy
import hashlib
import datetime
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

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

DATA_DIR = "/home/indra/eq_team"
DATA_FILE = os.path.join(DATA_DIR, "combined_5s.npy")
META_FILE = os.path.join(DATA_DIR, "metadata_5s.npy")

PROJECT_DIR = "/home/indra/eq_team/earthquake-classification"
SPLIT_FILE = os.path.join(PROJECT_DIR, "data", "splits_5s.npz")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")

MODEL_NAME = "mobilenetv2_v2"

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-4
IMG_SIZE = 224
SEQ_LEN = 500

EARLY_STOPPING = True
PATIENCE = 5
MIN_DELTA = 1e-4

RANDOM_SEED = 42
NUM_WORKERS = 8

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

# ==========================================================
# 1 UTILITIES
# ==========================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256sum(file_path):
    h = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def get_git_hash():
    try:
        import subprocess

        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_DIR
        ).decode().strip()

        return git_hash

    except Exception:
        return "nogit"


def make_run_dir():
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    git_hash = get_git_hash()

    run_name = f"{date_str}_{MODEL_NAME}_{git_hash}"
    run_dir = os.path.join(RUNS_DIR, run_name)

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)

    return run_dir, git_hash


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


# ==========================================================
# 2 DATASET
# ==========================================================

class EqDataset(Dataset):

    def __init__(self, data, labels, indices, augment=False):
        self.data = data
        self.labels = labels
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def valid_augment(self, x):
        # Noise injection
        if torch.rand(1) < 0.5:
            sigma = 0.01 + 0.04 * torch.rand(1).item()
            x = x + sigma * torch.randn_like(x)

        # Non-circular time shift (zero padding, bukan roll)
        if torch.rand(1) < 0.5:
            s = int(torch.randint(-20, 21, (1,)))

            if s > 0:
                x = torch.cat(
                    [torch.zeros_like(x[:, :s]), x[:, :-s]],
                    dim=-1
                )

            elif s < 0:
                shift = -s
                x = torch.cat(
                    [x[:, shift:], torch.zeros_like(x[:, :shift])],
                    dim=-1
                )

        # Channel dropout hanya E/N, bukan Z
        if torch.rand(1) < 0.05:
            ch = int(torch.randint(0, 2, (1,)))
            x[ch] = 0.0

        return x

    def __getitem__(self, idx):

        i = self.indices[idx]
        x = self.data[i].copy()

        assert x.shape[-1] == SEQ_LEN, (
            f"Expected sequence length {SEQ_LEN}, got {x.shape[-1]}"
        )

        # z-score per-trace gabungan 3 channel (bukan per-channel)
        mean = x.mean()
        std = x.std() + 1e-8
        x = (x - mean) / std

        x = torch.tensor(x, dtype=torch.float32)

        if self.augment:
            x = self.valid_augment(x)

        # (3,500) -> (1,3,500)
        x = x.unsqueeze(0)

        # resize 1D: (1,3,500) -> (1,3,224)
        x = F.interpolate(
            x,
            size=IMG_SIZE,
            mode="linear",
            align_corners=False
        )

        # (1,3,224) -> (3,224)
        x = x.squeeze(0)

        # pseudo-image: (3,224) -> (3,224,224)
        x = x.unsqueeze(-1).repeat(1, 1, IMG_SIZE)

        y = torch.tensor(
            self.labels[i],
            dtype=torch.long
        )

        return x, y


# ==========================================================
# 3 TRAIN / EVAL FUNCTION
# ==========================================================

def run_epoch(model, loader, criterion, optimizer=None, scaler=None, train=True):

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

        with torch.set_grad_enabled(train):

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
        correct += (pred_class == yb).sum().item()
        total += len(yb)

        pbar.set_postfix({
            "loss": f"{total_loss / total:.4f}",
            "acc": f"{correct / total:.4f}"
        })

    return total_loss / total, correct / total


def evaluate_test(model, loader, classes, eval_device=DEVICE):

    model.eval()
    model.to(eval_device)

    all_preds = []
    all_true = []

    use_amp = eval_device == "cuda"

    with torch.no_grad():

        for xb, yb in tqdm(loader, desc="Test", leave=False):

            xb = xb.to(eval_device, non_blocking=True)

            with torch.amp.autocast(
                "cuda",
                enabled=use_amp
            ):
                logits = model(xb)

            pred = logits.argmax(1).cpu().numpy()

            all_preds.extend(pred)
            all_true.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    report = classification_report(
        all_true,
        all_preds,
        target_names=classes,
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(
        all_true,
        all_preds
    )

    acc = (all_preds == all_true).mean()

    return report, cm, acc


# ==========================================================
# 4 MAIN
# ==========================================================

def main():

    set_seed(RANDOM_SEED)

    run_dir, git_hash = make_run_dir()

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    fig_dir = os.path.join(run_dir, "figures")
    metrics_dir = os.path.join(run_dir, "metrics")

    best_model_path = os.path.join(
        ckpt_dir,
        "best_mobilenetv2_v2.pt"
    )

    print("=" * 60)
    print("MobileNetV2 v2 Training + Dynamic Quantization")
    print("=" * 60)
    print("Device  :", DEVICE)
    print("AMP     :", AMP)
    print("Run dir :", run_dir)

    # ------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------
    print("\n[1] Loading metadata...")

    meta = np.load(META_FILE, allow_pickle=True).item()

    labels_raw = np.array(meta["label"])

    labels_mapped = np.array([
        LABEL_MAP.get(str(label), str(label))
        for label in labels_raw
    ])

    # ------------------------------------------------------
    # Load split (menggantikan train_test_split)
    # ------------------------------------------------------
    print("\n[2] Loading shared split...")

    split = np.load(SPLIT_FILE, allow_pickle=True)

    train_idx = split["train"]
    val_idx = split["val"]
    test_idx = split["test"]
    classes = list(split["classes"])
    split_type = str(split["split_type"])

    class_to_idx = {
        class_name: i
        for i, class_name in enumerate(classes)
    }

    labels = np.array([
        class_to_idx[label]
        for label in labels_mapped
    ])

    NUM_CLASSES = len(classes)

    print("Split file :", SPLIT_FILE)
    print("Split type :", split_type)
    print("Classes    :", classes)
    print("Train      :", len(train_idx))
    print("Val        :", len(val_idx))
    print("Test       :", len(test_idx))

    split_sha = sha256sum(SPLIT_FILE)

    # ------------------------------------------------------
    # Load memmap
    # ------------------------------------------------------
    print("\n[3] Loading memmap waveform...")

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

    # ------------------------------------------------------
    # Save config
    # ------------------------------------------------------
    config = {
        "model": MODEL_NAME,
        "data_file": DATA_FILE,
        "meta_file": META_FILE,
        "split_file": SPLIT_FILE,
        "split_sha256": split_sha,
        "split_type": split_type,
        "classes": classes,
        "label_map": LABEL_MAP,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "img_size": IMG_SIZE,
        "seq_len": SEQ_LEN,
        "normalization": "zscore_per_trace_joint",
        "augmentation": [
            "noise_injection",
            "non_circular_time_shift",
            "horizontal_channel_dropout"
        ],
        "forbidden_augmentation": [
            "flip",
            "roll"
        ],
        "compression": "dynamic_quantization",
        "compression_target": "nn.Linear",
        "device": DEVICE,
        "amp": AMP,
        "random_seed": RANDOM_SEED,
        "git_commit": git_hash
    }

    save_json(
        config,
        os.path.join(run_dir, "config.json")
    )

    # ------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------
    print("\n[4] Building dataloaders...")

    train_loader = DataLoader(
        EqDataset(data, labels, train_idx, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        EqDataset(data, labels, val_idx, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        EqDataset(data, labels, test_idx, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    # ------------------------------------------------------
    # Model — MobileNetV2 (arsitektur tidak diubah)
    # ------------------------------------------------------
    print("\n[5] Building MobileNetV2...")

    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    # MobileNetV2 classifier: Sequential(Dropout, Linear(1280, 1000))
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

    # ------------------------------------------------------
    # Training setup (hyperparameter & loss tidak diubah)
    # ------------------------------------------------------
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

    best_acc = 0.0
    patience_counter = 0

    # ------------------------------------------------------
    # Train loop
    # ------------------------------------------------------
    print("\n[6] Training...")

    for epoch in tqdm(range(EPOCHS), desc="Epochs"):

        tr_loss, tr_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            train=True
        )

        with torch.no_grad():
            vl_loss, vl_acc = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                train=False
            )

        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_acc + MIN_DELTA:

            best_acc = vl_acc
            patience_counter = 0

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "config": config,
                    "norm": "zscore_per_trace_joint",
                    "split_file": "data/splits_5s.npz",
                    "split_sha256": split_sha,
                    "split_type": split_type,
                    "git_commit": git_hash,
                    "epoch": epoch + 1,
                    "val_acc": float(best_acc),
                    "total_params": int(total_params)
                },
                best_model_path
            )

            marker = " <-- BEST"

        else:
            patience_counter += 1
            marker = f" | patience {patience_counter}/{PATIENCE}"

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss {tr_loss:.4f} "
            f"Acc {tr_acc:.4f} | "
            f"Val Loss {vl_loss:.4f} "
            f"Acc {vl_acc:.4f}"
            + marker
        )

        if EARLY_STOPPING and patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    # ------------------------------------------------------
    # Save history
    # ------------------------------------------------------
    save_json(
        history,
        os.path.join(metrics_dir, "history.json")
    )

    # ------------------------------------------------------
    # Evaluation baseline
    # ------------------------------------------------------
    print("\n[7] Evaluation baseline model...")

    checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    report, cm, test_acc = evaluate_test(
        model,
        test_loader,
        classes,
        eval_device=DEVICE
    )

    print(report)

    report_path = os.path.join(
        metrics_dir,
        "classification_report_baseline.txt"
    )

    with open(report_path, "w") as f:
        f.write(report)

    baseline_size_mb = os.path.getsize(best_model_path) / (1024 * 1024)

    # ------------------------------------------------------
    # Dynamic Quantization
    # ------------------------------------------------------
    print("\n[8] Applying Dynamic Quantization...")

    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()

    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8
    )

    quantized_path = os.path.join(
        ckpt_dir,
        "mobilenetv2_v2_dynamic_quantized.pt"
    )

    torch.save(
        {
            "model": quantized_model,
            "classes": classes,
            "config": config,
            "compression": "dynamic_quantization",
            "quantized_layers": "nn.Linear",
            "dtype": "qint8",
            "baseline_checkpoint": best_model_path,
            "split_sha256": split_sha,
            "split_type": split_type,
            "git_commit": git_hash
        },
        quantized_path
    )

    quant_report, quant_cm, quant_test_acc = evaluate_test(
        quantized_model,
        test_loader,
        classes,
        eval_device="cpu"
    )

    quant_report_path = os.path.join(
        metrics_dir,
        "classification_report_dynamic_quantized.txt"
    )

    with open(quant_report_path, "w") as f:
        f.write(quant_report)

    quantized_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)

    print("\nDynamic Quantization Result")
    print("Baseline size  :", f"{baseline_size_mb:.2f} MB")
    print("Quantized size :", f"{quantized_size_mb:.2f} MB")
    print("Baseline acc   :", f"{test_acc:.4f}")
    print("Quantized acc  :", f"{quant_test_acc:.4f}")

    # ------------------------------------------------------
    # Save summary metrics
    # ------------------------------------------------------
    summary = {
        "model": MODEL_NAME,
        "best_val_acc": float(best_acc),
        "baseline_test_acc": float(test_acc),
        "quantized_test_acc": float(quant_test_acc),
        "total_params": int(total_params),
        "baseline_size_mb": float(baseline_size_mb),
        "quantized_size_mb": float(quantized_size_mb),
        "compression": "dynamic_quantization",
        "compression_target": "nn.Linear",
        "checkpoint_baseline": best_model_path,
        "checkpoint_quantized": quantized_path,
        "classification_report_baseline": report_path,
        "classification_report_quantized": quant_report_path,
        "run_dir": run_dir,
        "split_sha256": split_sha,
        "split_type": split_type
    }

    save_json(
        summary,
        os.path.join(metrics_dir, "summary.json")
    )

    # ------------------------------------------------------
    # Plot baseline result
    # ------------------------------------------------------
    print("\n[9] Plotting...")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5)
    )

    axes[0].plot(history["train_loss"])
    axes[0].plot(history["val_loss"])
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(["Train", "Val"])

    axes[1].plot(history["train_acc"])
    axes[1].plot(history["val_acc"])
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(["Train", "Val"])

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[2]
    )

    axes[2].set_title("Baseline Confusion Matrix")
    axes[2].set_xlabel("Predicted Label")
    axes[2].set_ylabel("True Label")

    plt.tight_layout()

    plot_path = os.path.join(
        fig_dir,
        "mobilenetv2_v2_baseline_results.png"
    )

    plt.savefig(
        plot_path,
        dpi=150
    )

    plt.close(fig)

    # ------------------------------------------------------
    # Plot quantized confusion matrix
    # ------------------------------------------------------
    fig_q, ax_q = plt.subplots(
        1,
        1,
        figsize=(8, 6)
    )

    sns.heatmap(
        quant_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax_q
    )

    ax_q.set_title("Dynamic Quantized Confusion Matrix")
    ax_q.set_xlabel("Predicted Label")
    ax_q.set_ylabel("True Label")

    plt.tight_layout()

    quant_plot_path = os.path.join(
        fig_dir,
        "mobilenetv2_v2_dynamic_quantized_cm.png"
    )

    plt.savefig(
        quant_plot_path,
        dpi=150
    )

    plt.close(fig_q)

    print("\nSaved run dir          :", run_dir)
    print("Saved baseline model   :", best_model_path)
    print("Saved quantized model  :", quantized_path)
    print("Baseline report        :", report_path)
    print("Quantized report       :", quant_report_path)
    print("Baseline plot          :", plot_path)
    print("Quantized plot         :", quant_plot_path)
    print("Baseline size          :", f"{baseline_size_mb:.2f} MB")
    print("Quantized size         :", f"{quantized_size_mb:.2f} MB")
    print("Best val acc           :", f"{best_acc:.4f}")
    print("Baseline test acc      :", f"{test_acc:.4f}")
    print("Quantized test acc     :", f"{quant_test_acc:.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
