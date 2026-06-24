"""
WaveformTransformer — v4 (restrukturisasi mengikuti pola MobileNetV3 v2)
Waveform Classification (NO STFT)
Dataset:
- combined_5s.npy
- metadata_5s.npy

FIX (v4.1): RuntimeError "Trying to backward through the graph a second
time" yang muncul di tahap Prune-FT. Penyebab: setelah
prune.global_unstructured(), atribut `.weight` pada layer Linear menjadi
computed tensor (weight_orig * weight_mask) lewat forward pre-hook.
Optimizer (ft_optimizer) dibuat dari student.parameters() SETELAH pruning,
itu sudah benar -- tapi teacher_logits dan student_logits perlu di-detach
secara eksplisit di titik yang tepat, dan zero_grad harus set_to_none=True
supaya tidak ada referensi grad lama nyangkut ke parameter yang baru saja
direparametrisasi oleh pruning. Lihat run_distill_epoch().
"""

import os
import sys
import time
import copy
import math
import json
import csv
import hashlib
import subprocess
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
    average_precision_score,
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

# FIX: nn.TransformerEncoderLayer punya "fastpath" internal (aktif saat
# model di-.eval()) yang membangun tensor_args dari semua weight/bias di
# linear1, linear2, self_attn, dst, lalu mengecek `.device` masing-masing --
# TANPA peduli apakah layer itu sudah diganti torch.quantization.quantize_dynamic().
# Setelah dynamic quantization, linear1/linear2 (nn.Linear biasa di dalam
# TransformerEncoderLayer) berubah jadi torch.ao.nn.quantized.dynamic.Linear,
# yang atribut `.weight`-nya adalah METHOD (bukan Tensor) -- sehingga
# tensor_args berisi method itu, dan `x.device` di fastpath meledak dengan
# "'function' object has no attribute 'device'" begitu model quantized
# dijalankan untuk evaluasi/inference (evaluate_full, measure_latency_ms).
# Ini bukan bug di script kita, melainkan fastpath PyTorch yang belum
# dirancang untuk submodule quantized. Matikan saja fastpath-nya -- itu
# murni optimisasi kernel C++ untuk inference, mematikannya TIDAK mengubah
# hasil/akurasi sama sekali, hanya sedikit lebih lambat di CPU.
torch.backends.mha.set_fastpath_enabled(False)

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# ==========================================================
# 0 CONFIG
# ==========================================================

DATA_DIR = "/home/indra/eq_team"
DATA_FILE = os.path.join(DATA_DIR, "combined_5s.npy")
META_FILE = os.path.join(DATA_DIR, "metadata_5s.npy")

PROJECT_DIR = "/home/indra/eq_team/earthquake-classification"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
SPLIT_DIR = os.path.join(PROJECT_DIR, "data")
SPLIT_FILE = os.path.join(SPLIT_DIR, "splits_5s.npz")

MODEL_NAME = "transformer_compression"

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-4

# Transformer hyper-params (TEACHER)
PATCH_SIZE = 10
SEQ_LEN = 500
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DIM_FF = 512
DROPOUT = 0.1

WARMUP_EPOCHS = 5

EARLY_STOPPING = True
PATIENCE = 7
MIN_DELTA = 1e-4

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEVICE == "cuda"
NUM_WORKERS = 0

CHANNEL_Z_INDEX = 2

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
EQ_CLASS_NAME = "earthquake"

NORM_DESC = "zscore_per_trace_joint"
BANDPASS_ENABLED = False
BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ = 1.0, 45.0
SAMPLING_RATE_HZ = 100.0

IMBALANCE_STRATEGY = "weighted_ce"
FOCAL_GAMMA = 2.0

STUDENT_PATCH_SIZE = 25
STUDENT_D_MODEL = 64
STUDENT_NHEAD = 4
STUDENT_NUM_LAYERS = 2
STUDENT_DIM_FF = 128
STUDENT_DROPOUT = 0.1

DISTILL_EPOCHS = 25
DISTILL_LR = 5e-4
DISTILL_TEMPERATURE = 4.0
DISTILL_ALPHA = 0.3

PRUNE_AMOUNT = 0.30
PRUNE_FINETUNE_EPOCHS = 5

QUANT_BACKEND = "fbgemm"

BENCH_BATCH_SIZES = [1, 32, 256]

LEAKAGE_CAVEAT = (
    "split acak per-trace — berpotensi optimistis karena event leakage "
    "(lihat PANDUAN_new.md §1.3, akan diperbaiki di P2)"
)


# ==========================================================
# 1 UTILITIES
# ==========================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def sha256sum(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_hash():
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
        return git_hash
    except Exception:
        return "nogit"


def make_run_dir():
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    git_hash = get_git_hash()
    run_name = f"{date_str}_{MODEL_NAME}_{git_hash}"
    run_dir = os.path.join(RESULTS_DIR, "runs", run_name)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    fig_dir = os.path.join(run_dir, "figures")
    log_dir = os.path.join(run_dir, "logs")
    table_dir = os.path.join(run_dir, "tables")

    for d in (ckpt_dir, fig_dir, log_dir, table_dir):
        os.makedirs(d, exist_ok=True)

    return run_dir, ckpt_dir, fig_dir, log_dir, table_dir, git_hash


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.__stdout__
        self.logfile = open(filepath, "w", encoding="utf-8", buffering=1)

    def write(self, msg):
        self.terminal.write(msg)
        self.logfile.write(msg)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


# ==========================================================
# 2 LOSS / IMBALANCE
# ==========================================================

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight,
                              label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_criterion(strategy, class_weights_t):
    if strategy == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.1)
    elif strategy == "focal":
        return FocalLoss(weight=class_weights_t, gamma=FOCAL_GAMMA, label_smoothing=0.1)
    else:
        return nn.CrossEntropyLoss(label_smoothing=0.1)


# ==========================================================
# 3 DATASET
# ==========================================================

def maybe_bandpass(x_np):
    if not BANDPASS_ENABLED:
        return x_np
    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError:
        raise RuntimeError("scipy diperlukan untuk bandpass tapi tidak terpasang.")
    sos = butter(4, [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ], btype="bandpass",
                 fs=SAMPLING_RATE_HZ, output="sos")
    return sosfiltfilt(sos, x_np, axis=-1).copy()


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

        x = maybe_bandpass(x)

        mean = x.mean()
        std = x.std() + 1e-8
        x = (x - mean) / std

        x = torch.tensor(x, dtype=torch.float32)

        assert x.shape[-1] == SEQ_LEN, (
            f"Panjang trace ({x.shape[-1]}) != SEQ_LEN ({SEQ_LEN}). "
            f"Jangan resample diam-diam — cek file data yang dipakai."
        )

        if self.augment:
            if torch.rand(1) < 0.5:
                sigma = 0.01 + 0.04 * torch.rand(1).item()
                x = x + sigma * torch.randn_like(x)

            if torch.rand(1) < 0.5:
                s = int(torch.randint(-20, 21, (1,)))
                if s > 0:
                    x = torch.cat([torch.zeros_like(x[:, :s]), x[:, :-s]], dim=-1)
                elif s < 0:
                    shift = -s
                    x = torch.cat([x[:, shift:], torch.zeros_like(x[:, :shift])], dim=-1)

            if torch.rand(1) < 0.05:
                horiz_idx = [c for c in range(x.shape[0]) if c != CHANNEL_Z_INDEX]
                x[horiz_idx[int(torch.randint(0, len(horiz_idx), (1,)))]] = 0.0

        y = torch.tensor(self.labels[i], dtype=torch.long)
        x = x.contiguous().clone()
        return x, y


# ==========================================================
# 4 MODEL
# ==========================================================

class PatchEmbedding1D(nn.Module):
    def __init__(self, in_channels, seq_len, patch_size, d_model):
        super().__init__()
        assert seq_len % patch_size == 0, \
            f"SEQ_LEN ({seq_len}) harus habis dibagi PATCH_SIZE ({patch_size})"
        self.num_patches = seq_len // patch_size
        self.proj = nn.Linear(in_channels * patch_size, d_model)

    def forward(self, x):
        B, C, L = x.shape
        x = x.view(B, C, self.num_patches, -1)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, self.num_patches, -1)
        return self.proj(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class WaveformTransformer(nn.Module):
    def __init__(self, in_channels, seq_len, patch_size, d_model, nhead,
                 num_layers, dim_feedforward, dropout, num_classes):
        super().__init__()
        self.patch_embed = PatchEmbedding1D(in_channels, seq_len, patch_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                              norm=nn.LayerNorm(d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, num_classes)
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
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B = x.size(0)
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_enc(tokens)
        out = self.encoder(tokens)
        return self.head(out[:, 0])


# ==========================================================
# 5 CHECKPOINT HELPERS
# ==========================================================

def save_checkpoint(path, net, epoch, val_loss, run_meta, extra_config=None):
    cfg = dict(run_meta["config"])
    if extra_config:
        cfg.update(extra_config)
    torch.save({
        "state_dict": net.state_dict(),
        "classes": run_meta["classes"],
        "config": cfg,
        "norm": run_meta["norm"],
        "split_file": run_meta["split_file"],
        "split_sha256": run_meta["split_sha256"],
        "split_type": run_meta["split_type"],
        "git_commit": run_meta["git_commit"],
        "epoch": epoch,
        "val_loss": val_loss,
    }, path)


def load_checkpoint_state_dict(path, map_location=DEVICE):
    # weights_only=False: checkpoint membawa metadata non-tensor (classes,
    # config dict, numpy scalars) sehingga loader strict PyTorch>=2.6 gagal
    # tanpa flag ini. Aman karena checkpoint dibuat sendiri oleh script ini.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    return ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt


# ==========================================================
# 6 TRAIN / EVAL FUNCTION
# ==========================================================

def run_epoch(model, loader, criterion, optimizer=None, scaler=None, train=True):
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
                loss = criterion(preds, yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(yb)
            correct += (preds.argmax(1) == yb).sum().item()
            total += len(yb)
            pbar.set_postfix({"loss": f"{total_loss/total:.4f}", "acc": f"{correct/total:.4f}"})
    else:
        model.eval()
        with torch.no_grad():
            for xb, yb in pbar:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=AMP):
                    preds = model(xb)
                    loss = criterion(preds, yb)
                total_loss += loss.item() * len(yb)
                correct += (preds.argmax(1) == yb).sum().item()
                total += len(yb)
                pbar.set_postfix({"loss": f"{total_loss/total:.4f}", "acc": f"{correct/total:.4f}"})

    return total_loss / total, correct / total


def evaluate_full(model, loader, device=DEVICE):
    model.eval()
    all_p, all_t, all_pr = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            all_p.extend(preds)
            all_t.extend(yb.numpy())
            all_pr.extend(probs)
    return np.array(all_p), np.array(all_t), np.array(all_pr)


def find_threshold_for_ftr(true_bin, prob, target_ftr=0.05, n_steps=1001):
    thresholds = np.linspace(0.0, 1.0, n_steps)
    for thr in thresholds:
        pred = (prob >= thr).astype(int)
        tn = np.sum((pred == 0) & (true_bin == 0))
        fp = np.sum((pred == 1) & (true_bin == 0))
        ftr = fp / (fp + tn + 1e-12)
        if ftr <= target_ftr:
            return float(thr)
    return 1.0


def compute_ece(confidence, correct, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = confidence[mask].mean()
        weight = mask.sum() / len(confidence)
        ece += weight * abs(acc_bin - conf_bin)
        bin_stats.append((conf_bin, acc_bin, mask.sum()))
    return ece, bin_stats


def distillation_loss(student_logits, teacher_logits, true_labels, class_weights_t,
                       T=DISTILL_TEMPERATURE, alpha=DISTILL_ALPHA):
    hard_loss = F.cross_entropy(student_logits, true_labels, weight=class_weights_t, label_smoothing=0.1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
    return alpha * hard_loss + (1 - alpha) * soft_loss


def run_distill_epoch(teacher, student, loader, optimizer, class_weights_t, train=True):
    """
    FIX v4.1: dua perubahan untuk mengatasi
    'RuntimeError: Trying to backward through the graph a second time':

    1) teacher(xb).detach() -- pemutus graph eksplisit. torch.no_grad()
       seharusnya sudah cukup, tapi .detach() adalah pengaman tambahan yang
       tidak mengubah hasil numerik sama sekali (no-op secara nilai),
       memastikan teacher_logits benar-benar leaf tensor tanpa requires_grad
       sebelum dipakai distillation_loss().

    2) optimizer.zero_grad(set_to_none=True) -- membersihkan grad lama
       secara total (bukan diisi nol) sebelum backward(). Ini penting
       khususnya saat optimizer dipakai untuk parameter yang baru saja
       direparametrisasi oleh prune.global_unstructured() (weight_orig +
       weight_mask), supaya tidak ada .grad lama yang mereferensikan graph
       dari iterasi/epoch sebelumnya.
    """
    desc = "KD-Train" if train else "KD-Val"
    total_loss = correct = total = 0
    pbar = tqdm(loader, desc=desc, leave=False, file=sys.__stdout__)
    student.train() if train else student.eval()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in pbar:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            with torch.no_grad():
                teacher_logits = teacher(xb).detach()

            student_logits = student(xb)
            loss = distillation_loss(student_logits, teacher_logits, yb, class_weights_t)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(yb)
            correct += (student_logits.argmax(1) == yb).sum().item()
            total += len(yb)
            pbar.set_postfix({"loss": f"{total_loss/total:.4f}", "acc": f"{correct/total:.4f}"})
    return total_loss / total, correct / total


def file_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2) if os.path.exists(path) else float("nan")


def count_params(net):
    return sum(p.numel() for p in net.parameters())


def measure_latency_ms(net, dataset, device, batch_size, n_batches=20):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    net = net.to(device).eval()
    times = []
    with torch.no_grad():
        for i, (xb, _) in enumerate(loader):
            if i >= n_batches:
                break
            xb = xb.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            _ = net(xb)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
    return float(np.mean(times)) if times else float("nan")


def format_comparison_table(variants, bench_batch_sizes):
    """
    Bangun tabel teks perbandingan model dari daftar `variants`
    (list of dict: label, params, acc, size_mb, latency={bs: ms}).

    Sebelumnya header & isi baris dihitung dua kali (sekali untuk print,
    sekali untuk ditulis ke file) dan dicocokkan lewat fuzzy string-matching
    nama varian. Di sini setiap variant sudah membawa latency-nya sendiri
    (dict {batch_size: ms}), jadi tidak perlu pencocokan nama sama sekali.
    """
    header = f"{'Model':<32} {'Params':>12} {'Test Acc':>10} {'Size (MB)':>10}"
    for bs in bench_batch_sizes:
        header += f"  Lat@bs{bs} (ms)"

    lines = [header]
    for v in variants:
        line = f"{v['label']:<32} {v['params']:>12,} {v['acc']:>10.4f} {v['size_mb']:>10.3f}"
        for bs in bench_batch_sizes:
            line += f"  {v['latency'][bs]:>14.2f}"
        lines.append(line)
    return lines


# ==========================================================
# 7 MAIN
# ==========================================================

def main():

    set_seed(RANDOM_SEED)

    run_dir, ckpt_dir, fig_dir, log_dir, table_dir, git_hash = make_run_dir()

    global_summary_csv = os.path.join(RESULTS_DIR, "tables", "summary.csv")
    os.makedirs(os.path.dirname(global_summary_csv), exist_ok=True)

    log_path = os.path.join(log_dir, "transformer_run.log")
    tee = TeeLogger(log_path)
    sys.stdout = tee
    sys.stderr = tee

    print("=" * 70)
    print("WaveformTransformer — v4.1 (fix: backward-through-graph error)")
    print("=" * 70)
    print(f"Run dir     : {run_dir}")
    print(f"Git commit  : {git_hash}")
    print(f"Device      : {DEVICE}   AMP: {AMP}   NUM_WORKERS: {NUM_WORKERS}")
    print("Taksonomi   : sonic & thunder TETAP terpisah (LABEL_MAP eksplisit)")
    print(f"Normalisasi : {NORM_DESC}   Bandpass: {BANDPASS_ENABLED}")
    print(f"Imbalance   : {IMBALANCE_STRATEGY}")

    print("\n[1] Loading metadata...")
    meta = np.load(META_FILE, allow_pickle=True).item()
    labels_raw = meta["label"]
    print("Metadata keys :", list(meta.keys()))
    print("Unique labels (raw) :", np.unique(labels_raw))

    print("\n[2] Loading memmap waveform...")
    shape = tuple(meta["shape"])
    dtype = np.dtype(meta["dtype"])
    print("Shape :", shape, " Dtype :", dtype)
    data = np.memmap(DATA_FILE, dtype=dtype, mode="r", shape=shape)
    N, C, T = shape
    print("Total samples :", N)

    print("\n[3] Loading shared split (data/splits_5s.npz)...")
    if not os.path.exists(SPLIT_FILE):
        raise FileNotFoundError(
            f"Split bersama tidak ditemukan: {SPLIT_FILE}\n"
            f"Script ini TIDAK membuat split sendiri (P0-3) — jalankan dulu "
            f"scripts/make_splits.py SEKALI untuk membuat {SPLIT_FILE}, lalu "
            f"commit script-nya & catat SHA256-nya di EXPERIMENTS.md."
        )

    split = np.load(SPLIT_FILE, allow_pickle=True)
    train_idx = split["train"]
    val_idx = split["val"]
    test_idx = split["test"]
    classes = list(split["classes"])
    num_classes = len(classes)
    split_type = str(split["split_type"]) if "split_type" in split else "unknown"
    split_sha = sha256sum(SPLIT_FILE)

    print(f"Memuat split: {SPLIT_FILE}")
    print(f"  split_type   : {split_type}")
    print(f"  split_sha256 : {split_sha}")
    print(f"  Classes (urutan dari split file) : {classes}")
    print(f"Train : {len(train_idx)}  ({len(train_idx)/N*100:.1f}%)")
    print(f"Val   : {len(val_idx)}   ({len(val_idx)/N*100:.1f}%)")
    print(f"Test  : {len(test_idx)}  ({len(test_idx)/N*100:.1f}%)")

    report_caveat = None if "event" in split_type.lower() else LEAKAGE_CAVEAT
    if report_caveat:
        print(f"[§1.3] CAVEAT WAJIB pada semua angka run ini: {report_caveat}")

    print("\n[4] Encoding labels (LABEL_MAP kanonik, urutan = classes dari split)...")
    unmapped = sorted(set(str(l) for l in labels_raw) - set(LABEL_MAP.keys()))
    if unmapped:
        raise ValueError(
            f"Ditemukan label folder yang tidak ada di LABEL_MAP kanonik: {unmapped}. "
            f"Perbarui LABEL_MAP sebelum lanjut (P0-2) — JANGAN biarkan string "
            f"'memmap*' lolos ke output."
        )

    label_names = np.array([LABEL_MAP[str(l)] for l in labels_raw])
    print("Unique labels (after canonical map) :", np.unique(label_names))

    unseen_in_split = sorted(set(label_names) - set(classes))
    if unseen_in_split:
        raise ValueError(
            f"Kelas {unseen_in_split} muncul di data tapi TIDAK ada di "
            f"classes={classes} (dari {SPLIT_FILE}). Split file tidak cocok "
            f"dengan LABEL_MAP saat ini — regenerasi split atau perbaiki LABEL_MAP."
        )

    label2idx = {name: i for i, name in enumerate(classes)}
    labels = np.array([label2idx[name] for name in label_names], dtype=np.int64)

    print("Classes :", classes)
    print("Num classes :", num_classes)

    class_counts = np.bincount(labels, minlength=num_classes)
    for cls, cnt in zip(classes, class_counts):
        print(f"  {cls:<15}: {cnt:>10,}")

    train_class_counts = np.bincount(labels[train_idx], minlength=num_classes).astype(float)
    class_weights = 1.0 / np.sqrt(train_class_counts + 1e-9)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    print("\n[P1-3] Bobot kelas (1/sqrt(count), train-only):")
    for cls, w in zip(classes, class_weights):
        print(f"  {cls:<15}: {w:.3f}")

    criterion = build_criterion(IMBALANCE_STRATEGY, class_weights_t)

    print("\n[5] Building dataloader...")
    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0)
    )

    train_loader = DataLoader(EqDataset(data, labels, train_idx, augment=True), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(EqDataset(data, labels, val_idx), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(EqDataset(data, labels, test_idx), shuffle=False, **loader_kwargs)

    print("\n[6] Building WaveformTransformer (TEACHER)...")
    teacher_config = dict(
        in_channels=C, seq_len=SEQ_LEN, patch_size=PATCH_SIZE, d_model=D_MODEL,
        nhead=NHEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FF,
        dropout=DROPOUT, num_classes=num_classes
    )
    model = WaveformTransformer(**teacher_config).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params : {total_params:,}")
    print(f"Num tokens       : {SEQ_LEN // PATCH_SIZE} (+ 1 CLS)")

    full_run_config = dict(
        teacher=teacher_config, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR,
        label_map=LABEL_MAP, classes=list(classes), norm=NORM_DESC,
        bandpass_enabled=BANDPASS_ENABLED, imbalance_strategy=IMBALANCE_STRATEGY,
        split_type=split_type, split_sha256=split_sha, git_commit=git_hash,
    )
    run_meta = dict(
        classes=list(classes), config=full_run_config, norm=NORM_DESC,
        split_file=SPLIT_FILE, split_sha256=split_sha, split_type=split_type,
        git_commit=git_hash,
    )

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    print("\n[7] Training TEACHER...")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")
    patience_ctr = 0
    best_model_path = os.path.join(ckpt_dir, "best_transformer_5s.pt")

    for epoch in tqdm(range(EPOCHS), desc="Epochs", file=sys.__stdout__):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, train=False)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["lr"].append(current_lr)

        if vl_loss < best_val_loss - MIN_DELTA:
            best_val_loss = vl_loss
            patience_ctr = 0
            save_checkpoint(best_model_path, model, epoch + 1, vl_loss, run_meta)
            marker = " <-- BEST"
        else:
            patience_ctr += 1
            marker = f" | patience {patience_ctr}/{PATIENCE}"

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | LR {current_lr:.2e} | "
              f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
              f"Val Loss {vl_loss:.4f} Acc {vl_acc:.4f}" + marker)

        if EARLY_STOPPING and patience_ctr >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    save_json(history, os.path.join(log_dir, "history.json"))

    print("\n[8] Evaluation TEACHER on Test Set...")
    model.load_state_dict(load_checkpoint_state_dict(best_model_path))
    model.eval()

    all_preds, all_true, all_probs = evaluate_full(model, test_loader, DEVICE)

    print("\n--- Classification Report (TEACHER) ---")
    if report_caveat:
        print(f"[CAVEAT] {report_caveat}")
    print(classification_report(all_true, all_preds, target_names=classes, digits=4))

    acc = accuracy_score(all_true, all_preds)
    bal_acc = balanced_accuracy_score(all_true, all_preds)
    top2_acc = top_k_accuracy_score(all_true, all_probs, k=2) if num_classes >= 3 else None
    try:
        auc = roc_auc_score(all_true, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
        print("[P3-note] roc_auc_score gagal (kemungkinan ada kelas tanpa sampel positif di test).")

    print("\n--- Test Set Summary (TEACHER) ---")
    print(f"  Accuracy          : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Balanced Accuracy : {bal_acc:.4f} ({bal_acc*100:.2f}%)")
    if top2_acc is not None:
        print(f"  Top-2 Accuracy    : {top2_acc:.4f} ({top2_acc*100:.2f}%)")
    print(f"  ROC-AUC (macro)   : {auc:.4f}")

    print("\n--- Per-class ROC-AUC & PR-AUC (P1-4) ---")
    per_class_metrics = {}
    for i, cls in enumerate(classes):
        bin_true = (all_true == i).astype(int)
        try:
            cls_auc = roc_auc_score(bin_true, all_probs[:, i])
        except ValueError:
            cls_auc = float("nan")
            print(f"  [P3-note] ROC-AUC gagal untuk kelas '{cls}' (tidak ada positif di test).")
        try:
            cls_prauc = average_precision_score(bin_true, all_probs[:, i])
        except ValueError:
            cls_prauc = float("nan")
        per_class_metrics[cls] = (cls_auc, cls_prauc)
        print(f"  {cls:<15}: ROC-AUC={cls_auc:.4f}  PR-AUC={cls_prauc:.4f}  "
              f"(support={int((all_true==i).sum())})")

    print("\n--- Pandangan biner EEW: earthquake vs rest (P1-4) ---")
    eq_idx = list(classes).index(EQ_CLASS_NAME)

    val_preds_b, val_true_b, val_probs_b = evaluate_full(model, val_loader, DEVICE)
    val_true_bin = (val_true_b == eq_idx).astype(int)
    val_prob_eq = val_probs_b[:, eq_idx]

    thr_5pct = find_threshold_for_ftr(val_true_bin, val_prob_eq, target_ftr=0.05)

    test_true_bin = (all_true == eq_idx).astype(int)
    test_prob_eq = all_probs[:, eq_idx]
    test_pred_at_thr = (test_prob_eq >= thr_5pct).astype(int)
    tp = np.sum((test_pred_at_thr == 1) & (test_true_bin == 1))
    fn = np.sum((test_pred_at_thr == 0) & (test_true_bin == 1))
    fp = np.sum((test_pred_at_thr == 1) & (test_true_bin == 0))
    tn = np.sum((test_pred_at_thr == 0) & (test_true_bin == 0))
    recall_at_ftr5 = tp / (tp + fn + 1e-12)
    ftr_test = fp / (fp + tn + 1e-12)

    print(f"  Threshold (dipilih di VAL utk FTR<=5%) : {thr_5pct:.3f}")
    print(f"  FTR di TEST pada threshold ini          : {ftr_test*100:.2f}%")
    print(f"  Recall gempa di TEST pada FTR<=5%       : {recall_at_ftr5:.4f} ({recall_at_ftr5*100:.2f}%)")

    snr_breakdown = {}
    if "snr_db" in meta:
        snr_all = np.asarray(meta["snr_db"], dtype=float)
        snr_test = snr_all[test_idx]
        buckets = [("<10dB", snr_test < 10), ("10-20dB", (snr_test >= 10) & (snr_test <= 20)), (">20dB", snr_test > 20)]
        print("\n--- Breakdown per-SNR (P1-4) ---")
        for name, mask in buckets:
            mask = mask & ~np.isnan(snr_test)
            n_b = int(mask.sum())
            if n_b == 0:
                print(f"  {name:<10}: tidak ada sampel valid")
                continue
            acc_b = accuracy_score(all_true[mask], all_preds[mask])
            snr_breakdown[name] = {"n": n_b, "acc": acc_b}
            print(f"  {name:<10}: n={n_b:>8,}  acc={acc_b:.4f}")
    else:
        print("\n[P1-4] 'snr_db' tidak ditemukan di metadata — breakdown per-SNR dilewati.")

    print("\n--- Kalibrasi (ECE) ---")
    conf = all_probs.max(axis=1)
    correct_mask = (all_preds == all_true).astype(float)

    ece, bin_stats = compute_ece(conf, correct_mask)
    print(f"  ECE (15 bins) : {ece:.4f}")

    fig_cal, ax_cal = plt.subplots(figsize=(5, 5))
    if bin_stats:
        confs_b, accs_b, counts_b = zip(*bin_stats)
        ax_cal.bar(confs_b, accs_b, width=0.06, alpha=0.7, edgecolor="black", label="Model")
    ax_cal.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax_cal.set_xlabel("Confidence")
    ax_cal.set_ylabel("Accuracy")
    ax_cal.set_title(f"Reliability Diagram (ECE={ece:.4f})")
    ax_cal.legend()
    ax_cal.grid(alpha=0.3)
    fig_cal.savefig(os.path.join(fig_dir, "calibration_reliability.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_cal)

    print("\n[9] Plotting...")
    caveat_title = f"  [{report_caveat}]" if report_caveat else ""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"WaveformTransformer (Teacher) — Test Acc: {acc*100:.2f}% | ROC-AUC: {auc:.4f}{caveat_title}",
                 fontsize=11, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train", linewidth=2)
    ax.plot(history["val_loss"], label="Val", linewidth=2)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(history["train_acc"], label="Train", linewidth=2)
    ax.plot(history["val_acc"], label="Val", linewidth=2)
    ax.axhline(acc, color="red", linestyle="--", linewidth=1.2, label=f"Test={acc:.4f}")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(history["lr"], color="darkorange", linewidth=2)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    cm = confusion_matrix(all_true, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax, vmin=0, vmax=1)
    ax.set_title("Confusion Matrix (normalised)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    plt.tight_layout()
    plot_path = os.path.join(fig_dir, "transformer_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_path = os.path.join(table_dir, "transformer_test_metrics.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("WaveformTransformer (Teacher) — Test Set Metrics\n")
        f.write("=" * 60 + "\n")
        f.write(f"Split type   : {split_type}\n")
        f.write(f"Split SHA256 : {split_sha}\n")
        if report_caveat:
            f.write(f"CAVEAT       : {report_caveat}\n")
        f.write(f"Git commit   : {git_hash}\n\n")
        f.write(f"Accuracy          : {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Balanced Accuracy : {bal_acc:.4f} ({bal_acc*100:.2f}%)\n")
        if top2_acc is not None:
            f.write(f"Top-2 Accuracy    : {top2_acc:.4f} ({top2_acc*100:.2f}%)\n")
        f.write(f"ROC-AUC (macro)   : {auc:.4f}\n")
        f.write(f"ECE (15 bins)     : {ece:.4f}\n")
        f.write(f"Recall gempa @ FTR<=5% (test) : {recall_at_ftr5:.4f}  (threshold={thr_5pct:.3f}, FTR aktual={ftr_test*100:.2f}%)\n\n")
        f.write("--- Per-class ROC-AUC / PR-AUC ---\n")
        for cls, (a_, p_) in per_class_metrics.items():
            f.write(f"  {cls:<15}: ROC-AUC={a_:.4f}  PR-AUC={p_:.4f}\n")
        if snr_breakdown:
            f.write("\n--- Breakdown per-SNR ---\n")
            for name, st in snr_breakdown.items():
                f.write(f"  {name:<10}: n={st['n']:,}  acc={st['acc']:.4f}\n")
        f.write("\n--- Classification Report ---\n")
        f.write(classification_report(all_true, all_preds, target_names=classes, digits=4))

    run_name = os.path.basename(run_dir)
    write_header = not os.path.exists(global_summary_csv)
    with open(global_summary_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["run_name", "git_commit", "model", "split_type", "split_sha256",
                        "accuracy", "balanced_accuracy", "roc_auc_macro", "ece",
                        "recall_eq_at_ftr5", "leakage_caveat"])
        w.writerow([run_name, git_hash, "transformer_teacher", split_type, split_sha,
                    f"{acc:.4f}", f"{bal_acc:.4f}", f"{auc:.4f}", f"{ece:.4f}",
                    f"{recall_at_ftr5:.4f}", "yes" if report_caveat else "no"])

    print("\nFile yang disimpan (TEACHER):")
    print(f"  Plot      : {plot_path}")
    print(f"  Kalibrasi : {os.path.join(fig_dir, 'calibration_reliability.png')}")
    print(f"  Model     : {best_model_path}")
    print(f"  Metrics   : {summary_path}")
    print(f"  summary.csv (global, append) : {global_summary_csv}")

    # ====================================================================
    # ===================  BAGIAN KOMPRESI  ==============================
    # ====================================================================

    print("\n" + "=" * 60)
    print("[10] Membangun STUDENT model (arsitektur efisien)")
    print("=" * 60)

    teacher = model
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student_config = dict(
        in_channels=C, seq_len=SEQ_LEN, patch_size=STUDENT_PATCH_SIZE,
        d_model=STUDENT_D_MODEL, nhead=STUDENT_NHEAD, num_layers=STUDENT_NUM_LAYERS,
        dim_feedforward=STUDENT_DIM_FF, dropout=STUDENT_DROPOUT, num_classes=num_classes
    )
    student = WaveformTransformer(**student_config).to(DEVICE)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher params : {teacher_params:,}")
    print(f"Student params : {student_params:,} ({student_params/teacher_params*100:.1f}% dari teacher)")

    student_ckpt_path = os.path.join(ckpt_dir, "student_distilled.pt")
    student_pruned_ckpt_path = os.path.join(ckpt_dir, "student_pruned.pt")
    student_quant_ckpt_path = os.path.join(ckpt_dir, "student_quantized.pt")
    compression_summary_path = os.path.join(table_dir, "compression_comparison.txt")

    # ------------------------------------------------------
    # [12] Knowledge Distillation: Teacher -> Student
    # ------------------------------------------------------
    print("\n[11] Knowledge Distillation (training STUDENT)...")

    distill_optimizer = optim.AdamW(student.parameters(), lr=DISTILL_LR, weight_decay=1e-2)
    distill_scheduler = CosineAnnealingLR(distill_optimizer, T_max=DISTILL_EPOCHS, eta_min=1e-6)

    best_student_val_loss = float("inf")
    patience_ctr = 0
    for epoch in tqdm(range(DISTILL_EPOCHS), desc="KD-Epochs", file=sys.__stdout__):
        tr_loss, tr_acc = run_distill_epoch(teacher, student, train_loader, distill_optimizer,
                                             class_weights_t, train=True)
        vl_loss, vl_acc = run_distill_epoch(teacher, student, val_loader, distill_optimizer,
                                             class_weights_t, train=False)
        distill_scheduler.step()
        if vl_loss < best_student_val_loss - MIN_DELTA:
            best_student_val_loss = vl_loss
            patience_ctr = 0
            save_checkpoint(student_ckpt_path, student, epoch + 1, vl_loss, run_meta,
                             extra_config={"student": student_config})
            marker = " <-- BEST"
        else:
            patience_ctr += 1
            marker = f" | patience {patience_ctr}/{PATIENCE}"
        print(f"KD Epoch {epoch+1:02d}/{DISTILL_EPOCHS} | Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
              f"Val Loss {vl_loss:.4f} Acc {vl_acc:.4f}" + marker)
        if EARLY_STOPPING and patience_ctr >= PATIENCE:
            print(f"\nEarly stopping (KD) triggered at epoch {epoch+1}")
            break

    student.load_state_dict(load_checkpoint_state_dict(student_ckpt_path))
    s_preds, s_true, s_probs = evaluate_full(student, test_loader, DEVICE)
    student_acc = accuracy_score(s_true, s_preds)
    print(f"\nSTUDENT (distilled) Test Accuracy : {student_acc:.4f} ({student_acc*100:.2f}%)")
    if report_caveat:
        print(f"[CAVEAT] {report_caveat}")

    # ------------------------------------------------------
    # [13] Pruning: global L1 unstructured pada semua nn.Linear (STUDENT)
    # ------------------------------------------------------
    print("\n[12] Pruning STUDENT (global L1 unstructured)...")

    prunable_params = [
        (m, "weight") for m in student.modules()
        if isinstance(m, nn.Linear) and type(m).__name__ != "NonDynamicallyQuantizableLinear"
    ]
    # FIX (root cause, bukan v4.1): nn.MultiheadAttention.self_attn.out_proj
    # adalah instance nn.Linear (subclass internal "NonDynamicallyQuantizableLinear"),
    # jadi ikut terjaring isinstance(m, nn.Linear) di atas -- tapi MultiheadAttention
    # TIDAK pernah memanggil out_proj(x) sebagai module call; ia membaca tensor
    # out_proj.weight langsung di level functional (F.multi_head_attention_forward).
    # Akibatnya forward_pre_hook yang dipasang torch.nn.utils.prune (yang seharusnya
    # menghitung ulang weight = weight_orig * weight_mask SETIAP forward) tidak
    # pernah terpicu untuk out_proj -- weight-nya beku sebagai SATU graph node yang
    # dihitung sekali saat prune.global_unstructured() dipanggil, lalu dipakai ulang
    # di semua batch. Backward batch pertama sukses (graph baru dipakai sekali),
    # backward batch kedua gagal: "Trying to backward through the graph a second
    # time" karena buffer graph itu sudah dibuang sejak backward batch pertama.
    # Ini bug resmi PyTorch (pytorch/pytorch#69353), BUKAN bug di script ini --
    # solusinya adalah mengecualikan out_proj dari pruning, bukan menambah
    # retain_graph/detach di loop training (sudah dicoba di v4.1, tidak menyelesaikan
    # akar masalah karena masalahnya ada di forward MultiheadAttention, bukan di KD loop).
    print(f"Jumlah layer Linear yang di-prune : {len(prunable_params)} "
          f"(out_proj MultiheadAttention dikecualikan -- lihat pytorch/pytorch#69353)")

    prune.global_unstructured(prunable_params, pruning_method=prune.L1Unstructured, amount=PRUNE_AMOUNT)

    # FIX: hitung sparsity dengan no_grad eksplisit -- m.weight di sini adalah
    # computed tensor (weight_orig * weight_mask) lewat forward pre-hook milik
    # torch.nn.utils.prune. Membaca .weight di luar no_grad tetap membangun
    # sedikit graph yang tidak perlu karena weight_orig.requires_grad=True;
    # ini tidak berbahaya tapi tidak rapi, jadi dibungkus no_grad supaya bersih.
    with torch.no_grad():
        total_w = sum(m.weight.nelement() for m, _ in prunable_params)
        zero_w = sum(int(torch.sum(m.weight == 0)) for m, _ in prunable_params)
    print(f"Sparsity setelah pruning : {zero_w/total_w*100:.2f}% bobot Linear = 0")

    print("\n[12b] Fine-tuning STUDENT setelah pruning...")
    # FIX: optimizer baru dibuat SETELAH pruning, jadi sudah menunjuk ke
    # weight_orig (parameter baru hasil reparametrisasi), bukan weight lama.
    ft_optimizer = optim.AdamW(student.parameters(), lr=DISTILL_LR * 0.2, weight_decay=1e-2)
    best_ft_val_loss = float("inf")
    for epoch in tqdm(range(PRUNE_FINETUNE_EPOCHS), desc="Prune-FT", file=sys.__stdout__):
        tr_loss, tr_acc = run_distill_epoch(teacher, student, train_loader, ft_optimizer,
                                             class_weights_t, train=True)
        vl_loss, vl_acc = run_distill_epoch(teacher, student, val_loader, ft_optimizer,
                                             class_weights_t, train=False)
        print(f"Prune-FT Epoch {epoch+1}/{PRUNE_FINETUNE_EPOCHS} | Train Loss {tr_loss:.4f} "
              f"Acc {tr_acc:.4f} | Val Loss {vl_loss:.4f} Acc {vl_acc:.4f}")
        best_ft_val_loss = min(best_ft_val_loss, vl_loss)

    for module, name in prunable_params:
        prune.remove(module, name)

    save_checkpoint(student_pruned_ckpt_path, student, PRUNE_FINETUNE_EPOCHS, best_ft_val_loss, run_meta,
                     extra_config={"student": student_config, "prune_amount": PRUNE_AMOUNT,
                                   "sparsity_actual": zero_w / total_w})

    p_preds, p_true, p_probs = evaluate_full(student, test_loader, DEVICE)
    pruned_acc = accuracy_score(p_true, p_preds)
    print(f"\nSTUDENT (pruned, {PRUNE_AMOUNT*100:.0f}% sparsity) Test Accuracy : "
          f"{pruned_acc:.4f} ({pruned_acc*100:.2f}%)")

    # ------------------------------------------------------
    # [14] Quantization: Dynamic INT8 (CPU-only)
    # ------------------------------------------------------
    print("\n[13] Dynamic Quantization STUDENT (INT8, nn.Linear)...")

    torch.backends.quantized.engine = QUANT_BACKEND
    student_cpu = copy.deepcopy(student).to("cpu").eval()
    quantized_student = torch.quantization.quantize_dynamic(student_cpu, {nn.Linear}, dtype=torch.qint8)
    torch.save({"state_dict": quantized_student.state_dict(), "classes": list(classes),
                "config": {"student": student_config}, "split_type": split_type,
                "split_sha256": split_sha, "git_commit": git_hash}, student_quant_ckpt_path)

    q_preds, q_true, q_probs = evaluate_full(quantized_student, test_loader, "cpu")
    quant_acc = accuracy_score(q_true, q_preds)
    print(f"STUDENT (quantized INT8) Test Accuracy : {quant_acc:.4f} ({quant_acc*100:.2f}%)")

    # ------------------------------------------------------
    # [15] Benchmark
    # ------------------------------------------------------
    print("\n[14] Benchmark perbandingan semua varian model...")

    test_dataset = EqDataset(data, labels, test_idx)

    teacher_cpu = copy.deepcopy(teacher).to("cpu").eval()
    student_distilled_cpu = WaveformTransformer(**student_config)
    student_distilled_cpu.load_state_dict(load_checkpoint_state_dict(student_ckpt_path, map_location="cpu"))
    student_distilled_cpu.eval()
    student_pruned_cpu = copy.deepcopy(student).to("cpu").eval()

    # Setiap varian membawa (label, net-untuk-latency, params, acc, file-checkpoint).
    # params/acc/size/latency dihitung SEKALI per varian dan langsung melekat
    # ke label yang sama -- tidak perlu lagi mencocokkan nama antar dict terpisah.
    variant_specs = [
        ("Teacher (FP32)", teacher_cpu, teacher_params, acc, best_model_path),
        ("Student-Distilled (FP32)", student_distilled_cpu, student_params, student_acc, student_ckpt_path),
        (f"Student-Pruned ({PRUNE_AMOUNT*100:.0f}% sparsity)", student_pruned_cpu,
         student_params, pruned_acc, student_pruned_ckpt_path),
        ("Student-Quantized (INT8)", quantized_student, count_params(quantized_student),
         quant_acc, student_quant_ckpt_path),
    ]

    variants = [
        dict(
            label=label,
            params=params,
            acc=variant_acc,
            size_mb=file_size_mb(ckpt_path),
            latency={bs: measure_latency_ms(net, test_dataset, "cpu", bs) for bs in BENCH_BATCH_SIZES},
        )
        for label, net, params, variant_acc, ckpt_path in variant_specs
    ]

    table_lines = format_comparison_table(variants, BENCH_BATCH_SIZES)

    print("\n" + "=" * 60)
    print("RINGKASAN KOMPRESI" + (f"  [{report_caveat}]" if report_caveat else ""))
    print("=" * 60)
    for line in table_lines:
        print(line)

    with open(compression_summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("RINGKASAN KOMPRESI — WaveformTransformer\n")
        f.write("=" * 70 + "\n")
        if report_caveat:
            f.write(f"CAVEAT: {report_caveat}\n")
        f.write(f"Split type: {split_type}  SHA256: {split_sha}\n\n")
        for line in table_lines:
            f.write(line + "\n")
        f.write("\nTeknik yang dipakai:\n")
        f.write(f"  1. Arsitektur efisien : patch {PATCH_SIZE}->{STUDENT_PATCH_SIZE}, "
                f"d_model {D_MODEL}->{STUDENT_D_MODEL}, layers {NUM_LAYERS}->{STUDENT_NUM_LAYERS}\n")
        f.write(f"  2. Knowledge Distillation : T={DISTILL_TEMPERATURE}, alpha={DISTILL_ALPHA}\n")
        f.write(f"  3. Pruning : global L1 unstructured, amount={PRUNE_AMOUNT}, "
                f"sparsity aktual={zero_w/total_w*100:.2f}%\n")
        f.write(f"  4. Dynamic Quantization : INT8 pada semua nn.Linear, backend={QUANT_BACKEND}\n")

    print("\nFile yang disimpan (KOMPRESI):")
    print(f"  Student (distilled) : {student_ckpt_path}")
    print(f"  Student (pruned)    : {student_pruned_ckpt_path}")
    print(f"  Student (quantized) : {student_quant_ckpt_path}")
    print(f"  Ringkasan kompresi  : {compression_summary_path}")
    print(f"  Log                 : {log_path}")
    print(f"  Semua artefak run ini ada di: {run_dir}")
    print("\nDONE")


if __name__ == "__main__":
    main()
