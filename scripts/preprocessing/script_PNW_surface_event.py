"""
build_memmap_surface_event.py
==============================
Pecah trace PNW Exotic Dataset (kelas surface event, thunder, noise)
menjadi window 3 detik, 5 detik, dan 10 detik, lalu simpan sebagai
numpy memmap (.npy) + metadata (.npz) + index.json.

Struktur output:
    /home/indra/indra/PNW/memmap_surface_event/
    ├── index.json
    ├── metadata.npz
    ├── trace_names.npy
    ├── waveforms_3s.npy
    ├── waveforms_5s.npy
    ├── waveforms_10s.npy
    └── waveforms_full.npy

Cara pakai:
    conda activate pytorch_Py12
    python build_memmap_surface_event.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
import h5py
from scipy.signal import butter, filtfilt
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────────────────────────────────────
HDF5_PATH  = "/home/indra/indra/PNW/exotic_waveforms.hdf5"
CSV_PATH   = "/home/indra/indra/PNW/exotic_metadata.csv"
OUTPUT_DIR = "/home/indra/indra/PNW/memmap_surface_event"

SAMPLE_RATE = 100   # Hz

# Durasi window (detik) → sampel
WINDOW_CONFIGS = {
    "3s":   3  * SAMPLE_RATE,   # 300 sampel
    "5s":   5  * SAMPLE_RATE,   # 500 sampel
    "10s":  10 * SAMPLE_RATE,   # 1000 sampel
    "full": 18001,               # trace penuh
}

# Kelas surface event yang diambil (label = 0)
SURFACE_EVENT_CLASSES = {
    "surface event",
    "thunder",
    "noise",
    "other",
}

# Jumlah window random per trace (untuk 3s, 5s, 10s)
N_RANDOM_WINDOWS = 3

# Split ratio
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15 (sisanya)

SEED = 42

# Filter Butterworth bandpass
BANDPASS_LOW  = 1.0   # Hz
BANDPASS_HIGH = 45.0  # Hz
BUTTER_ORDER  = 4

# ──────────────────────────────────────────────────────────────────────────────
# UTILITY
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq  = 0.5 * fs
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def preprocess_trace(waveform):
    """
    waveform : np.ndarray (3, n_samples)
    Returns  : np.ndarray (3, n_samples) float32, sudah demean + filter + normalize
    """
    out = np.zeros_like(waveform, dtype=np.float32)
    for ch in range(waveform.shape[0]):
        x = waveform[ch].astype(np.float64)
        x = x - np.mean(x)
        try:
            x = butter_bandpass_filter(x, BANDPASS_LOW, BANDPASS_HIGH,
                                       SAMPLE_RATE, BUTTER_ORDER)
        except Exception:
            pass
        peak = np.max(np.abs(x))
        if peak > 1e-9:
            x = x / peak
        out[ch] = x.astype(np.float32)
    return out


def extract_windows_random(waveform, window_samp, n_windows):
    """
    Ambil n_windows window random dari trace.
    Returns: list of np.ndarray (3, window_samp)
    """
    n_samp    = waveform.shape[1]
    max_start = n_samp - window_samp

    if max_start <= 0:
        # Trace lebih pendek dari window → pad dengan zeros
        pad = np.zeros((3, window_samp), dtype=np.float32)
        pad[:, :n_samp] = waveform
        return [pad]

    windows     = []
    starts_used = set()
    attempts    = 0

    while len(windows) < n_windows and attempts < n_windows * 10:
        start = random.randint(0, max_start)
        if start not in starts_used:
            starts_used.add(start)
            w = waveform[:, start:start + window_samp].copy()
            windows.append((w, start))
        attempts += 1

    return windows  # list of (array, start_sample)


def extract_full_trace(waveform, target_len=18001):
    """
    Ambil trace penuh, pad/crop ke target_len.
    Returns: np.ndarray (3, target_len)
    """
    n = waveform.shape[1]
    out = np.zeros((3, target_len), dtype=np.float32)
    copy_len = min(n, target_len)
    out[:, :copy_len] = waveform[:, :copy_len]
    return out


def read_waveform_from_hdf5(f, trace_name):
    """
    trace_name format: "bucket1$8992,:3,:18001"
    """
    try:
        bucket_part = trace_name.split(",")[0]
        bucket, idx = bucket_part.split("$")
        idx  = int(idx)
        data = f["data"][bucket][idx]
        return data.astype(np.float32)
    except Exception as e:
        print(f"  [WARN] Gagal baca trace '{trace_name}': {e}")
        return None


def split_event_ids(event_ids, train_r=TRAIN_RATIO, val_r=VAL_RATIO, seed=SEED):
    rng = np.random.default_rng(seed)
    ids = list(set(event_ids))
    rng.shuffle(ids)
    n       = len(ids)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    train_ids = set(ids[:n_train])
    val_ids   = set(ids[n_train:n_train + n_val])
    test_ids  = set(ids[n_train + n_val:])
    return train_ids, val_ids, test_ids


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Membaca metadata CSV...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Total trace di CSV : {len(df)}")
    print(f"  source_type unik   : {df['source_type'].unique()}")

    # ── Filter hanya surface event classes ──────────────────────────────────
    df["source_type_norm"] = df["source_type"].str.strip().str.lower()
    df_surface = df[df["source_type_norm"].isin(SURFACE_EVENT_CLASSES)].copy()
    df_surface["label"] = 0  # semua non-earthquake

    print(f"\nTrace surface event yang digunakan: {len(df_surface)}")
    for stype in df_surface["source_type"].unique():
        n = (df_surface["source_type"] == stype).sum()
        print(f"  {stype:20s}: {n:6d} trace")

    # ── Split berdasarkan event_id ──────────────────────────────────────────
    print("\nMembagi event ke train/val/test...")
    id_col = "event_id"
    if id_col not in df_surface.columns:
        for c in ["source_id", "pnsn_id"]:
            if c in df_surface.columns:
                id_col = c
                break
        else:
            id_col = "trace_name"
            print("  [WARN] event_id tidak ditemukan, split per trace.")

    train_ids, val_ids, test_ids = split_event_ids(df_surface[id_col].tolist())

    df_surface["split"] = "train"
    df_surface.loc[df_surface[id_col].isin(val_ids),  "split"] = "val"
    df_surface.loc[df_surface[id_col].isin(test_ids), "split"] = "test"

    for sp in ["train", "val", "test"]:
        n_ev = df_surface[df_surface["split"] == sp][id_col].nunique()
        n_tr = (df_surface["split"] == sp).sum()
        print(f"  {sp:5s}: {n_ev:5d} event, {n_tr:6d} trace")

    # ── Ekstrak window dari HDF5 ─────────────────────────────────────────────
    print(f"\nMembuka HDF5: {HDF5_PATH}")

    # Container per window size
    # key: win_key (3s/5s/10s/full) → dict split → list array
    all_windows = {k: defaultdict(list) for k in WINDOW_CONFIGS}
    all_labels  = defaultdict(list)   # split → list label
    all_trace_names = defaultdict(list)   # split → list trace_name
    meta_rows   = []

    with h5py.File(HDF5_PATH, "r") as f:
        for i, (_, row) in enumerate(df_surface.iterrows()):
            trace_name  = row["trace_name"]
            split       = row["split"]
            source_type = row["source_type"]
            snr         = row.get("trace_snr_db", float("nan"))

            waveform = read_waveform_from_hdf5(f, trace_name)
            if waveform is None:
                continue

            waveform = preprocess_trace(waveform)

            # ── Trace penuh ────────────────────────────────────────────────
            full_w = extract_full_trace(waveform, WINDOW_CONFIGS["full"])
            all_windows["full"][split].append(full_w)

            # Untuk full trace: 1 entry per trace
            all_labels[split].append(0)
            all_trace_names[split].append(trace_name)

            meta_rows.append({
                "trace_name":   trace_name,
                "source_type":  source_type,
                "label":        0,
                "split":        split,
                "window_key":   "full",
                "window_start": 0,
                "snr_db":       snr,
            })

            # ── Window 3s, 5s, 10s ────────────────────────────────────────
            for win_key in ["3s", "5s", "10s"]:
                win_samp = WINDOW_CONFIGS[win_key]
                wins = extract_windows_random(waveform, win_samp, N_RANDOM_WINDOWS)
                for (w, start) in wins:
                    all_windows[win_key][split].append(w)
                    meta_rows.append({
                        "trace_name":   trace_name,
                        "source_type":  source_type,
                        "label":        0,
                        "split":        split,
                        "window_key":   win_key,
                        "window_start": start,
                        "snr_db":       snr,
                    })

            if (i + 1) % 500 == 0:
                print(f"  Diproses: {i + 1}/{len(df_surface)} trace...")

    # ── Simpan .npy ─────────────────────────────────────────────────────────
    print("\nMenyimpan file .npy...")

    index_info = {
        "dataset":      "PNW Exotic - Surface Event",
        "sample_rate":  SAMPLE_RATE,
        "bandpass":     [BANDPASS_LOW, BANDPASS_HIGH],
        "n_channels":   3,
        "label_map":    {"0": "surface_event/noise"},
        "splits":       {},
        "files": {
            "waveforms_3s":   "waveforms_3s.npy",
            "waveforms_5s":   "waveforms_5s.npy",
            "waveforms_10s":  "waveforms_10s.npy",
            "waveforms_full": "waveforms_full.npy",
            "metadata":       "metadata.npz",
            "trace_names":    "trace_names.npy",
        }
    }

    # Gabungkan semua split untuk setiap window size
    for win_key, samp in WINDOW_CONFIGS.items():
        fname = f"waveforms_{win_key}.npy"
        fpath = os.path.join(OUTPUT_DIR, fname)

        arrs = []
        for sp in ["train", "val", "test"]:
            arrs.extend(all_windows[win_key][sp])

        if len(arrs) == 0:
            print(f"  [WARN] {win_key}: tidak ada data, skip.")
            continue

        X = np.stack(arrs, axis=0).astype(np.float32)
        np.save(fpath, X)
        print(f"  {fname}: shape={X.shape}  →  {fpath}")

    # ── Simpan trace_names.npy ───────────────────────────────────────────────
    all_names = []
    for sp in ["train", "val", "test"]:
        all_names.extend(all_trace_names[sp])

    trace_names_arr = np.array(all_names, dtype=object)
    tname_path = os.path.join(OUTPUT_DIR, "trace_names.npy")
    np.save(tname_path, trace_names_arr)
    print(f"  trace_names.npy: {len(trace_names_arr)} trace  →  {tname_path}")

    # ── Simpan metadata.npz ──────────────────────────────────────────────────
    meta_df   = pd.DataFrame(meta_rows)
    meta_path = os.path.join(OUTPUT_DIR, "metadata.npz")

    # Simpan setiap kolom sebagai array
    np.savez(
        meta_path,
        trace_name   = meta_df["trace_name"].values.astype(str),
        source_type  = meta_df["source_type"].values.astype(str),
        label        = meta_df["label"].values.astype(np.uint8),
        split        = meta_df["split"].values.astype(str),
        window_key   = meta_df["window_key"].values.astype(str),
        window_start = meta_df["window_start"].values.astype(np.int32),
        snr_db = meta_df["snr_db"].apply(
            lambda x: np.mean([float(v) for v in str(x).split("|")])
        ).values.astype(np.float32)
    )
    print(f"  metadata.npz: {len(meta_df)} entri  →  {meta_path}")

    # Juga simpan CSV versi metadata untuk kemudahan inspeksi
    meta_csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    meta_df.to_csv(meta_csv_path, index=False)
    print(f"  metadata.csv (bonus): {len(meta_df)} baris  →  {meta_csv_path}")

    # ── Hitung statistik per split untuk index.json ──────────────────────────
    for sp in ["train", "val", "test"]:
        n_trace = len(all_trace_names[sp])
        index_info["splits"][sp] = {
            "n_traces":       n_trace,
            "n_windows_3s":   len(all_windows["3s"][sp]),
            "n_windows_5s":   len(all_windows["5s"][sp]),
            "n_windows_10s":  len(all_windows["10s"][sp]),
            "n_windows_full": len(all_windows["full"][sp]),
        }

    index_info["total_traces"] = sum(
        index_info["splits"][sp]["n_traces"] for sp in ["train", "val", "test"]
    )

    # ── Simpan index.json ────────────────────────────────────────────────────
    json_path = os.path.join(OUTPUT_DIR, "index.json")
    with open(json_path, "w") as jf:
        json.dump(index_info, jf, indent=2)
    print(f"  index.json  →  {json_path}")

    # ── Ringkasan akhir ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SELESAI! Struktur output:")
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size  = os.path.getsize(fpath)
        unit  = "KB" if size < 1_000_000 else "MB"
        size_disp = size / 1024 if size < 1_000_000 else size / 1_048_576
        print(f"  {fname:30s}  {size_disp:7.1f} {unit}")
    print("=" * 60)


if __name__ == "__main__":
    main()
