#!/usr/bin/env python
"""Convert STEAD earthquake HDF5 → NumPy memmap untuk label 'no_event'.

Label no_event diambil dari data EARTHQUAKE, yaitu segmen SEBELUM P-wave
datang (pre-P-arrival window), mulai dari detik ke-0.

Logika pemotongan:
    window = waveform[0 : min(p_arrival_sample, window_samples)]

Jika P-arrival lebih kecil dari window yang diminta, trace DILEWATI
(tidak cukup pre-event noise) — threshold dikontrol via --min-pre-p.

Menghasilkan file memmap (N, 3, T) float32 dengan akses random O(1).
Pemrosesan dilakukan per chunk untuk menghindari OOM.

Penggunaan:
    # Semua variant (3s, 5s, 10s):
    python convert_no_event.py

    # Window tertentu saja:
    python convert_no_event.py --windows 3 5

    # Lanjut setelah interupsi:
    python convert_no_event.py --resume

    # Minimal pre-P ratio (default: 1.0 = harus >= window penuh):
    python convert_no_event.py --min-pre-p 0.8
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SAMPLING_RATE      = 100
FULL_SAMPLES       = 6000
DEFAULT_CHUNK_SIZE = 100_000


# =========================================================
# Extract window pre-P-arrival
# =========================================================
def extract_pre_p_window(full_waveform: np.ndarray,
                          p_arrival_sample: int,
                          window_samples: int) -> np.ndarray:
    """
    Ambil window dari detik 0, TIDAK melewati P-wave.
        end = min(window_samples, p_arrival_sample)
    Ini memastikan pure pre-event signal.
    """
    out = np.zeros((3, window_samples), dtype=np.float32)
    end = min(window_samples, p_arrival_sample)
    if end > 0:
        out[:, :end] = full_waveform[:, :end]
    return out


# =========================================================
# Parse p_arrival_sample dari CSV
# =========================================================
def parse_p_arrival(df: pd.DataFrame) -> np.ndarray:
    for c in ["p_arrival_sample", "p_arrival_idx", "p_sample_idx", "p_pick_sample"]:
        if c in df.columns:
            arr = pd.to_numeric(df[c], errors="coerce").values
            return np.where(np.isnan(arr), -1, arr).astype(np.int32)
    raise ValueError(
        "Kolom p_arrival tidak ditemukan! Kolom tersedia: "
        + str(df.columns.tolist())
    )


# =========================================================
# Deteksi titik resume
# =========================================================
def _detect_resume_point(output_dir: Path, window_specs: list, n: int) -> int:
    check_name, check_samples = None, 0
    for name, ns in window_specs:
        if ns > check_samples:
            check_name, check_samples = name, ns

    if check_name is None:
        return 0

    fpath = output_dir / f"noevent_waveforms_{check_name}.npy"
    if not fpath.exists():
        return 0

    mm = np.memmap(fpath, dtype=np.float32, mode="r",
                   shape=(n, 3, check_samples))

    if np.any(mm[-1] != 0):
        del mm
        return n

    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if np.any(mm[mid] != 0):
            lo = mid
        else:
            hi = mid - 1

    resume_from = lo + 1 if np.any(mm[lo] != 0) else 0
    del mm
    return resume_from


# =========================================================
# Fungsi utama konversi
# =========================================================
def convert(hdf5_path, csv_path, output_dir, windows,
            min_pre_p_ratio=1.0, chunk_size=DEFAULT_CHUNK_SIZE,
            resume=False):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load metadata CSV
    # --------------------------------------------------
    print(f"Memuat metadata dari {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)

    if "trace_category" in df.columns:
        before = len(df)
        df = df[df["trace_category"] == "earthquake_local"].reset_index(drop=True)
        print(f"  Filter earthquake_local: {before:,} → {len(df):,} trace")
    else:
        print("  Kolom 'trace_category' tidak ditemukan — semua baris dipakai.")
        df = df.reset_index(drop=True)

    p_arrivals = parse_p_arrival(df)

    # --------------------------------------------------
    # Window specs
    # --------------------------------------------------
    window_specs = []
    for w in windows:
        sec  = float(w)
        samp = int(sec * SAMPLING_RATE)
        name = f"{int(sec)}s" if sec == int(sec) else f"{sec}s"
        window_specs.append((name, samp))

    # --------------------------------------------------
    # Filter trace: P-arrival >= window_samples * ratio
    # --------------------------------------------------
    max_window_samp = max(ns for _, ns in window_specs)
    min_p_required  = int(max_window_samp * min_pre_p_ratio)

    valid_mask = (p_arrivals >= min_p_required)
    print(f"  Dilewati (P-arrival terlalu dekat): {(~valid_mask).sum():,} trace")

    df         = df[valid_mask].reset_index(drop=True)
    p_arrivals = p_arrivals[valid_mask]
    trace_names = df["trace_name"].values
    n = len(trace_names)

    print(f"  Trace valid untuk no_event: {n:,}")

    if n == 0:
        raise ValueError(
            "Tidak ada trace valid! Coba turunkan --min-pre-p (mis. 0.5 atau 0.0)."
        )

    # --------------------------------------------------
    # Statistik P-arrival
    # --------------------------------------------------
    print(f"\nStatistik P-arrival:")
    print(f"  min  = {p_arrivals.min():,} samples ({p_arrivals.min()/SAMPLING_RATE:.2f}s)")
    print(f"  mean = {p_arrivals.mean():.0f} samples ({p_arrivals.mean()/SAMPLING_RATE:.2f}s)")
    print(f"  max  = {p_arrivals.max():,} samples ({p_arrivals.max()/SAMPLING_RATE:.2f}s)")

    # --------------------------------------------------
    # Buat / buka memmap
    # --------------------------------------------------
    memmaps = {}
    for name, ns in window_specs:
        shape  = (n, 3, ns)
        mem_gb = n * 3 * ns * 4 / (1024**3)
        fname  = f"noevent_waveforms_{name}.npy"
        fpath  = output_dir / fname

        if resume and fpath.exists():
            mode = "r+"
            print(f"Membuka memmap (resume): {fname}  shape={shape}  ({mem_gb:.2f} GB)")
        else:
            mode = "w+"
            print(f"Membuat memmap baru   : {fname}  shape={shape}  ({mem_gb:.2f} GB)")

        memmaps[name] = (np.memmap(fpath, dtype=np.float32, mode=mode, shape=shape), ns)

    # --------------------------------------------------
    # Deteksi titik resume
    # --------------------------------------------------
    start_idx = 0
    if resume:
        start_idx = _detect_resume_point(output_dir, window_specs, n)
        if start_idx > 0:
            print(f"\nMelanjutkan dari trace {start_idx:,} ({start_idx/n*100:.1f}%)")
        else:
            print("\nTidak ada progres sebelumnya, mulai dari awal")

    # --------------------------------------------------
    # Baca HDF5 → isi memmap
    # --------------------------------------------------
    if start_idx >= n:
        print("Semua trace sudah dikonversi sebelumnya!")
    else:
        print(f"\nMengkonversi {n - start_idx:,} trace dari {hdf5_path} ...")
        t0 = time.time()
        traces_written = 0

        with h5py.File(hdf5_path, "r") as f:
            for i in range(start_idx, n):

                p_arr = p_arrivals[i]
                if p_arr <= 0:
                    continue

                # HDF5 STEAD: (6000, 3) → transpose → (3, 6000)
                raw = np.array(f[f"data/{trace_names[i]}"], dtype=np.float32).T

                for name, ns in window_specs:
                    memmaps[name][0][i] = extract_pre_p_window(raw, p_arr, ns)

                traces_written += 1

                # Flush per chunk
                if traces_written % chunk_size == 0:
                    for name, (mm, _) in memmaps.items():
                        mm.flush()
                    elapsed    = time.time() - t0
                    rate       = traces_written / elapsed
                    total_done = start_idx + traces_written
                    eta        = (n - total_done) / rate
                    print(f"  {total_done:>10,}/{n:,} ({total_done/n*100:5.1f}%)  "
                          f"{rate:.0f} trace/s  ETA {eta/60:.1f}min  [flushed]")

                elif traces_written % 10_000 == 0:
                    elapsed    = time.time() - t0
                    rate       = traces_written / elapsed
                    total_done = start_idx + traces_written
                    eta        = (n - total_done) / rate
                    print(f"  {total_done:>10,}/{n:,} ({total_done/n*100:5.1f}%)  "
                          f"{rate:.0f} trace/s  ETA {eta/60:.1f}min")

        # Flush akhir
        for name, (mm, _) in memmaps.items():
            mm.flush()
            del mm

        elapsed = time.time() - t0
        print(f"\nSelesai: {traces_written:,} trace dalam {elapsed:.0f}s "
              f"({traces_written/elapsed:.0f} trace/s)")

    # --------------------------------------------------
    # Simpan trace_names & p_arrivals
    # --------------------------------------------------
    np.save(output_dir / "noevent_trace_names.npy", trace_names)
    np.save(output_dir / "noevent_p_arrivals.npy", p_arrivals)

    # --------------------------------------------------
    # Simpan metadata.npz
    # --------------------------------------------------
    def _safe_float(col):
        return pd.to_numeric(df[col], errors="coerce").values.astype(np.float32)

    save_kwargs = {"p_arrival_sample": p_arrivals,
                   "trace_index": np.arange(n, dtype=np.int32)}

    for col in ["network_code", "receiver_code", "receiver_type",
                "receiver_latitude", "receiver_longitude", "receiver_elevation_m",
                "source_magnitude", "source_depth_km", "source_distance_km",
                "trace_start_time"]:
        if col in df.columns:
            if df[col].dtype == object:
                save_kwargs[col] = df[col].values.astype(str)
            else:
                save_kwargs[col] = _safe_float(col)

    np.savez(output_dir / "noevent_metadata.npz", **save_kwargs)

    # --------------------------------------------------
    # Simpan index.json
    # --------------------------------------------------
    index = {
        "dataset"         : "STEAD_earthquake_pre_P (no_event label)",
        "n_traces"        : n,
        "window_start"    : "sample_0 (detik ke-0)",
        "window_end"      : "min(window_samples, p_arrival_sample)",
        "min_pre_p_ratio" : min_pre_p_ratio,
        "sampling_rate"   : SAMPLING_RATE,
        "hdf5_path"       : str(hdf5_path),
        "csv_path"        : str(csv_path),
        "variants"        : {},
    }
    for name, ns in window_specs:
        index["variants"][name] = {
            "file"    : f"noevent_waveforms_{name}.npy",
            "shape"   : [n, 3, ns],
            "dtype"   : "float32",
            "window_s": ns / SAMPLING_RATE,
            "size_gb" : round(n * 3 * ns * 4 / (1024**3), 3),
        }

    with open(output_dir / "noevent_index.json", "w") as fp:
        json.dump(index, fp, indent=2)

    # --------------------------------------------------
    # Ringkasan akhir
    # --------------------------------------------------
    print("\n=== File Output ===")
    for name, ns in window_specs:
        fname   = f"noevent_waveforms_{name}.npy"
        size_gb = (output_dir / fname).stat().st_size / (1024**3)
        print(f"  {fname:<35s}  shape=({n}, 3, {ns:>5})  {size_gb:.3f} GB")
    print(f"  {'noevent_trace_names.npy':<35s}  {n:,} entries")
    print(f"  {'noevent_p_arrivals.npy':<35s}  {n:,} entries (int32)")
    print(f"  {'noevent_metadata.npz':<35s}  magnitude, depth, SNR, dll.")
    print(f"  {'noevent_index.json':<35s}  metadata & shape info")
    print(f"\nOutput directory: {output_dir.resolve()}")


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Konversi STEAD earthquake → memmap label 'no_event' (pre-P window)"
    )
    parser.add_argument(
        "--hdf5", default="/home/indra/indra/STEAD/merged/merge.hdf5")
    parser.add_argument(
        "--csv",  default="/home/indra/indra/STEAD/merged/merge.csv")
    parser.add_argument(
        "--output", default="/home/indra/indra/STEAD/memmap_no_event")  # ← folder yang diminta
    parser.add_argument(
        "--windows", nargs="+", default=["3", "5", "10"], metavar="SEC")
    parser.add_argument(
        "--min-pre-p", type=float, default=1.0, metavar="RATIO",
        help="Rasio min P-arrival/window_size (default 1.0). "
             "Turunkan ke 0.8 atau 0.5 jika terlalu banyak trace dibuang.")
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--resume", action="store_true",
        help="Lanjutkan konversi yang terinterupsi.")

    args = parser.parse_args()

    convert(
        hdf5_path       = args.hdf5,
        csv_path        = args.csv,
        output_dir      = args.output,
        windows         = args.windows,
        min_pre_p_ratio = args.min_pre_p,
        chunk_size      = args.chunk_size,
        resume          = args.resume,
    )


if __name__ == "__main__":
    main()