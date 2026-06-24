#!/usr/bin/env python
"""Convert STEAD noise HDF5 to NumPy memmap files for fast training I/O.

Berbeda dengan data earthquake yang menggunakan P-arrival sebagai anchor,
data noise dipotong mulai dari detik ke-0 (sample 0) hingga durasi yang diminta.

Menghasilkan file memmap (N, 3, T) float32 dengan akses random O(1).
Pemrosesan dilakukan per chunk untuk menghindari OOM.

Penggunaan:
    # Semua variant (3s, 5s, 10s):
    python convert_noise_to_memmap.py

    # Window tertentu saja:
    python convert_noise_to_memmap.py --windows 3 5

    # Custom output directory:
    python convert_noise_to_memmap.py --output /fast-nvme/stead_memmap_noise

    # Lanjut setelah interupsi:
    python convert_noise_to_memmap.py --resume
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SAMPLING_RATE = 100          # Hz
FULL_SAMPLES  = 6000         # 60s @ 100 Hz (panjang penuh satu trace STEAD)
DEFAULT_CHUNK_SIZE = 100_000 # trace per flush — batas dirty-page memory


# ---------------------------------------------------------------------------
# Helper: potong window mulai sample 0
# ---------------------------------------------------------------------------
def extract_window_from_zero(full_waveform: np.ndarray,
                              window_samples: int) -> np.ndarray:
    """Ambil window [0 : window_samples] dari waveform (3, 6000).

    Jika trace lebih pendek dari window_samples (seharusnya tidak terjadi
    untuk STEAD), sisanya di-pad dengan nol.
    """
    actual = full_waveform.shape[1]   # seharusnya 6000
    out = np.zeros((3, window_samples), dtype=np.float32)
    end = min(window_samples, actual)
    out[:, :end] = full_waveform[:, :end]
    return out


# ---------------------------------------------------------------------------
# Helper: deteksi titik resume
# ---------------------------------------------------------------------------
def _detect_resume_point(output_dir: Path, window_specs: list, n: int) -> int:
    """Cari berapa trace yang sudah ditulis dengan binary-search pada memmap."""
    # Pilih memmap terbesar sebagai acuan
    check_name, check_samples = None, 0
    for name, n_samp in window_specs:
        if n_samp > check_samples:
            check_name, check_samples = name, n_samp

    if check_name is None:
        return 0

    fpath = output_dir / f"noise_waveforms_{check_name}.npy"
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


# ---------------------------------------------------------------------------
# Fungsi utama konversi
# ---------------------------------------------------------------------------
def convert(hdf5_path: str, csv_path: str, output_dir: str,
            windows: list[str],
            chunk_size: int = DEFAULT_CHUNK_SIZE,
            resume: bool = False):
    """Konversi HDF5 noise → memmap files dengan chunk-based flushing."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Muat metadata CSV
    # ------------------------------------------------------------------ #
    print(f"Memuat metadata dari {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)

    # Filter: hanya baris noise (trace_category == 'noise')
    if "trace_category" in df.columns:
        before = len(df)
        df = df[df["trace_category"] == "noise"].reset_index(drop=True)
        print(f"  Filter noise: {before:,} → {len(df):,} trace")
    else:
        print("  Kolom 'trace_category' tidak ditemukan — semua baris dipakai.")
        df = df.reset_index(drop=True)

    trace_names = df["trace_name"].values
    n = len(trace_names)
    print(f"  {n:,} trace noise akan dikonversi")

    if n == 0:
        raise ValueError("Tidak ada trace noise ditemukan! "
                         "Pastikan CSV berisi kolom 'trace_category' = 'noise'.")

    # ------------------------------------------------------------------ #
    # Parse spesifikasi window
    # ------------------------------------------------------------------ #
    window_specs = []   # list of (name_str, n_samples)
    for w in windows:
        sec = float(w)
        samp = int(sec * SAMPLING_RATE)
        if samp > FULL_SAMPLES:
            raise ValueError(
                f"Window {sec}s ({samp} samples) melebihi panjang trace "
                f"STEAD ({FULL_SAMPLES} samples / 60s).")
        name = f"{int(sec)}s" if sec == int(sec) else f"{sec}s"
        window_specs.append((name, samp))

    # ------------------------------------------------------------------ #
    # Estimasi ukuran
    # ------------------------------------------------------------------ #
    total_gb  = sum(n * 3 * ns * 4 / (1024**3) for _, ns in window_specs)
    chunk_gb  = sum(chunk_size * 3 * ns * 4 / (1024**3) for _, ns in window_specs)
    print(f"\nTotal ukuran memmap : {total_gb:.2f} GB")
    print(f"Chunk size          : {chunk_size:,} trace (~{chunk_gb:.2f} GB dirty pages)")

    # ------------------------------------------------------------------ #
    # Buat / buka file memmap
    # ------------------------------------------------------------------ #
    memmaps = {}
    for name, n_samp in window_specs:
        shape  = (n, 3, n_samp)
        mem_gb = n * 3 * n_samp * 4 / (1024**3)
        fname  = f"noise_waveforms_{name}.npy"
        fpath  = output_dir / fname

        if resume and fpath.exists():
            mode = "r+"
            print(f"Membuka memmap (resume): {fname}  shape={shape}  ({mem_gb:.2f} GB)")
        else:
            mode = "w+"
            print(f"Membuat memmap baru   : {fname}  shape={shape}  ({mem_gb:.2f} GB)")

        mm = np.memmap(fpath, dtype=np.float32, mode=mode, shape=shape)
        memmaps[name] = (mm, n_samp)

    # ------------------------------------------------------------------ #
    # Deteksi titik resume
    # ------------------------------------------------------------------ #
    start_idx = 0
    if resume:
        start_idx = _detect_resume_point(output_dir, window_specs, n)
        if start_idx > 0:
            print(f"\nMelanjutkan dari trace {start_idx:,} "
                  f"({start_idx / n * 100:.1f}% sudah selesai)")
        else:
            print("\nTidak ada progres sebelumnya, mulai dari awal")

    # ------------------------------------------------------------------ #
    # Baca HDF5 dan isi memmap
    # ------------------------------------------------------------------ #
    if start_idx >= n:
        print("Semua trace sudah dikonversi sebelumnya!")
    else:
        remaining = n - start_idx
        print(f"\nMengkonversi {remaining:,} trace dari {hdf5_path} ...")
        t0 = time.time()
        traces_written = 0

        with h5py.File(hdf5_path, "r") as f:
            for i in range(start_idx, n):
                # HDF5 STEAD menyimpan (6000, 3) → transpose ke (3, 6000)
                raw = np.array(f[f"data/{trace_names[i]}"],
                               dtype=np.float32).T   # → (3, 6000)

                for name, n_samp in window_specs:
                    memmaps[name][0][i] = extract_window_from_zero(raw, n_samp)

                traces_written += 1

                # Flush per chunk
                if traces_written % chunk_size == 0:
                    for name, (mm, _) in memmaps.items():
                        mm.flush()
                    elapsed = time.time() - t0
                    rate    = traces_written / elapsed
                    total_done = start_idx + traces_written
                    eta    = (n - total_done) / rate
                    pct    = total_done / n * 100
                    print(f"  {total_done:>10,}/{n:,} ({pct:5.1f}%)  "
                          f"{rate:.0f} trace/s  ETA {eta / 60:.1f}min  [flushed]")

                elif traces_written % 10_000 == 0:
                    elapsed = time.time() - t0
                    rate    = traces_written / elapsed
                    total_done = start_idx + traces_written
                    eta    = (n - total_done) / rate
                    pct    = total_done / n * 100
                    print(f"  {total_done:>10,}/{n:,} ({pct:5.1f}%)  "
                          f"{rate:.0f} trace/s  ETA {eta / 60:.1f}min")

        # Flush akhir
        for name, (mm, _) in memmaps.items():
            mm.flush()
            del mm

        elapsed = time.time() - t0
        print(f"\nSelesai: {traces_written:,} trace dalam {elapsed:.0f}s "
              f"({traces_written / elapsed:.0f} trace/s)")

    # ------------------------------------------------------------------ #
    # Simpan trace_names.npy
    # ------------------------------------------------------------------ #
    trace_name_path = output_dir / "noise_trace_names.npy"
    np.save(trace_name_path, trace_names)

    # ------------------------------------------------------------------ #
    # Simpan metadata.npz
    # ------------------------------------------------------------------ #
    def _safe_float(col: pd.Series) -> np.ndarray:
        return pd.to_numeric(col, errors="coerce").values.astype(np.float32)

    def _parse_snr(col: pd.Series) -> np.ndarray:
        result = np.full(len(col), np.nan, dtype=np.float32)
        for i, val in enumerate(col.values):
            try:
                result[i] = float(val)
            except (ValueError, TypeError):
                try:
                    cleaned = str(val).strip("[]")
                    nums    = [float(x) for x in cleaned.split()]
                    result[i] = np.mean(nums) if nums else np.nan
                except (ValueError, TypeError):
                    result[i] = np.nan
        return result

    save_kwargs = {}

    # Kolom wajib yang mungkin ada di noise CSV
    optional_cols = {
        "network_code"       : lambda c: df[c].values.astype(str),
        "receiver_code"      : lambda c: df[c].values.astype(str),
        "receiver_type"      : lambda c: df[c].values.astype(str),
        "receiver_latitude"  : lambda c: _safe_float(df[c]),
        "receiver_longitude" : lambda c: _safe_float(df[c]),
        "receiver_elevation_m": lambda c: _safe_float(df[c]),
        "snr_db"             : lambda c: _parse_snr(df[c]),
        "coda_end_sample"    : lambda c: _safe_float(df[c]),
        "trace_start_time"   : lambda c: df[c].values.astype(str),
    }

    for col, fn in optional_cols.items():
        if col in df.columns:
            save_kwargs[col] = fn(col)

    # Selalu simpan array indeks agar bisa dicocokkan dengan trace_names.npy
    save_kwargs["trace_index"] = np.arange(n, dtype=np.int32)

    meta_path = output_dir / "noise_metadata.npz"
    np.savez(meta_path, **save_kwargs)
    print(f"\nMetadata tersimpan: {meta_path}")

    # ------------------------------------------------------------------ #
    # Simpan index.json
    # ------------------------------------------------------------------ #
    index = {
        "dataset"       : "STEAD_noise",
        "n_traces"      : n,
        "window_start"  : "sample_0 (detik ke-0)",
        "sampling_rate" : SAMPLING_RATE,
        "hdf5_path"     : str(hdf5_path),
        "csv_path"      : str(csv_path),
        "variants"      : {},
    }
    for name, n_samp in window_specs:
        fname = f"noise_waveforms_{name}.npy"
        index["variants"][name] = {
            "file"     : fname,
            "shape"    : [n, 3, n_samp],
            "dtype"    : "float32",
            "window_s" : n_samp / SAMPLING_RATE,
            "size_gb"  : round(n * 3 * n_samp * 4 / (1024**3), 3),
        }

    index_path = output_dir / "noise_index.json"
    with open(index_path, "w") as fp:
        json.dump(index, fp, indent=2)

    # ------------------------------------------------------------------ #
    # Ringkasan akhir
    # ------------------------------------------------------------------ #
    print("\n=== File Output ===")
    for name, n_samp in window_specs:
        fname = f"noise_waveforms_{name}.npy"
        fpath = output_dir / fname
        size_gb = fpath.stat().st_size / (1024**3)
        print(f"  {fname:<30s}  shape=({n}, 3, {n_samp:>5})  {size_gb:.3f} GB")
    print(f"  {'noise_trace_names.npy':<30s}  {n:,} entries")
    print(f"  {'noise_metadata.npz':<30s}  network, SNR, koordinat, dll.")
    print(f"  {'noise_index.json':<30s}  metadata & shape info")
    print(f"\nOutput directory: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Konversi STEAD noise HDF5 → NumPy memmap (window dari detik ke-0)")

    parser.add_argument(
        "--hdf5",
        default="/home/indra/indra/STEAD/merged/merge.hdf5",
        help="Path ke file merge.hdf5")
    parser.add_argument(
        "--csv",
        default="/home/indra/indra/STEAD/merged/merge.csv",
        help="Path ke file merge.csv")
    parser.add_argument(
        "--output",
        default="/home/indra/indra/STEAD/memmap_noise",
        help="Direktori output untuk file memmap noise")
    parser.add_argument(
        "--windows", nargs="+", default=["3", "5", "10"],
        metavar="SEC",
        help="Durasi window dalam detik (default: 3 5 10). "
             "Contoh: --windows 3 5 10")
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Jumlah trace per siklus flush (default: {DEFAULT_CHUNK_SIZE:,}). "
             "Mengontrol puncak dirty-page memory.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Lanjutkan konversi yang terinterupsi (lewati trace yang sudah ditulis)")

    args = parser.parse_args()

    convert(
        hdf5_path  = args.hdf5,
        csv_path   = args.csv,
        output_dir = args.output,
        windows    = args.windows,
        chunk_size = args.chunk_size,
        resume     = args.resume,
    )


if __name__ == "__main__":
    main()
