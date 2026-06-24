
"""
convert_pnw_explosion_to_memmap.py
=====================================
Mengonversi dataset PNW kelas 'explosion' dari format HDF5 (SeisBench)
ke format memmap (.npy) dengan potongan 3s, 5s, dan 10s.

Struktur output mengikuti konvensi earthquake memmap:
    PNW/memmap_explosion/
    ├── index.json            # metadata lengkap (shape, size_gb, paths, dll)
    ├── metadata.npz          # back_azimuth_deg, p_arrival_sample,
    │                         #   source_magnitude, source_distance_km, snr_db
    ├── trace_names.npy       # array 1D nama trace
    ├── waveforms_3s.npy      # shape (N, 3, 300)  — float32
    ├── waveforms_5s.npy      # shape (N, 3, 500)  — float32
    └── waveforms_10s.npy     # shape (N, 3, 1000) — float32

Konvensi pemotongan (P-centered, 100 Hz, pre_p = 1.0 s = 100 sample):
    - 3s  : 100 sample sebelum P → 200 sample setelah P  = 300 total
    - 5s  : 100 sample sebelum P → 400 sample setelah P  = 500 total
    - 10s : 100 sample sebelum P → 900 sample setelah P  = 1000 total

Catatan PNW explosion vs earthquake:
    - source_type filter: 'explosion' (bukan 'earthquake')
    - Kolom back_azimuth_deg & source_distance_km tidak ada di CSV,
      dihitung dari koordinat source & station (haversine).
    - SNR disimpan sebagai string pipe-separated "E|N|Z" per trace,
      dikonversi ke mean scalar float32.
    - Label 'px' pada source_type_pnsn_label adalah label PNSN untuk explosion.

Jalankan:
    python convert_pnw_explosion_to_memmap.py \
        --hdf5  /path/to/PNW/comcat_waveforms.hdf5 \
        --csv   /path/to/PNW/comcat_metadata.csv \
        --out   /path/to/PNW/memmap_explosion \
        [--pre-p 1.0] [--chunk-size 1000] [--resume]
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────────────────────────────────────
SAMPLING_RATE   = 100    # Hz — semua trace PNW comcat
N_SAMPLES_TRACE = 15001  # fixed untuk comcat PNW
N_CHANNELS      = 3      # E, N, Z

WINDOW_SECS = [3.0, 5.0, 10.0]          # detik
DEFAULT_PRE_P_SEC  = 1.0                 # 1s pre-P (= 100 sample)
DEFAULT_CHUNK_SIZE = 1_000              # traces per flush

# Label source_type yang dianggap 'explosion' di PNW comcat
# Berdasarkan CSV: source_type == 'explosion'
# source_type_pnsn_label == 'px' (quarry blast / explosion label PNSN)
EXPLOSION_SOURCE_TYPE = "explosion"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers geometri — hitung BAZ & jarak dari koordinat
# ─────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Jarak permukaan bumi (km) antara dua titik (derajat)."""
    R = 6371.0
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    Δφ = np.radians(lat2 - lat1)
    Δλ = np.radians(lon2 - lon1)
    a = np.sin(Δφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(Δλ / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def back_azimuth_deg(src_lat, src_lon, sta_lat, sta_lon):
    """
    Back azimuth: arah dari station ke source (0–360°).
    Dihitung dengan formula azimuth balik (geodesi sferis).
    """
    φs  = np.radians(src_lat);  φr  = np.radians(sta_lat)
    Δλ  = np.radians(src_lon - sta_lon)
    x   = np.sin(Δλ) * np.cos(φs)
    y   = np.cos(φr) * np.sin(φs) - np.sin(φr) * np.cos(φs) * np.cos(Δλ)
    baz = (np.degrees(np.arctan2(x, y)) + 360) % 360
    return baz


# ─────────────────────────────────────────────────────────────────────────────
# Helpers HDF5
# ─────────────────────────────────────────────────────────────────────────────
def parse_trace_name(trace_name: str):
    """
    Format PNW: "bucket4$0,:3,:15001"
    Return (bucket_name, row_idx).
    """
    bucket_part, idx_part = trace_name.split("$")
    row_idx = int(idx_part.split(",")[0])
    return bucket_part, row_idx


def load_waveform(hdf5_file: h5py.File, trace_name: str):
    """
    Load satu waveform dari HDF5.
    Return array shape (3, n_samples) float32, atau None jika gagal.
    """
    bucket_name, row_idx = parse_trace_name(trace_name)
    try:
        wave = hdf5_file["data"][bucket_name][row_idx]   # (3, n_samples)
        return wave.astype(np.float32)
    except Exception as e:
        warnings.warn(f"Gagal load '{trace_name}': {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers windowing & SNR
# ─────────────────────────────────────────────────────────────────────────────
def extract_window(wave: np.ndarray, p_sample: int,
                   window_samples: int, pre_p_samples: int) -> np.ndarray:
    """
    Potong window P-centered; zero-pad jika keluar batas.
    Identik dengan referensi earthquake agar kompatibel saat training.
    """
    start = p_sample - pre_p_samples
    end   = start + window_samples

    out = np.zeros((N_CHANNELS, window_samples), dtype=np.float32)
    src_start = max(start, 0)
    src_end   = min(end, wave.shape[-1])
    dst_start = src_start - start
    dst_end   = dst_start + (src_end - src_start)
    if src_end > src_start:
        out[:, dst_start:dst_end] = wave[:, src_start:src_end]
    return out


def parse_snr_pnw(val) -> float:
    """
    PNW SNR disimpan sebagai "E_snr|N_snr|Z_snr" (pipe-separated).
    Kembalikan mean scalar float32; NaN jika gagal parse.
    """
    try:
        parts = [float(x) for x in str(val).split("|")]
        return float(np.mean(parts)) if parts else np.nan
    except (ValueError, TypeError):
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Resume detection
# ─────────────────────────────────────────────────────────────────────────────
def _detect_resume_point(output_dir: Path, window_specs: list, n: int) -> int:
    """
    Cari jumlah trace yang sudah ditulis dengan binary search di memmap.
    Mengecek memmap terbesar.
    """
    check_name, check_samples = None, 0
    for name, n_samp in window_specs:
        if n_samp > check_samples:
            check_name, check_samples = name, n_samp

    if check_name is None:
        return 0

    fpath = output_dir / f"waveforms_{check_name}.npy"
    if not fpath.exists():
        return 0

    mm = np.memmap(fpath, dtype=np.float32, mode="r",
                   shape=(n, N_CHANNELS, check_samples))

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


# ─────────────────────────────────────────────────────────────────────────────
# Main convert
# ─────────────────────────────────────────────────────────────────────────────
def convert(hdf5_path: str, csv_path: str, output_dir: str,
            pre_p_sec: float = DEFAULT_PRE_P_SEC,
            chunk_size: int = DEFAULT_CHUNK_SIZE,
            resume: bool = False):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_p_samp = int(pre_p_sec * SAMPLING_RATE)

    # ── 1. Load & filter metadata ──────────────────────────────────────────
    print(f"[1/5] Membaca metadata dari {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)

    # Tampilkan distribusi source_type untuk verifikasi
    print(f"      Distribusi source_type di CSV:")
    for stype, cnt in df["source_type"].value_counts().items():
        print(f"        {stype:20s}: {cnt:,}")

    # Filter: HANYA kelas explosion
    df = df[df["source_type"] == EXPLOSION_SOURCE_TYPE].copy().reset_index(drop=True)
    print(f"\n      Trace explosion: {len(df):,}")

    if len(df) == 0:
        sys.exit(
            f"❌  Tidak ada trace dengan source_type='{EXPLOSION_SOURCE_TYPE}' di CSV!\n"
            f"   Periksa kembali nama kolom atau nilai filter."
        )

    # Opsional: tampilkan distribusi label PNSN untuk verifikasi
    if "source_type_pnsn_label" in df.columns:
        print(f"      Distribusi source_type_pnsn_label (explosion):")
        for lbl, cnt in df["source_type_pnsn_label"].value_counts().items():
            print(f"        {str(lbl):20s}: {cnt:,}")

    # Filter: wajib punya P-arrival sample
    before = len(df)
    df = df.dropna(subset=["trace_P_arrival_sample"]).reset_index(drop=True)
    df["trace_P_arrival_sample"] = df["trace_P_arrival_sample"].astype(int)
    print(f"\n      Trace dengan P-pick valid: {len(df):,}  (dibuang: {before - len(df):,})")

    # ── 2. Build window specs ──────────────────────────────────────────────
    window_specs = []
    for sec in WINDOW_SECS:
        n_samp = int(sec * SAMPLING_RATE)
        name   = f"{int(sec)}s" if sec == int(sec) else f"{sec}s"
        window_specs.append((name, n_samp))

    # Pre-filter: semua window harus muat dalam trace
    max_post = max(n_samp - pre_p_samp for _, n_samp in window_specs)
    valid_mask = (
        (df["trace_P_arrival_sample"] >= pre_p_samp) &
        (df["trace_P_arrival_sample"] + max_post <= N_SAMPLES_TRACE)
    )
    dropped = (~valid_mask).sum()
    df = df[valid_mask].reset_index(drop=True)
    print(f"      Trace lolos cek batas window: {len(df):,}  (dibuang: {dropped:,})")

    N = len(df)
    if N == 0:
        sys.exit("❌  Tidak ada trace valid setelah semua filter!")

    # ── 3. Hitung BAZ & distance ───────────────────────────────────────────
    print("\n[2/5] Menghitung back_azimuth_deg & source_distance_km ...")
    baz_arr = back_azimuth_deg(
        df["source_latitude_deg"].values,
        df["source_longitude_deg"].values,
        df["station_latitude_deg"].values,
        df["station_longitude_deg"].values,
    ).astype(np.float32)

    dist_arr = haversine_km(
        df["source_latitude_deg"].values,
        df["source_longitude_deg"].values,
        df["station_latitude_deg"].values,
        df["station_longitude_deg"].values,
    ).astype(np.float32)

    # ── 4. Alokasi / buka memmap ──────────────────────────────────────────
    print("\n[3/5] Mengalokasi file memmap ...")
    memmaps = {}
    total_gb = 0.0
    for name, n_samp in window_specs:
        shape   = (N, N_CHANNELS, n_samp)
        mem_gb  = N * N_CHANNELS * n_samp * 4 / (1024 ** 3)
        total_gb += mem_gb
        fpath   = output_dir / f"waveforms_{name}.npy"
        mode    = "r+" if (resume and fpath.exists()) else "w+"
        mm      = np.memmap(fpath, dtype=np.float32, mode=mode, shape=shape)
        memmaps[name] = (mm, n_samp)
        print(f"      waveforms_{name}.npy  shape={shape}  ({mem_gb:.3f} GB)  "
              f"[{'resume' if mode == 'r+' else 'baru'}]")
    print(f"      Total estimasi ukuran: {total_gb:.3f} GB")

    # ── 5. Detect resume point ─────────────────────────────────────────────
    start_idx = 0
    if resume:
        start_idx = _detect_resume_point(output_dir, window_specs, N)
        if start_idx > 0:
            print(f"\n      Resume dari trace {start_idx:,} "
                  f"({start_idx / N * 100:.1f}% sudah selesai)")
        else:
            print("\n      Tidak ada progress sebelumnya, mulai dari awal.")

    # ── 6. Baca HDF5 & isi memmap ─────────────────────────────────────────
    if start_idx >= N:
        print("\nSemua trace sudah dikonversi sebelumnya!")
    else:
        print(f"\n[4/5] Membaca waveform & mengisi memmap "
              f"(mulai dari {start_idx:,}) ...")
        t0 = time.time()
        traces_written = 0
        traces_failed  = 0

        with h5py.File(hdf5_path, "r") as hf:
            for i in tqdm(range(start_idx, N), desc="Converting",
                          unit="trace", ncols=90):
                row        = df.iloc[i]
                trace_name = row["trace_name"]
                p_samp     = int(row["trace_P_arrival_sample"])

                wave = load_waveform(hf, trace_name)
                if wave is None:
                    # Biarkan slot nol; dicatat sebagai failed
                    traces_failed += 1
                    traces_written += 1
                    continue

                for name, n_samp in window_specs:
                    memmaps[name][0][i] = extract_window(
                        wave, p_samp, n_samp, pre_p_samp
                    )

                traces_written += 1

                # Flush per chunk
                if traces_written % chunk_size == 0:
                    for name, (mm, _) in memmaps.items():
                        mm.flush()

        # Final flush
        for name, (mm, _) in memmaps.items():
            mm.flush()
            del mm

        elapsed = time.time() - t0
        print(f"\n      Selesai : {traces_written:,} trace dalam {elapsed:.0f}s "
              f"({traces_written / max(elapsed, 1):.0f} traces/s)")
        if traces_failed > 0:
            print(f"      ⚠  Gagal load waveform: {traces_failed:,} trace "
                  f"(slot dibiarkan nol di memmap)")

    # ── 7. Simpan trace_names & metadata ─────────────────────────────────
    print("\n[5/5] Menyimpan trace_names, metadata, dan index.json ...")

    trace_names = df["trace_name"].values.astype(str)
    np.save(output_dir / "trace_names.npy", trace_names)

    # Parse SNR PNW: "E|N|Z" → mean scalar
    snr_values = np.array(
        [parse_snr_pnw(v) for v in df["trace_snr_db"].values],
        dtype=np.float32
    )

    # P-arrival samples
    p_arrival = df["trace_P_arrival_sample"].values.astype(np.int32)

    def _safe_float(col: pd.Series) -> np.ndarray:
        return pd.to_numeric(col, errors="coerce").values.astype(np.float32)

    # Simpan metadata.npz — struktur identik dengan earthquake memmap
    np.savez(
        output_dir / "metadata.npz",
        back_azimuth_deg     = baz_arr,
        p_arrival_sample     = p_arrival,
        source_magnitude     = _safe_float(df["preferred_source_magnitude"]),
        source_distance_km   = dist_arr,
        snr_db               = snr_values,
        # Kolom tambahan PNW berguna saat training / analisis
        source_depth_km      = _safe_float(df["source_depth_km"]),
        source_latitude_deg  = df["source_latitude_deg"].values.astype(np.float32),
        source_longitude_deg = df["source_longitude_deg"].values.astype(np.float32),
        station_latitude_deg = df["station_latitude_deg"].values.astype(np.float32),
        station_longitude_deg= df["station_longitude_deg"].values.astype(np.float32),
        trace_P_arrival_sample = p_arrival,   # alias eksplisit
        trace_S_arrival_sample = (
            df["trace_S_arrival_sample"].values.astype(np.float32)
            if "trace_S_arrival_sample" in df.columns
            else np.full(N, np.nan, dtype=np.float32)
        ),
        # Kolom khusus explosion — berguna untuk analisis tipe sumber
        source_type_pnsn_label = (
            df["source_type_pnsn_label"].values.astype(str)
            if "source_type_pnsn_label" in df.columns
            else np.array(["unknown"] * N)
        ),
    )

    # ── 8. Simpan index.json ───────────────────────────────────────────────
    index = {
        "n_traces"      : N,
        "pre_p_sec"     : pre_p_sec,
        "sampling_rate" : SAMPLING_RATE,
        "source"        : "PNW comcat (explosion)",
        "hdf5_path"     : str(hdf5_path),
        "csv_path"      : str(csv_path),
        "variants"      : {},
    }
    for name, n_samp in window_specs:
        fname = f"waveforms_{name}.npy"
        fpath = output_dir / fname
        size_gb = round(fpath.stat().st_size / (1024 ** 3), 4) if fpath.exists() else 0
        index["variants"][name] = {
            "file"    : fname,
            "shape"   : [N, N_CHANNELS, n_samp],
            "dtype"   : "float32",
            "size_gb" : size_gb,
        }

    with open(output_dir / "index.json", "w") as fp:
        json.dump(index, fp, indent=2)

    # ── 9. Ringkasan ────────────────────────────────────────────────────────
    print("\n✅  Selesai! Ringkasan output:")
    print(f"   Folder         : {output_dir}")
    print(f"   Jumlah trace   : {N:,}")
    for name, n_samp in window_specs:
        fpath = output_dir / f"waveforms_{name}.npy"
        size_gb = fpath.stat().st_size / (1024 ** 3) if fpath.exists() else 0
        print(f"   waveforms_{name}.npy : shape ({N}, {N_CHANNELS}, {n_samp})  [{size_gb:.4f} GB]")
    print(f"   trace_names.npy    : shape ({N},)")
    print(f"   metadata.npz       : back_azimuth_deg, p_arrival_sample,")
    print(f"                        source_magnitude, source_distance_km,")
    print(f"                        snr_db, source_depth_km, lat/lon,")
    print(f"                        S_arrival, source_type_pnsn_label")
    print(f"   index.json         : {N:,} entri")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Konversi PNW explosion dataset → memmap format (3s/5s/10s)"
    )
    parser.add_argument(
    "--hdf5",
    default="/home/indra/indra/PNW/comcat_waveforms.hdf5",
        help="Path ke comcat_waveforms.hdf5"
    )
    parser.add_argument(
    "--csv",
    default="/home/indra/indra/PNW/comcat_metadata.csv",
        help="Path ke comcat_metadata.csv"
    )
    parser.add_argument(
    "--out",
    default="/home/indra/indra/PNW/memmap_explosion",
    help="Folder output otomatis ke /home/indra/indra/PNW/memmap_explosion"
    )
    parser.add_argument(
        "--pre-p", type=float, default=DEFAULT_PRE_P_SEC,
        help=f"Detik sebelum P-arrival yang disertakan dalam window "
             f"(default: {DEFAULT_PRE_P_SEC}s = {int(DEFAULT_PRE_P_SEC * SAMPLING_RATE)} sample)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Jumlah trace per flush ke disk (default: {DEFAULT_CHUNK_SIZE:,})"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Lanjutkan konversi yang terputus (skip trace yang sudah ditulis)"
    )
    args = parser.parse_args()

    convert(
        hdf5_path  = args.hdf5,
        csv_path   = args.csv,
        output_dir = args.out,
        pre_p_sec  = args.pre_p,
        chunk_size = args.chunk_size,
        resume     = args.resume,
    )