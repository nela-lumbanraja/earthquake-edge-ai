"""
convert_pnw_sonic_to_memmap.py
=====================================
Mengonversi dataset PNW kelas 'sonic_boom' dari format HDF5 (SeisBench)
ke format memmap (.npy) dengan potongan 3s, 5s, dan 10s.

Struktur output:
    PNW/memmap_sonic/
    ├── index.json
    ├── metadata.npz
    ├── trace_names.npy
    ├── waveforms_3s.npy      # shape (N, 3, 300)
    ├── waveforms_5s.npy      # shape (N, 3, 500)
    └── waveforms_10s.npy     # shape (N, 3, 1000)

Konvensi pemotongan (P-centered, 100 Hz, pre_p = 1.0 s = 100 sample):
    - 3s  : 100 sample sebelum P → 200 sample setelah P  = 300 total
    - 5s  : 100 sample sebelum P → 400 sample setelah P  = 500 total
    - 10s : 100 sample sebelum P → 900 sample setelah P  = 1000 total
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────────────────────────────────────
SAMPLING_RATE     = 100
FULL_SAMPLES      = 18001
N_CHANNELS        = 3
WINDOW_SECS       = [3.0, 5.0, 10.0]
DEFAULT_P_ARRIVAL = 7000
DEFAULT_PRE_P_SEC = 1.0
DEFAULT_CHUNK_SIZE = 10_000

# Kandidat nilai source_type untuk sonic boom (case-insensitive)
SONIC_KEYWORDS = ["sonic_boom", "sonic boom", "sonicboom", "sb"]


# ─────────────────────────────────────────────────────────────────────────────
# Filter source_type (case-insensitive, strip whitespace)
# ─────────────────────────────────────────────────────────────────────────────
def _is_sonic(val) -> bool:
    if pd.isna(val):
        return False
    normalized = str(val).strip().lower().replace(" ", "_")
    return normalized in SONIC_KEYWORDS or str(val).strip().lower() in SONIC_KEYWORDS


def _filter_sonic(df: pd.DataFrame) -> pd.DataFrame:
    """Filter baris sonic boom dengan matching case-insensitive."""
    if "source_type" not in df.columns:
        raise KeyError(
            "Kolom 'source_type' tidak ditemukan di CSV.\n"
            "Pastikan menggunakan exotic_metadata.csv, bukan comcat_metadata.csv."
        )

    mask = df["source_type"].apply(_is_sonic)

    # Fallback: cek kolom source_type_pnsn_label jika source_type tidak cocok
    if mask.sum() == 0 and "source_type_pnsn_label" in df.columns:
        mask_pnsn = df["source_type_pnsn_label"].apply(
            lambda v: str(v).strip().lower() == "sb" if not pd.isna(v) else False
        )
        if mask_pnsn.sum() > 0:
            print(f"      ⚠  'source_type' tidak cocok; fallback ke "
                  f"source_type_pnsn_label == 'sb' → {mask_pnsn.sum():,} trace")
            mask = mask_pnsn

    return df[mask].copy().reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: parsing trace name
# ─────────────────────────────────────────────────────────────────────────────
def _parse_trace_name(trace_name: str):
    dollar_pos = trace_name.index("$")
    bucket_key = trace_name[:dollar_pos]
    rest       = trace_name[dollar_pos + 1:]
    row_idx    = int(rest.split(",")[0])
    return bucket_key, row_idx


def read_pnw_exotic_trace(f: h5py.File, trace_name: str) -> np.ndarray:
    bucket_key, row_idx = _parse_trace_name(trace_name)
    ds  = f[f"data/{bucket_key}"]
    raw = np.asarray(ds[row_idx], dtype=np.float32)

    if raw.shape == (N_CHANNELS, FULL_SAMPLES):
        return raw
    elif raw.shape == (FULL_SAMPLES, N_CHANNELS):
        return raw.T
    else:
        raise ValueError(
            f"Shape tidak dikenal {raw.shape} untuk trace '{trace_name}'. "
            f"Diharapkan ({N_CHANNELS}, {FULL_SAMPLES}) atau ({FULL_SAMPLES}, {N_CHANNELS})."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: windowing
# ─────────────────────────────────────────────────────────────────────────────
def extract_window(full_waveform: np.ndarray,
                   p_sample: int,
                   window_samples: int,
                   pre_p_samples: int) -> np.ndarray:
    start = p_sample - pre_p_samples
    end   = start + window_samples

    out       = np.zeros((N_CHANNELS, window_samples), dtype=np.float32)
    src_start = max(start, 0)
    src_end   = min(end, FULL_SAMPLES)
    dst_start = src_start - start
    dst_end   = dst_start + (src_end - src_start)

    if src_end > src_start:
        out[:, dst_start:dst_end] = full_waveform[:, src_start:src_end]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: parsing SNR
# ─────────────────────────────────────────────────────────────────────────────
def _parse_snr(col: pd.Series) -> np.ndarray:
    result = np.full(len(col), np.nan, dtype=np.float32)
    for i, val in enumerate(col.values):
        if pd.isna(val):
            continue
        try:
            result[i] = float(val)
        except (ValueError, TypeError):
            try:
                parts = [
                    float(x) for x in str(val).split("|")
                    if x.strip().lower() != "nan"
                ]
                result[i] = float(np.mean(parts)) if parts else np.nan
            except (ValueError, TypeError):
                pass
    return result


def _safe_float_col(col: pd.Series) -> np.ndarray:
    return pd.to_numeric(col, errors="coerce").values.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: resume detection
# ─────────────────────────────────────────────────────────────────────────────
def _detect_resume_point(output_dir: Path, window_specs: list, n: int) -> int:
    check_name, check_samples = None, 0
    for name, ns in window_specs:
        if name == "full":
            check_name, check_samples = name, ns
            break
        if ns > check_samples:
            check_name, check_samples = name, ns

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

    resume_from = (lo + 1) if np.any(mm[lo] != 0) else 0
    del mm
    return resume_from


# ─────────────────────────────────────────────────────────────────────────────
# Fungsi konversi utama
# ─────────────────────────────────────────────────────────────────────────────
def convert(hdf5_path: str,
            csv_path: str,
            output_dir: str,
            windows: list,
            pre_p_sec: float = DEFAULT_PRE_P_SEC,
            chunk_size: int = DEFAULT_CHUNK_SIZE,
            resume: bool = False):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pre_p_samp = int(pre_p_sec * SAMPLING_RATE)

    # ── 1. Load & filter metadata ──────────────────────────────────────────
    print(f"[1/5] Membaca metadata dari {csv_path} ...")
    df_all = pd.read_csv(csv_path, low_memory=False)

    # Tampilkan semua nilai unik source_type untuk diagnosis
    if "source_type" in df_all.columns:
        print("      Semua nilai 'source_type' yang ditemukan di CSV:")
        for stype, cnt in df_all["source_type"].value_counts(dropna=False).items():
            marker = " ◀ sonic (akan dipakai)" if _is_sonic(stype) else ""
            print(f"        {str(stype):30s}: {cnt:,}{marker}")
    else:
        print("      ⚠  Kolom 'source_type' tidak ditemukan!")

    df = _filter_sonic(df_all)
    print(f"\n      Trace sonic_boom terfilter: {len(df):,}")

    if len(df) == 0:
        # Tampilkan petunjuk debug lebih lengkap
        uniq = df_all.get("source_type", pd.Series(dtype=str)).dropna().unique().tolist()
        raise ValueError(
            f"Tidak ada trace sonic boom ditemukan.\n"
            f"Nilai 'source_type' yang ada di CSV: {uniq}\n"
            f"Keyword yang dicocokkan: {SONIC_KEYWORDS}\n"
            f"Coba tambahkan keyword yang sesuai ke SONIC_KEYWORDS di bagian atas skrip."
        )

    if "source_type_pnsn_label" in df.columns:
        print("      Distribusi source_type_pnsn_label:")
        for lbl, cnt in df["source_type_pnsn_label"].value_counts(dropna=False).items():
            print(f"        {str(lbl):20s}: {cnt:,}")

    # ── 2. Parse P-arrival ─────────────────────────────────────────────────
    if "trace_P_arrival_sample" in df.columns:
        p_arr_raw = pd.to_numeric(df["trace_P_arrival_sample"], errors="coerce")
        n_fallback = p_arr_raw.isna().sum()
        p_samples  = p_arr_raw.fillna(DEFAULT_P_ARRIVAL).astype(np.int32).values
        if n_fallback > 0:
            print(f"\n      ⚠  {n_fallback:,} trace menggunakan P-arrival fallback "
                  f"(sample {DEFAULT_P_ARRIVAL} = {DEFAULT_P_ARRIVAL / SAMPLING_RATE:.1f}s)")
    else:
        print(f"\n      ⚠  Kolom 'trace_P_arrival_sample' tidak ada. "
              f"Semua trace menggunakan P-arrival default = sample {DEFAULT_P_ARRIVAL}.")
        p_samples = np.full(len(df), DEFAULT_P_ARRIVAL, dtype=np.int32)

    trace_names = df["trace_name"].values
    n = len(trace_names)

    # ── 3. Parse window specs ──────────────────────────────────────────────
    window_specs = []
    for w in windows:
        if str(w) == "full":
            window_specs.append(("full", FULL_SAMPLES))
        else:
            sec  = float(w)
            samp = int(sec * SAMPLING_RATE)
            if samp > FULL_SAMPLES:
                raise ValueError(
                    f"Window {sec}s ({samp} samples) melebihi panjang trace "
                    f"PNW exotic ({FULL_SAMPLES} samples / {FULL_SAMPLES / SAMPLING_RATE:.2f}s)."
                )
            name = f"{int(sec)}s" if sec == int(sec) else f"{sec}s"
            window_specs.append((name, samp))

    # Pre-filter: P-arrival boundary check
    if any(name != "full" for name, _ in window_specs):
        max_post = max(
            ns - pre_p_samp for name, ns in window_specs if name != "full"
        )
        valid_mask = (
            (p_samples >= pre_p_samp) &
            (p_samples + max_post <= FULL_SAMPLES)
        )
        dropped = (~valid_mask).sum()
        if dropped > 0:
            print(f"\n      Trace dibuang (P-arrival keluar batas window): {dropped:,}")
            trace_names = trace_names[valid_mask]
            p_samples   = p_samples[valid_mask]
            df          = df[valid_mask].reset_index(drop=True)
            n = len(trace_names)
            print(f"      Trace valid setelah filter: {n:,}")

    if n == 0:
        raise ValueError(
            "Tidak ada trace valid setelah semua filter! "
            "Coba turunkan --pre-p atau periksa nilai P-arrival di CSV."
        )

    # ── 4. Alokasi memmap ──────────────────────────────────────────────────
    print(f"\n[2/5] Mengalokasi file memmap ...")
    total_gb = sum(n * N_CHANNELS * ns * 4 / (1024 ** 3) for _, ns in window_specs)
    print(f"      Total ukuran estimasi : {total_gb * 1024:.2f} MB  ({total_gb:.4f} GB)")

    memmaps = {}
    for name, ns in window_specs:
        shape  = (n, N_CHANNELS, ns)
        mem_mb = n * N_CHANNELS * ns * 4 / (1024 ** 2)
        fname  = f"waveforms_{name}.npy"
        fpath  = output_dir / fname
        mode   = "r+" if (resume and fpath.exists()) else "w+"
        action = "Membuka (resume)" if mode == "r+" else "Membuat baru  "
        print(f"      {action}: {fname}  shape={shape}  ({mem_mb:.2f} MB)")
        mm = np.memmap(fpath, dtype=np.float32, mode=mode, shape=shape)
        memmaps[name] = (mm, ns)

    # ── 5. Detect resume point ─────────────────────────────────────────────
    start_idx = 0
    if resume:
        start_idx = _detect_resume_point(output_dir, window_specs, n)
        if start_idx > 0:
            print(f"\n      Resume dari trace {start_idx:,} "
                  f"({start_idx / n * 100:.1f}% sudah selesai)")
        else:
            print("\n      Tidak ada progres sebelumnya, mulai dari awal.")

    # ── 6. Baca HDF5 & isi memmap ─────────────────────────────────────────
    if start_idx >= n:
        print("\nSemua trace sudah dikonversi sebelumnya!")
    else:
        remaining = n - start_idx
        print(f"\n[3/5] Membaca {remaining:,} trace dari {hdf5_path} ...")
        t0      = time.time()
        written = 0
        failed  = 0

        with h5py.File(hdf5_path, "r") as f:
            for i in range(start_idx, n):
                try:
                    raw = read_pnw_exotic_trace(f, trace_names[i])
                except Exception as e:
                    warnings.warn(f"Gagal baca trace '{trace_names[i]}': {e}")
                    failed  += 1
                    written += 1
                    continue

                for name, ns in window_specs:
                    mm, _ = memmaps[name]
                    if name == "full":
                        mm[i] = raw
                    else:
                        mm[i] = extract_window(raw, p_samples[i], ns, pre_p_samp)

                written += 1

                if written % chunk_size == 0:
                    for mm, _ in memmaps.values():
                        mm.flush()
                    elapsed    = time.time() - t0
                    rate       = written / max(elapsed, 1e-6)
                    total_done = start_idx + written
                    eta        = (n - total_done) / rate if rate > 0 else 0
                    print(f"  {total_done:>6,}/{n}  "
                          f"({total_done / n * 100:5.1f}%)  "
                          f"{rate:.0f} tr/s  ETA {eta:.1f}s  [flushed]")
                elif written % 20 == 0 or written == remaining:
                    elapsed    = time.time() - t0
                    rate       = written / max(elapsed, 1e-6)
                    total_done = start_idx + written
                    print(f"  {total_done:>6,}/{n}  "
                          f"({total_done / n * 100:5.1f}%)  "
                          f"{rate:.0f} tr/s")

        for mm, _ in memmaps.values():
            mm.flush()
            del mm

        elapsed = time.time() - t0
        print(f"\n      Selesai: {written:,} trace dalam {elapsed:.1f}s "
              f"({written / max(elapsed, 1e-6):.0f} tr/s)")
        if failed > 0:
            print(f"      ⚠  Gagal baca {failed:,} trace — slot dibiarkan nol di memmap.")

    # ── 7. Simpan trace_names.npy ─────────────────────────────────────────
    print("\n[4/5] Menyimpan trace_names.npy ...")
    np.save(output_dir / "trace_names.npy", trace_names.astype(str))

    # ── 8. Simpan metadata.npz & index.json ──────────────────────────────
    print("[5/5] Menyimpan metadata.npz dan index.json ...")

    snr_values = (
        _parse_snr(df["trace_snr_db"])
        if "trace_snr_db" in df.columns
        else np.full(n, np.nan, dtype=np.float32)
    )

    pnsn_labels = (
        df["source_type_pnsn_label"].values.astype(str)
        if "source_type_pnsn_label" in df.columns
        else np.array(["unknown"] * n)
    )

    # Ambil nilai source_type asli (sebelum normalisasi) dari baris pertama
    raw_source_type = str(df["source_type"].iloc[0]) if "source_type" in df.columns else "sonic_boom"

    np.savez(
        output_dir / "metadata.npz",
        p_arrival_sample       = p_samples,
        snr_db                 = snr_values,
        source_magnitude       = np.full(n, np.nan, dtype=np.float32),
        source_distance_km     = np.full(n, np.nan, dtype=np.float32),
        source_type            = np.array([raw_source_type] * n),
        source_type_pnsn_label = pnsn_labels,
        station_latitude       = (
            _safe_float_col(df["station_latitude_deg"])
            if "station_latitude_deg" in df.columns
            else np.full(n, np.nan, dtype=np.float32)
        ),
        station_longitude      = (
            _safe_float_col(df["station_longitude_deg"])
            if "station_longitude_deg" in df.columns
            else np.full(n, np.nan, dtype=np.float32)
        ),
    )

    index = {
        "n_traces"                 : n,
        "source_type"              : raw_source_type,
        "pre_p_sec"                : pre_p_sec,
        "sampling_rate"            : SAMPLING_RATE,
        "full_samples"             : FULL_SAMPLES,
        "default_p_arrival_sample" : DEFAULT_P_ARRIVAL,
        "hdf5_path"                : str(hdf5_path),
        "csv_path"                 : str(csv_path),
        "variants"                 : {},
        "class_imbalance_note"     : (
            f"Sonic boom memiliki trace sangat sedikit ({n}). "
            "Strategi: weighted loss, augmentasi (time-shift ±0.5s, amplitude ×[0.5,2]), "
            "oversampling, atau undersample kelas mayoritas."
        ),
        "notes": (
            "Source dari PNWExotic (exotic_waveforms.hdf5). "
            "source_magnitude dan source_distance_km adalah NaN."
        ),
    }

    for name, ns in window_specs:
        fname = f"waveforms_{name}.npy"
        index["variants"][name] = {
            "file"    : fname,
            "shape"   : [n, N_CHANNELS, ns],
            "dtype"   : "float32",
            "size_mb" : round(n * N_CHANNELS * ns * 4 / (1024 ** 2), 3),
        }

    with open(output_dir / "index.json", "w") as fp:
        json.dump(index, fp, indent=2)

    # ── 9. Ringkasan akhir ────────────────────────────────────────────────
    print("\n✅  Selesai! Ringkasan output:")
    print(f"   Folder              : {output_dir.resolve()}")
    print(f"   Source type         : {raw_source_type}")
    print(f"   Jumlah trace valid  : {n:,}")
    for name, ns in window_specs:
        fpath   = output_dir / f"waveforms_{name}.npy"
        size_mb = fpath.stat().st_size / (1024 ** 2) if fpath.exists() else 0
        print(f"   waveforms_{name}.npy  : shape ({n}, {N_CHANNELS}, {ns:>6})  "
              f"[{size_mb:.2f} MB]")
    print(f"   trace_names.npy     : shape ({n},)")
    print(f"   metadata.npz        : p_arrival_sample, snr_db, source_magnitude (NaN),")
    print(f"                         source_distance_km (NaN), source_type,")
    print(f"                         source_type_pnsn_label, station_latitude, station_longitude")
    print(f"   index.json          : {n:,} entri")
    print(f"\n[Class imbalance reminder]")
    print(f"   Sonic boom traces   : {n}")
    print(f"     1. Weighted loss   — weight = N_total / (n_classes × {n})")
    print(f"     2. Augmentasi      — time-shift ±0.5s, amplitude ×[0.5,2], add noise")
    print(f"     3. Oversampling    — repeat sonic traces hingga balanced")
    print(f"     4. Undersampling   — cap kelas besar ke 5–10× sonic count")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Konversi PNW exotic sonic boom class → NumPy memmap files "
            "(3s/5s/10s, P-centered, 100 Hz)"
        )
    )
    parser.add_argument(
        "--hdf5",
        default="/home/indra/indra/PNW/exotic_waveforms.hdf5",
        help="Path ke PNW exotic HDF5 file"
    )
    parser.add_argument(
        "--csv",
        default="/home/indra/indra/PNW/exotic_metadata.csv",
        help="Path ke exotic_metadata.csv"
    )
    parser.add_argument(
        "--output",
        default="/home/indra/indra/PNW/memmap_sonic",
        help="Direktori output"
    )
    parser.add_argument(
        "--windows", nargs="+", default=["3", "5", "10"],
        metavar="SEC",
        help="Durasi window dalam detik atau 'full'. Default: 3 5 10"
    )
    parser.add_argument(
        "--pre-p", type=float, default=DEFAULT_PRE_P_SEC,
        help=f"Detik sebelum P-arrival (default: {DEFAULT_PRE_P_SEC}s)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Jumlah trace per flush (default: {DEFAULT_CHUNK_SIZE:,})"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Lanjutkan konversi yang terputus"
    )

    args = parser.parse_args()

    convert(
        hdf5_path  = args.hdf5,
        csv_path   = args.csv,
        output_dir = args.output,
        windows    = args.windows,
        pre_p_sec  = args.pre_p,
        chunk_size = args.chunk_size,
        resume     = args.resume,
    )


if __name__ == "__main__":
    main()