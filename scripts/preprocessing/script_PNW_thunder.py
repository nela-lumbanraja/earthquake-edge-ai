#!/usr/bin/env python
"""Convert PNW exotic dataset (thunder class) to NumPy memmap files.

Adapted from convert_to_memmap.py (STEAD HDF5 → memmap).
PNW differences vs STEAD:
  - Source: seisbench-style HDF5 with datasets keyed by bucket + index
  - Trace shape stored as (18001, 3) at 100 Hz  →  transposed to (3, 18001)
  - P arrival fixed at sample 7000 (70 s) for all thunder traces
  - No back_azimuth_deg column → omitted from metadata.npz
  - SNR stored as pipe-separated string per channel (e.g. "23.9|19.7|28.4")
  - source_type filter: only 'thunder' rows are processed
tmux
Output structure mirrors STEAD memmap layout under PNW/memmap_thunder/:
  waveforms_3s.npy    shape (N, 3, 300)
  waveforms_5s.npy    shape (N, 3, 500)
  waveforms_10s.npy   shape (N, 3, 1000)
  trace_names.npy     string array of trace identifiers
  metadata.npz        p_arrival_sample, snr_db, source_magnitude (NaN), source_distance_km (NaN)
  index.json          variant descriptions

Window extraction is P-centred with a configurable pre-P buffer (default 1 s).
Because thunder has no reliable S arrival, only P-centred windows are produced
(no 'full' variant by default — the 180 s trace is large and mostly noise).

Usage:
    # Default: 3s, 5s, 10s windows
    python convert_pnw_thunder_to_memmap.py

    # Custom windows (seconds) and output directory
    python convert_pnw_thunder_to_memmap.py --windows 3 5 10 --output /data/PNW/memmap_thunder

    # Include full trace
    python convert_pnw_thunder_to_memmap.py --windows full 3 5 10

    # Resume after interruption
    python convert_pnw_thunder_to_memmap.py --resume

Class imbalance note
--------------------
Thunder has only ~146 traces vs thousands of earthquake/noise traces.
Recommended strategies when training:
  1. Weighted loss  (simplest): weight_thunder = N_total / (n_classes * N_thunder)
  2. Oversampling   : repeat thunder traces or use augmentation (time-shift, amplitude scale, add noise)
  3. Undersampling  : cap earthquake traces to ~5–10× thunder count
  4. Augmentation   : random time-shift ±0.5 s, amplitude scale ×[0.5,2], channel shuffle
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SAMPLING_RATE = 100          # Hz
FULL_SAMPLES  = 18001        # 180.01 s — full PNW exotic trace length
DEFAULT_P_ARRIVAL  = 7000    # sample 7000 = 70 s from trace start
DEFAULT_PRE_P_SEC  = 1.0     # 1 s pre-P buffer
DEFAULT_CHUNK_SIZE = 10_000  # smaller than STEAD (only 146 thunder traces anyway)
SOURCE_TYPE_FILTER = "thunder"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_snr(col: pd.Series) -> np.ndarray:
    """Parse PNW pipe-separated snr_db → mean scalar SNR per trace.

    Values look like "23.9|19.7|28.4" or "nan|nan|18.0" or plain NaN.
    """
    result = np.full(len(col), np.nan, dtype=np.float32)
    for i, val in enumerate(col.values):
        if pd.isna(val):
            continue
        try:
            result[i] = float(val)
        except (ValueError, TypeError):
            try:
                parts = [float(x) for x in str(val).split("|") if x.strip().lower() != "nan"]
                result[i] = float(np.mean(parts)) if parts else np.nan
            except (ValueError, TypeError):
                pass
    return result


def _safe_float_col(col: pd.Series) -> np.ndarray:
    return pd.to_numeric(col, errors="coerce").values.astype(np.float32)


def extract_window(full_waveform: np.ndarray, p_sample: int,
                   window_samples: int, pre_p_samples: int) -> np.ndarray:
    """Extract P-centred window from full (3, FULL_SAMPLES) waveform.

    Pads with zeros if the window exceeds the trace boundaries.
    """
    start = p_sample - pre_p_samples
    end   = start + window_samples

    out = np.zeros((3, window_samples), dtype=np.float32)
    src_start = max(start, 0)
    src_end   = min(end, FULL_SAMPLES)
    dst_start = src_start - start
    dst_end   = dst_start + (src_end - src_start)

    if src_end > src_start:
        out[:, dst_start:dst_end] = full_waveform[:, src_start:src_end]
    return out


def _detect_resume_point(output_dir: Path, window_specs: list, n: int) -> int:
    """Binary-search the last non-zero row to find resume position."""
    # Prefer largest window for the check
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

    mm = np.memmap(fpath, dtype=np.float32, mode="r", shape=(n, 3, check_samples))

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


# ---------------------------------------------------------------------------
# PNW-specific: read a single trace from bucket-based HDF5
# ---------------------------------------------------------------------------

def _parse_trace_name(trace_name: str):
    """Parse 'bucketN$idx,:3,:18001' → (bucket_key, row_index).

    PNW exotic HDF5 stores data in datasets like 'data/bucket3'
    shaped (M, 3, 18001).  The trace_name encodes bucket + row.
    """
    # Format: "bucket3$0,:3,:18001"
    dollar_pos = trace_name.index("$")
    bucket_key = trace_name[:dollar_pos]          # e.g. "bucket3"
    rest        = trace_name[dollar_pos + 1:]     # e.g. "0,:3,:18001"
    row_idx     = int(rest.split(",")[0])          # e.g. 0
    return bucket_key, row_idx


def read_pnw_trace(f: h5py.File, trace_name: str) -> np.ndarray:
    """Read one trace from PNW HDF5.  Returns (3, FULL_SAMPLES) float32."""
    bucket_key, row_idx = _parse_trace_name(trace_name)
    # Dataset path: data/bucket3  shape (M, 3, 18001)
    ds   = f[f"data/{bucket_key}"]
    raw  = ds[row_idx]                            # shape (3, 18001) or (18001, 3)
    raw  = np.asarray(raw, dtype=np.float32)
    # Normalise to (3, FULL_SAMPLES)
    if raw.shape == (3, FULL_SAMPLES):
        return raw
    elif raw.shape == (FULL_SAMPLES, 3):
        return raw.T
    else:
        raise ValueError(f"Unexpected trace shape {raw.shape} for {trace_name}")


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(hdf5_path: str, csv_path: str, output_dir: str,
            windows: list[str], pre_p_sec: float = DEFAULT_PRE_P_SEC,
            chunk_size: int = DEFAULT_CHUNK_SIZE, resume: bool = False):
    """Filter thunder traces from PNW exotic CSV/HDF5 and write memmap files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load and filter metadata ---
    print(f"Loading metadata from {csv_path} ...")
    df_all = pd.read_csv(csv_path, low_memory=False)
    df = df_all[df_all["source_type"] == SOURCE_TYPE_FILTER].reset_index(drop=True)
    print(f"  Total rows : {len(df_all):,}")
    print(f"  Thunder rows: {len(df):,}")

    if len(df) == 0:
        raise ValueError(f"No rows with source_type='{SOURCE_TYPE_FILTER}' found.")

    trace_names = df["trace_name"].values
    n = len(trace_names)

    p_samples = (
    pd.to_numeric(
        df["trace_P_arrival_sample"],
        errors="coerce"
    )
    .fillna(DEFAULT_P_ARRIVAL)
    .astype(np.int32)
    .values
)

    pre_p_samp = int(pre_p_sec * SAMPLING_RATE)

    # --- Parse window specs ---
    window_specs: list[tuple[str, int]] = []
    for w in windows:
        if w == "full":
            window_specs.append(("full", FULL_SAMPLES))
        else:
            sec  = float(w)
            samp = int(sec * SAMPLING_RATE)
            name = f"{int(sec)}s" if sec == int(sec) else f"{sec}s"
            window_specs.append((name, samp))

    # --- Memory estimates ---
    total_gb = sum(n * 3 * ns * 4 / (1024**3) for _, ns in window_specs)
    chunk_gb = sum(min(chunk_size, n) * 3 * ns * 4 / (1024**3) for _, ns in window_specs)
    print(f"\nTotal memmap size : {total_gb*1024:.1f} MB  ({total_gb:.4f} GB)")
    print(f"Chunk size        : {min(chunk_size,n):,} traces  (~{chunk_gb*1024:.1f} MB dirty pages)")

    # --- Create / open memmap files ---
    memmaps: dict[str, tuple[np.memmap, int]] = {}
    for name, ns in window_specs:
        shape   = (n, 3, ns)
        mem_mb  = n * 3 * ns * 4 / (1024**2)
        fname   = f"waveforms_{name}.npy"
        fpath   = output_dir / fname
        mode    = "r+" if (resume and fpath.exists()) else "w+"
        action  = "Opening (resume)" if mode == "r+" else "Creating"
        print(f"  {action}: {fname}  shape={shape}  ({mem_mb:.1f} MB)")
        mm = np.memmap(fpath, dtype=np.float32, mode=mode, shape=shape)
        memmaps[name] = (mm, ns)

    # --- Resume detection ---
    start_idx = 0
    if resume:
        start_idx = _detect_resume_point(output_dir, window_specs, n)
        if start_idx > 0:
            print(f"\nResuming from trace {start_idx:,} ({start_idx/n*100:.1f}% done)")
        else:
            print("\nNo prior progress found – starting from scratch")

    # --- Convert ---
    if start_idx >= n:
        print("All traces already converted!")
    else:
        remaining = n - start_idx
        print(f"\nConverting {remaining:,} thunder traces from {hdf5_path} ...")
        t0 = time.time()
        written = 0

        with h5py.File(hdf5_path, "r") as f:
            for i in range(start_idx, n):
                raw = read_pnw_trace(f, trace_names[i])   # (3, 18001) float32

                for name, ns in window_specs:
                    mm, _ = memmaps[name]
                    if name == "full":
                        mm[i] = raw
                    else:
                        mm[i] = extract_window(raw, p_samples[i], ns, pre_p_samp)

                written += 1

                if written % chunk_size == 0:
                    for name, (mm, _) in memmaps.items():
                        mm.flush()
                    elapsed = time.time() - t0
                    rate    = written / elapsed
                    done    = start_idx + written
                    eta     = (n - done) / rate if rate > 0 else 0
                    print(f"  {done:>6,}/{n}  {rate:.0f} tr/s  ETA {eta:.1f}s  [flushed]")
                elif written % 20 == 0 or written == remaining:
                    elapsed = time.time() - t0
                    rate    = written / elapsed
                    done    = start_idx + written
                    pct     = done / n * 100
                    print(f"  {done:>6,}/{n}  ({pct:5.1f}%)  {rate:.0f} tr/s")

        # Final flush
        for name, (mm, _) in memmaps.items():
            mm.flush()
            del mm

        elapsed = time.time() - t0
        print(f"\nDone: {written:,} traces in {elapsed:.1f}s  ({written/elapsed:.0f} tr/s)")

    # --- Save auxiliary files ---

    # trace_names.npy
    np.save(output_dir / "trace_names.npy", trace_names)

    # metadata.npz  (mirrors STEAD layout; BAZ not available in PNW exotic)
    snr_values = (_parse_snr(df["trace_snr_db"])
                  if "trace_snr_db" in df.columns else np.full(n, np.nan, dtype=np.float32))

    np.savez(
        output_dir / "metadata.npz",
        p_arrival_sample  = p_samples,
        snr_db            = snr_values,
        source_magnitude  = np.full(n, np.nan, dtype=np.float32),   # not in PNW exotic
        source_distance_km= np.full(n, np.nan, dtype=np.float32),   # not in PNW exotic
        source_type       = np.array([SOURCE_TYPE_FILTER] * n),     # all "thunder"
        station_latitude  = _safe_float_col(df["station_latitude_deg"])
                            if "station_latitude_deg" in df.columns
                            else np.full(n, np.nan, dtype=np.float32),
        station_longitude = _safe_float_col(df["station_longitude_deg"])
                            if "station_longitude_deg" in df.columns
                            else np.full(n, np.nan, dtype=np.float32),
    )

    # index.json
    index = {
        "n_traces"      : n,
        "source_type"   : SOURCE_TYPE_FILTER,
        "pre_p_sec"     : pre_p_sec,
        "sampling_rate" : SAMPLING_RATE,
        "full_samples"  : FULL_SAMPLES,
        "default_p_arrival_sample": DEFAULT_P_ARRIVAL,
        "hdf5_path"     : str(hdf5_path),
        "csv_path"      : str(csv_path),
        "variants"      : {},
        "class_imbalance_note": (
            "Thunder has very few traces compared to earthquake/noise. "
            "Recommended mitigations: (1) weighted loss, (2) waveform augmentation "
            "(time-shift ±0.5s, amplitude scale ×[0.5,2]), "
            "(3) oversampling thunder, (4) undersample majority class."
        ),
    }
    for name, ns in window_specs:
        fname = f"waveforms_{name}.npy"
        fpath = output_dir / fname
        index["variants"][name] = {
            "file"    : fname,
            "shape"   : [n, 3, ns],
            "dtype"   : "float32",
            "size_mb" : round(n * 3 * ns * 4 / (1024**2), 3),
        }

    with open(output_dir / "index.json", "w") as fp:
        json.dump(index, fp, indent=2)

    # --- Summary ---
    print("\n=== Output files ===")
    for name, ns in window_specs:
        fname = f"waveforms_{name}.npy"
        fpath = output_dir / fname
        size_mb = fpath.stat().st_size / (1024**2)
        print(f"  {fname:<25s}  shape=({n}, 3, {ns:>6})  {size_mb:.2f} MB")
    print(f"  {'trace_names.npy':<25s}  {n} entries")
    print(f"  {'metadata.npz':<25s}  p_arrival, snr_db, station coords")
    print(f"  {'index.json':<25s}  variant metadata")
    print(f"\nAll files written to: {output_dir}")
    print(f"\n[Class imbalance reminder]")
    print(f"  Thunder traces : {n}")
    print(f"  Suggested actions:")
    print(f"    1. Weighted loss  – weight_thunder = N_total / (n_classes × {n})")
    print(f"    2. Augmentation   – time-shift ±0.5s, amplitude ×[0.5,2], add Gaussian noise")
    print(f"    3. Oversampling   – repeat thunder until ~balanced with other classes")
    print(f"    4. Undersampling  – cap earthquake to 5–10× thunder count during training")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert PNW exotic thunder class to NumPy memmap files")
    parser.add_argument(
        "--hdf5",
        default="/home/indra/indra/PNW/exotic_waveforms.hdf5",
        help="Path to PNW exotic HDF5 file")
    parser.add_argument(
        "--csv",
        default="/home/indra/indra/PNW/exotic_metadata.csv",
        help="Path to exotic_metadata.csv")
    parser.add_argument(
        "--output",
        default="/home/indra/indra/PNW/memmap_thunder",
        help="Output directory (will be created if absent)")
    parser.add_argument(
        "--windows", nargs="+", default=["3", "5", "10"],
        help="Window variants in seconds, or 'full' for the full 180s trace. "
             "Default: 3 5 10")
    parser.add_argument(
        "--pre-p", type=float, default=DEFAULT_PRE_P_SEC,
        help=f"Seconds of pre-P data to include in windows (default: {DEFAULT_PRE_P_SEC})")
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Traces per flush cycle (default: {DEFAULT_CHUNK_SIZE}). "
             "With only ~146 thunder traces the whole set fits in one chunk.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume interrupted conversion")
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
