import os
import json
import numpy as np

# =========================================================
# PATH
# =========================================================
PNW_DIR = "/home/indra/indra/PNW"
STEAD_DIR = "/home/indra/indra/STEAD"
OUT_DIR = "/home/indra/eq_team"

os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# SOURCE FOLDERS
# =========================================================
pnw_folders = [
    "memmap_earthquake",
    "memmap_explosion",
    "memmap_sonic",
    "memmap_thunder",
    "memmap_surface_event"
]

stead_folders = [
    "memmap",
    "memmap_no_event",
    "memmap_noise"
]

# =========================================================
# METADATA KEYS
# =========================================================
META_KEYS = [
    "back_azimuth_deg",
    "p_arrival_sample",
    "source_magnitude",
    "source_distance_km",
    "snr_db",
]

# =========================================================
# INDEX LOADER
# =========================================================
def get_index(folder_path, folder_name):

    if folder_name == "memmap":
        index_file = "index.json"
    elif folder_name == "memmap_no_event":
        index_file = "noevent_index.json"
    elif folder_name == "memmap_noise":
        index_file = "noise_index.json"
    else:
        index_file = "index.json"

    path = os.path.join(folder_path, index_file)

    with open(path, "r") as f:
        return json.load(f)


# =========================================================
# METADATA LOADER
# =========================================================
def load_metadata(folder_path):

    meta_path = os.path.join(folder_path, "metadata.npz")

    if not os.path.exists(meta_path):
        print(f"[WARN] metadata missing: {meta_path}")
        return None

    return np.load(meta_path, allow_pickle=True)


# =========================================================
# COMBINE
# =========================================================
for sec in ["3s", "5s", "10s"]:

    print(f"\n==================== {sec} ====================")

    all_sources = []

    total_rows = 0
    dtype = None
    channels = None
    samples = None

    # =====================================================
    # PNW
    # =====================================================
    for folder in pnw_folders:

        fp = os.path.join(PNW_DIR, folder)
        info = get_index(fp, folder)

        print("PNW :", folder)

        if folder == "memmap_surface_event":

            key = f"waveforms_{sec}"
            file_name = info["files"][key]
            file_path = os.path.join(fp, file_name)

            arr = np.load(file_path, mmap_mode="r")

            var = {
                "file": file_name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype)
            }

        else:
            var = info["variants"][sec]

        all_sources.append((fp, folder, var))

        total_rows += var["shape"][0]
        channels = var["shape"][1]
        samples = var["shape"][2]
        dtype = np.dtype(var["dtype"])

    # =====================================================
    # STEAD
    # =====================================================
    for folder in stead_folders:

        fp = os.path.join(STEAD_DIR, folder)
        info = get_index(fp, folder)

        print("STEAD:", folder)

        var = info["variants"][sec]

        all_sources.append((fp, folder, var))
        total_rows += var["shape"][0]

    print(f"\nTotal traces = {total_rows:,}")
    print(f"Shape        = ({total_rows}, {channels}, {samples})")
    print(f"Dtype        = {dtype}")

    # =====================================================
    # OUTPUT WAVEFORM MEMMAP
    # =====================================================
    out_wave = os.path.join(
        OUT_DIR,
        f"combined_{sec}.npy"
    )

    combined = np.memmap(
        out_wave,
        dtype=dtype,
        mode="w+",
        shape=(total_rows, channels, samples)
    )

    # =====================================================
    # METADATA PREALLOC
    # =====================================================
    meta_arrays = {
        key: np.full(
            total_rows,
            np.nan,
            dtype=np.float32
        )
        for key in META_KEYS
    }

    label_array = np.empty(
        total_rows,
        dtype=object
    )

    # =====================================================
    # COPY LOOP
    # =====================================================
    start = 0

    for fp, folder, var in all_sources:

        file_path = os.path.join(
            fp,
            var["file"]
        )

        shape = tuple(var["shape"])

        n = shape[0]
        end = start + n

        print(f"\ncopy -> {file_path}")
        print(f"rows : {start:,} - {end:,}")

        # ==========================================
        # LOAD WAVEFORM
        # ==========================================
        if "surface_event" in fp:

            data = np.load(
                file_path,
                mmap_mode="r"
            )

        else:

            data = np.memmap(
                file_path,
                dtype=np.dtype(var["dtype"]),
                mode="r",
                shape=shape
            )

        combined[start:end] = data

        # ==========================================
        # LABEL
        # ==========================================
        label_array[start:end] = folder

        # ==========================================
        # METADATA
        # ==========================================
        meta = load_metadata(fp)

        if meta is not None:

            for key in META_KEYS:

                if key in meta.files:

                    arr = np.asarray(
                        meta[key],
                        dtype=np.float32
                    )

                    if len(arr) == n:

                        meta_arrays[key][start:end] = arr

                    else:

                        print(
                            f"[WARN] {key} len mismatch "
                            f"{len(arr)} != {n}"
                        )

                else:
                    print(
                        f"[WARN] {key} "
                        f"missing in {fp}"
                    )

        start = end

    combined.flush()

    print(f"\nwaveform saved -> {out_wave}")

    # =====================================================
    # METADATA DICTIONARY → .NPY
    # =====================================================
    metadata_dict = {
        "label": label_array,
        "shape": np.array(
            [total_rows, channels, samples]
        ),
        "dtype": str(dtype)
    }

    metadata_dict.update(meta_arrays)

    meta_out = os.path.join(
        OUT_DIR,
        f"metadata_{sec}.npy"
    )

    np.save(
        meta_out,
        metadata_dict,
        allow_pickle=True
    )

    print(f"metadata saved -> {meta_out}")

    # =====================================================
    # SUMMARY
    # =====================================================
    print("\nMetadata summary")

    for key in META_KEYS:

        arr = meta_arrays[key]
        valid = np.sum(~np.isnan(arr))

        print(
            f"{key:<25s}: "
            f"{valid:,}/{total_rows:,}"
        )

    print(
        "labels:",
        np.unique(label_array)
    )

print("\n✓ ALL COMPLETE")