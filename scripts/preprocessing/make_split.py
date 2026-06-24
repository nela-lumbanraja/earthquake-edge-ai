"""
make_splits.py
Membuat split bersama untuk semua model:
- train 70%
- validation 10%
- test 20%

Split ini masih random per-trace, tetapi sudah:
1. memakai LABEL_MAP kanonik 7 kelas
2. dipakai bersama oleh semua model
3. disimpan agar test set semua model sama
"""

import os
import hashlib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# CONFIG
# ==========================================================
DATA_DIR = "/home/indra/eq_team"
META_FILE = os.path.join(DATA_DIR, "metadata_5s.npy")

PROJECT_DIR = "/home/indra/eq_team/earthquake-classification"
SPLIT_DIR = os.path.join(PROJECT_DIR, "data")
SPLIT_FILE = os.path.join(SPLIT_DIR, "splits_5s.npz")

RANDOM_STATE = 42

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
# UTILITY
# ==========================================================
def sha256sum(file_path):
    h = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()

# ==========================================================
# MAIN
# ==========================================================
def main():

    print("=" * 60)
    print("MAKE SHARED SPLIT 5S")
    print("=" * 60)

    os.makedirs(SPLIT_DIR, exist_ok=True)

    print("\n[1] Loading metadata...")
    meta = np.load(META_FILE, allow_pickle=True).item()

    labels_raw = np.array(meta["label"])

    print("Total samples :", len(labels_raw))
    print("Original labels:")
    unique_raw, count_raw = np.unique(labels_raw, return_counts=True)

    for label, count in zip(unique_raw, count_raw):
        print(f"  {label:25s}: {count}")

    print("\n[2] Applying canonical LABEL_MAP...")

    labels_mapped = np.array([
        LABEL_MAP.get(str(label), str(label))
        for label in labels_raw
    ])

    unique_mapped, count_mapped = np.unique(
        labels_mapped,
        return_counts=True
    )

    print("Mapped labels:")
    for label, count in zip(unique_mapped, count_mapped):
        print(f"  {label:15s}: {count}")

    print("\n[3] Encoding labels...")

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels_mapped)

    print("Classes:", list(le.classes_))
    print("Num classes:", len(le.classes_))

    print("\n[4] Creating train/val/test split...")

    idx = np.arange(len(labels_encoded))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels_encoded
    )

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.125,
        random_state=RANDOM_STATE,
        stratify=labels_encoded[train_idx]
    )

    print("Train :", len(train_idx))
    print("Val   :", len(val_idx))
    print("Test  :", len(test_idx))

    print("\n[5] Checking class distribution...")

    def print_distribution(name, indices):
        print(f"\n{name}:")
        split_labels = labels_mapped[indices]
        unique, counts = np.unique(split_labels, return_counts=True)

        for label, count in zip(unique, counts):
            print(f"  {label:15s}: {count}")

    print_distribution("Train", train_idx)
    print_distribution("Val", val_idx)
    print_distribution("Test", test_idx)

    print("\n[6] Saving split file...")

    np.savez(
        SPLIT_FILE,
        train=train_idx,
        val=val_idx,
        test=test_idx,
        classes=le.classes_,
        split_type="random_per_trace_v2",
        random_state=RANDOM_STATE
    )

    split_sha = sha256sum(SPLIT_FILE)

    print("Saved split :", SPLIT_FILE)
    print("SHA256      :", split_sha)

    print("\nDONE")


if __name__ == "__main__":
    main()