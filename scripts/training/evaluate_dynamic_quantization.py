import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# ==========================================================
# CONFIG
# ==========================================================

DATA_DIR = "/home/indra/eq_team"
PROJECT_DIR = "/home/indra/eq_team/earthquake-classification"

DATA_FILE = os.path.join(DATA_DIR, "combined_5s.npy")
META_FILE = os.path.join(DATA_DIR, "metadata_5s.npy")

SPLIT_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "splits_5s.npz"
)

BEST_MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "results",
    "runs",
    "20260617_201329_CNN5s",
    "checkpoints",
    "best_cnn_5s.pt"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "dynamic_quant_evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 256
IMG_SIZE = 224
NUM_WORKERS = 8

DEVICE = "cpu"


# ==========================================================
# LABEL MAP
# ==========================================================

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
# DATASET
# ==========================================================

class EqDataset(Dataset):

    def __init__(self, data, labels, indices):
        self.data = data
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        i = self.indices[idx]

        x = self.data[i].copy()

        mean = x.mean()
        std = x.std() + 1e-8
        x = (x - mean) / std

        x = torch.tensor(
            x,
            dtype=torch.float32
        )

        x = x.unsqueeze(0)

        x = F.interpolate(
            x,
            size=IMG_SIZE,
            mode="linear",
            align_corners=False
        )

        x = x.squeeze(0)

        x = x.unsqueeze(-1).repeat(
            1,
            1,
            IMG_SIZE
        )

        y = torch.tensor(
            self.labels[i],
            dtype=torch.long
        )

        return x, y


# ==========================================================
# CNN5s
# ==========================================================

class CNN5s(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                256,
                512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Dropout(0.4),

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ==========================================================
# LOAD METADATA
# ==========================================================

print("\nLoading metadata...")

meta = np.load(
    META_FILE,
    allow_pickle=True
).item()

labels_raw = np.array(
    meta["label"]
)

shape = tuple(
    meta["shape"]
)

dtype = np.dtype(
    meta["dtype"]
)

print("Shape :", shape)
print("Dtype :", dtype)


# ==========================================================
# LOAD MEMMAP
# ==========================================================

print("\nLoading memmap waveform...")

data = np.memmap(
    DATA_FILE,
    dtype=dtype,
    mode="r",
    shape=shape
)

print("Total samples :", len(data))


# ==========================================================
# LOAD SPLIT
# ==========================================================

print("\nLoading split...")

split = np.load(
    SPLIT_FILE,
    allow_pickle=True
)

test_idx = split["test"]

classes = list(
    split["classes"]
)

print("Classes :", classes)
print("Test samples :", len(test_idx))


# ==========================================================
# LABEL ENCODING
# ==========================================================

labels_canonical = np.array([
    LABEL_MAP.get(
        str(label),
        str(label)
    )
    for label in labels_raw
])

classes_canonical = []

for cls in classes:

    mapped = LABEL_MAP.get(
        str(cls),
        str(cls)
    )

    if mapped not in classes_canonical:
        classes_canonical.append(mapped)

for cls in np.unique(
    labels_canonical
):
    if cls not in classes_canonical:
        classes_canonical.append(cls)

class_to_idx = {
    c: i
    for i, c in enumerate(
        classes_canonical
    )
}

labels = np.array([
    class_to_idx[x]
    for x in labels_canonical
])

NUM_CLASSES = len(
    classes_canonical
)

print("Num classes :", NUM_CLASSES)


# ==========================================================
# TEST LOADER
# ==========================================================

test_loader = DataLoader(
    EqDataset(
        data,
        labels,
        test_idx
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ==========================================================
# LOAD MODEL
# ==========================================================

print("\nLoading best model...")

model = CNN5s(
    NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

baseline_size = (
    os.path.getsize(
        BEST_MODEL_PATH
    )
    / 1024
    / 1024
)

print(
    f"Baseline size : "
    f"{baseline_size:.2f} MB"
)


# ==========================================================
# DYNAMIC QUANTIZATION
# ==========================================================

print("\nApplying Dynamic Quantization...")

quantized_model = torch.quantization.quantize_dynamic(
    copy.deepcopy(model),
    {nn.Linear},
    dtype=torch.qint8
)

quantized_model.eval()

QUANT_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "best_cnn_5s_dynamic_quantized.pt"
)

torch.save(
    quantized_model.state_dict(),
    QUANT_MODEL_PATH
)

quant_size = (
    os.path.getsize(
        QUANT_MODEL_PATH
    )
    / 1024
    / 1024
)

print(
    f"Quantized size : "
    f"{quant_size:.2f} MB"
)


# ==========================================================
# EVALUATION
# ==========================================================

print("\nRunning inference...")

all_preds = []
all_true = []

with torch.no_grad():

    for xb, yb in tqdm(
        test_loader,
        desc="Evaluating"
    ):

        logits = quantized_model(
            xb
        )

        preds = logits.argmax(
            dim=1
        )

        all_preds.extend(
            preds.numpy()
        )

        all_true.extend(
            yb.numpy()
        )

all_preds = np.array(
    all_preds
)

all_true = np.array(
    all_true
)

acc = accuracy_score(
    all_true,
    all_preds
)

print(
    f"\nAccuracy : "
    f"{acc:.6f}"
)


# ==========================================================
# REPORT
# ==========================================================

report = classification_report(
    all_true,
    all_preds,
    target_names=classes_canonical,
    digits=4,
    zero_division=0
)

print("\n")
print(report)

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report_dynamic_quantized.txt"
)

with open(
    report_path,
    "w"
) as f:
    f.write(report)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    all_true,
    all_preds
)

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes_canonical,
    yticklabels=classes_canonical
)

plt.title(
    "CNN5s Dynamic Quantized Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "True"
)

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix_dynamic_quantized.png"
)

plt.savefig(
    cm_path,
    dpi=150
)

plt.close()


print("\n===================================")
print("DONE")
print("===================================")
print("Accuracy           :", acc)
print("Baseline size (MB) :", round(baseline_size, 2))
print("Quantized size(MB) :", round(quant_size, 2))
print("Report             :", report_path)
print("Confusion Matrix   :", cm_path)
print("Quantized Model    :", QUANT_MODEL_PATH)