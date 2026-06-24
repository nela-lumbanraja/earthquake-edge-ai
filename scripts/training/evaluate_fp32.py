import os
import time
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
    "fp32_evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 256
IMG_SIZE = 224
NUM_WORKERS = 8


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
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading metadata...")

meta = np.load(
    META_FILE,
    allow_pickle=True
).item()

labels_raw = np.array(meta["label"])

shape = tuple(meta["shape"])
dtype = np.dtype(meta["dtype"])

data = np.memmap(
    DATA_FILE,
    dtype=dtype,
    mode="r",
    shape=shape
)

split = np.load(
    SPLIT_FILE,
    allow_pickle=True
)

test_idx = split["test"]
classes = list(split["classes"])

labels_canonical = np.array([
    LABEL_MAP.get(str(x), str(x))
    for x in labels_raw
])

class_to_idx = {
    c: i
    for i, c in enumerate(classes)
}

labels = np.array([
    class_to_idx[x]
    for x in labels_canonical
])

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

print("Loading model...")

model = CNN5s(len(classes))

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

model_size = (
    os.path.getsize(BEST_MODEL_PATH)
    / 1024
    / 1024
)

print(f"Model size: {model_size:.2f} MB")

# ==========================================================
# EVALUATION
# ==========================================================

all_preds = []
all_true = []

start_time = time.time()

with torch.no_grad():

    for xb, yb in tqdm(
        test_loader,
        desc="Evaluating FP32"
    ):

        logits = model(xb)

        preds = logits.argmax(dim=1)

        all_preds.extend(
            preds.numpy()
        )

        all_true.extend(
            yb.numpy()
        )

elapsed = time.time() - start_time

all_preds = np.array(all_preds)
all_true = np.array(all_true)

acc = accuracy_score(
    all_true,
    all_preds
)

print(f"\nAccuracy: {acc:.6f}")
print(f"Inference Time: {elapsed:.2f} sec")

# ==========================================================
# REPORT
# ==========================================================

report = classification_report(
    all_true,
    all_preds,
    target_names=classes,
    digits=4,
    zero_division=0
)

print(report)

with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report_fp32.txt"
    ),
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

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)

plt.title("CNN5s FP32 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix_fp32.png"
    ),
    dpi=150
)

plt.close()

print("\n====================================")
print("FP32 EVALUATION DONE")
print("====================================")
print(f"Accuracy      : {acc:.6f}")
print(f"Model Size MB : {model_size:.2f}")
print(f"Inference Sec : {elapsed:.2f}")