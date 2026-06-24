import os
import torch
import torch.nn as nn
import onnx

from torchvision.models import mobilenet_v2

# =====================================================
# CONFIG
# =====================================================

PROJECT_DIR = "/home/indra/eq_team/earthquake-classification"

CKPT_PATH = (
    PROJECT_DIR +
    "/results/runs/2026-06-22_18-44-06_mobilenetv2_v2_nogit/"
    "checkpoints/best_mobilenetv2_v2.pt"
)

ONNX_DIR = os.path.join(
    PROJECT_DIR,
    "models",
    "mobilenetv2"
)

os.makedirs(
    ONNX_DIR,
    exist_ok=True
)

ONNX_PATH = os.path.join(
    ONNX_DIR,
    "mobilenetv2_v2.onnx"
)

OPSET_VERSION = 13

# =====================================================
# LOAD CHECKPOINT
# =====================================================

print("=" * 60)
print("MobileNetV2 → ONNX Export")
print("=" * 60)

print("\n[1] Loading checkpoint...")

checkpoint = torch.load(
    CKPT_PATH,
    map_location="cpu",
    weights_only=False
)

classes = checkpoint["classes"]
num_classes = len(classes)

print("Checkpoint :", CKPT_PATH)
print("Classes    :", classes)
print("Num classes:", num_classes)

# =====================================================
# BUILD MODEL
# =====================================================

print("\n[2] Building MobileNetV2...")

model = mobilenet_v2(
    weights=None
)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    num_classes
)

model.load_state_dict(
    checkpoint["state_dict"]
)

model.eval()

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("Parameters :", f"{total_params:,}")

# =====================================================
# DUMMY INPUT
# =====================================================

dummy_input = torch.randn(
    1,
    3,
    224,
    224,
    dtype=torch.float32
)

# =====================================================
# REMOVE OLD ONNX
# =====================================================

if os.path.exists(ONNX_PATH):
    os.remove(ONNX_PATH)

data_file = ONNX_PATH + ".data"

if os.path.exists(data_file):
    os.remove(data_file)

# =====================================================
# EXPORT
# =====================================================

print("\n[3] Exporting ONNX...")
print("Output dir  :", ONNX_DIR)
print("Output file :", ONNX_PATH)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=OPSET_VERSION,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {
            0: "batch_size"
        },
        "output": {
            0: "batch_size"
        }
    },
    dynamo=False
)

# =====================================================
# VALIDATE
# =====================================================

print("\n[4] Validating ONNX...")

onnx_model = onnx.load(
    ONNX_PATH
)

onnx.checker.check_model(
    onnx_model
)

# =====================================================
# SAVE CLASSES (penting — dipakai saat inference di RPi)
# =====================================================

import json

classes_path = os.path.join(
    ONNX_DIR,
    "mobilenetv2_v2_classes.json"
)

with open(classes_path, "w") as f:
    json.dump(
        {"classes": classes},
        f,
        indent=2
    )

print("Classes file:", classes_path)

# =====================================================
# FILE INFO
# =====================================================

onnx_size_mb = (
    os.path.getsize(ONNX_PATH)
    / (1024 * 1024)
)

print("\nExport selesai dan ONNX valid")
print("ONNX file :", ONNX_PATH)
print("ONNX size :", f"{onnx_size_mb:.2f} MB")
print("Opset     :", OPSET_VERSION)

if os.path.exists(data_file):

    data_size_mb = (
        os.path.getsize(data_file)
        / (1024 * 1024)
    )

    print("ONNX data :", data_file)
    print("Data size :", f"{data_size_mb:.2f} MB")

print("\nIsi folder output:")

for f in sorted(os.listdir(ONNX_DIR)):
    path = os.path.join(ONNX_DIR, f)

    if os.path.isfile(path):
        size_mb = (
            os.path.getsize(path)
            / (1024 * 1024)
        )

        print(
            f" - {f} ({size_mb:.2f} MB)"
        )

print("\nDONE")