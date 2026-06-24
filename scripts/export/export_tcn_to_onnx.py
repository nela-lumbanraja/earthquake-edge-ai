# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import onnx


# =====================================================
# CONFIG
# =====================================================

PROJECT_DIR = "/home/indra/eq_team/earthquake-classification"

CKPT_PATH = os.path.join(
    PROJECT_DIR,
    "results",
    "runs",
    "20260618_TCN",
    "best_model.pt"
)

SPLIT_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "splits_5s.npz"
)

ONNX_DIR = os.path.join(
    PROJECT_DIR,
    "models",
    "onnx"
)

os.makedirs(ONNX_DIR, exist_ok=True)

ONNX_PATH = os.path.join(
    ONNX_DIR,
    "tcn_model.onnx"
)

SEQ_LEN = 500
TCN_CHANNELS = [64, 128, 128, 256]
KERNEL_SIZE = 7
DROPOUT = 0.2
OPSET_VERSION = 13


# =====================================================
# MODEL TCN
# =====================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float
    ) -> None:
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.downsample = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            )
        else:
            self.downsample = None

        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        if self.downsample is None:
            residual = x
        else:
            residual = self.downsample(x)

        return self.final_relu(out + residual)


class TCNClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: list[int],
        kernel_size: int = 7,
        dropout: float = 0.2
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_channels = in_channels

        for i, out_channels in enumerate(channels):
            dilation = 2 ** i

            layers.append(
                TemporalBlock(
                    prev_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

            prev_channels = out_channels

        self.tcn = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# =====================================================
# LOAD CLASSES
# =====================================================

print("=" * 60)
print("TCN → ONNX Export")
print("=" * 60)

print("\n[1] Loading classes...")

split = np.load(SPLIT_FILE, allow_pickle=True)
classes = list(split["classes"])
num_classes = len(classes)

print("Classes    :", classes)
print("Num classes:", num_classes)


# =====================================================
# BUILD MODEL
# =====================================================

print("\n[2] Building TCN model...")

model = TCNClassifier(
    in_channels=3,
    num_classes=num_classes,
    channels=TCN_CHANNELS,
    kernel_size=KERNEL_SIZE,
    dropout=DROPOUT
)

print("Loading checkpoint:", CKPT_PATH)

checkpoint: Any = torch.load(
    CKPT_PATH,
    map_location="cpu",
    weights_only=False
)

if isinstance(checkpoint, dict):
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

total_params = sum(p.numel() for p in model.parameters())

print("Parameters :", f"{total_params:,}")


# =====================================================
# DUMMY INPUT
# =====================================================

dummy_input = torch.randn(
    1,
    3,
    SEQ_LEN,
    dtype=torch.float32
)

print("Dummy input shape:", tuple(dummy_input.shape))


# =====================================================
# EXPORT ONNX
# =====================================================

if os.path.exists(ONNX_PATH):
    os.remove(ONNX_PATH)

print("\n[3] Exporting ONNX...")
print("Output file:", ONNX_PATH)

torch.onnx.export(
    model,
    (dummy_input,),
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
# VALIDATE ONNX
# =====================================================

print("\n[4] Validating ONNX...")

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)


# =====================================================
# FILE INFO
# =====================================================

onnx_size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)

print("\nExport selesai dan ONNX valid")
print("ONNX file :", ONNX_PATH)
print("ONNX size :", f"{onnx_size_mb:.2f} MB")
print("Opset     :", OPSET_VERSION)

print("\nDONE")