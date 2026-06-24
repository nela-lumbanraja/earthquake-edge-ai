"""
Export CNN5s (waveform classification) -> ONNX
================================================

Meng-export checkpoint best_cnn_5s.pt (plain state_dict, FP32) ke ONNX,
lalu memvalidasi struktur DAN kesetaraan numerik terhadap PyTorch.

Catatan penting (preprocessing harus identik saat inference):
  Model TIDAK menerima waveform mentah (3, 500). Saat training tiap sampel
  diproses jadi pseudo-image (3, 224, 224) dengan langkah:
    1) z-score per-trace atas gabungan 3 channel (mean/std skalar global per sampel)
    2) interpolate linear (3, 500) -> (3, 224)
    3) repeat sepanjang sumbu terakhir -> (3, 224, 224)
  Pipeline inference WAJIB meniru langkah ini, kalau tidak output ONNX tidak
  bermakna meski file-nya valid.

Urutan kelas (index -> label) HARUS sama dengan training (alfabetis):
  0 earthquake, 1 explosion, 2 no_event, 3 noise, 4 sonic, 5 surface_event, 6 thunder
"""

import os
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort


# =====================================================
# CONFIG
# =====================================================
PROJECT_DIR = "/home/indra/eq_team/earthquake-classification"
RUN_DIR = os.path.join(PROJECT_DIR, "results", "runs", "20260617_201329_CNN5s")
CKPT_PATH = os.path.join(RUN_DIR, "checkpoints", "best_cnn_5s.pt")

ONNX_DIR = os.path.join(PROJECT_DIR, "models", "cnn_v2")
ONNX_PATH = os.path.join(ONNX_DIR, "cnn_V2.onnx")

OPSET_VERSION = 13
IMG_SIZE = 224

# Urutan kanonik sesuai run_summary.txt / training (alfabetis). JANGAN diubah.
CLASSES = ["earthquake", "explosion", "no_event", "noise",
           "sonic", "surface_event", "thunder"]

os.makedirs(ONNX_DIR, exist_ok=True)


# =====================================================
# MODEL (arsitektur identik dengan training)
# =====================================================
class CNN5s(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_state_dict(path):
    """Robust: dukung plain state_dict maupun checkpoint dict berbungkus."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def infer_num_classes(state_dict):
    """Ambil num_classes dari bobot Linear terakhir, bukan hardcode."""
    return state_dict["classifier.5.weight"].shape[0]


# =====================================================
# 1) LOAD MODEL
# =====================================================
print("=" * 60)
print("CNN5s -> ONNX Export")
print("=" * 60)

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint tidak ditemukan: {CKPT_PATH}")

print("\n[1] Loading checkpoint & building model...")
state_dict = load_state_dict(CKPT_PATH)
num_classes = infer_num_classes(state_dict)

if num_classes != len(CLASSES):
    raise ValueError(
        f"Mismatch: checkpoint punya {num_classes} kelas, "
        f"CLASSES punya {len(CLASSES)}. Perbaiki daftar CLASSES."
    )

model = CNN5s(num_classes)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing or unexpected:
    print("  WARNING missing keys   :", missing)
    print("  WARNING unexpected keys:", unexpected)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print("  Checkpoint :", CKPT_PATH)
print("  Num classes:", num_classes)
print("  Parameters :", f"{total_params:,}")


# =====================================================
# 2) EXPORT ONNX
# =====================================================
print("\n[2] Exporting ONNX...")
for p in (ONNX_PATH, ONNX_PATH + ".data"):
    if os.path.exists(p):
        os.remove(p)

dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)

export_kwargs = dict(
    export_params=True,
    opset_version=OPSET_VERSION,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
# 'dynamo' hanya ada di torch >= 2.5; pakai TorchScript exporter klasik kalau tersedia.
try:
    torch.onnx.export(model, dummy_input, ONNX_PATH, dynamo=False, **export_kwargs)
except TypeError:
    torch.onnx.export(model, dummy_input, ONNX_PATH, **export_kwargs)

print("  Output file :", ONNX_PATH)


# =====================================================
# 3) VALIDASI STRUKTUR + EMBED METADATA
# =====================================================
print("\n[3] Validating ONNX structure & embedding metadata...")
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

# Simpan daftar kelas di metadata model supaya pipeline inference tidak salah urutan.
meta = onnx_model.metadata_props.add()
meta.key = "classes"
meta.value = ",".join(CLASSES)
onnx.save(onnx_model, ONNX_PATH)
print("  Struktur valid. Metadata 'classes' tertanam.")


# =====================================================
# 4) PARITY CHECK: PyTorch vs ONNX Runtime
# =====================================================
print("\n[4] Numerical parity check (PyTorch vs ONNXRuntime)...")
np.random.seed(0)
test_in = torch.randn(4, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)

with torch.no_grad():
    torch_out = model(test_in).numpy()

sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
onnx_out = sess.run(["output"], {"input": test_in.numpy()})[0]

max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))
argmax_match = bool(np.array_equal(torch_out.argmax(1), onnx_out.argmax(1)))
print(f"  Max abs diff   : {max_abs_diff:.3e}")
print(f"  Argmax cocok   : {argmax_match}")
assert argmax_match, "Prediksi ONNX tidak cocok dengan PyTorch!"
assert max_abs_diff < 1e-3, "Selisih numerik terlalu besar!"
print("  PASS - ONNX setara dengan PyTorch.")


# =====================================================
# 5) INFO FILE
# =====================================================
size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
print("\n" + "=" * 60)
print("Export selesai & terverifikasi")
print("  ONNX file :", ONNX_PATH)
print("  ONNX size :", f"{size_mb:.2f} MB")
print("  Opset     :", OPSET_VERSION)
print("  Classes   :", CLASSES)
print("DONE")