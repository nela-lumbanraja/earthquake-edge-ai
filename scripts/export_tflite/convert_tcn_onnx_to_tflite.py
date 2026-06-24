# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import os
import subprocess
import shutil

PROJECT_DIR = "/home/indra/eq_team/earthquake-classification"

ONNX_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "onnx",
    "tcn_model.onnx"
)

OUT_DIR = os.path.join(
    PROJECT_DIR,
    "models",
    "tflite",
    "tcn_onnx2tf"
)

FINAL_TFLITE = os.path.join(
    PROJECT_DIR,
    "models",
    "tflite",
    "tcn_model_float32.tflite"
)

os.makedirs(
    os.path.dirname(FINAL_TFLITE),
    exist_ok=True
)

if not os.path.exists(ONNX_PATH):
    raise FileNotFoundError(
        "File ONNX tidak ditemukan: " + ONNX_PATH
    )

if shutil.which("onnx2tf") is None:
    raise RuntimeError(
        "onnx2tf belum terinstall di environment ini."
    )

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)

print("=" * 60)
print("TCN ONNX to TFLite with onnx2tf")
print("=" * 60)

print("Input ONNX :", ONNX_PATH)
print("Output dir :", OUT_DIR)

subprocess.run(
    [
        "onnx2tf",
        "-i",
        ONNX_PATH,
        "-o",
        OUT_DIR,
        "-cotof"
    ],
    check=True
)

tflite_files: list[str] = []

for root, _dirs, files in os.walk(OUT_DIR):
    for file in files:
        if file.endswith(".tflite"):
            tflite_path = os.path.join(root, file)
            tflite_files.append(tflite_path)

if not tflite_files:
    raise FileNotFoundError(
        "Tidak ada file .tflite yang dihasilkan oleh onnx2tf. "
        "Cek output folder: " + OUT_DIR
    )

src_tflite: str = tflite_files[0]

shutil.copy2(
    src_tflite,
    FINAL_TFLITE
)

size_mb = os.path.getsize(FINAL_TFLITE) / (1024 * 1024)

print("\nConversion selesai")
print("Source TFLite :", src_tflite)
print("Final TFLite  :", FINAL_TFLITE)
print("Size          :", f"{size_mb:.2f} MB")
print("DONE")