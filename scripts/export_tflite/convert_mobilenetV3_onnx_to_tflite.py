import os
import subprocess
import tensorflow as tf

PROJECT_DIR = "/home/indra/eq_team/earthquake-classification"

ONNX_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "onnx",
    "mobilenetv3_v2.onnx"
)

TF_DIR = os.path.join(
    PROJECT_DIR,
    "models",
    "tflite",
    "mobilenetv3_v2_saved_model"
)

TFLITE_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "tflite",
    "mobilenetv3_v2_float32.tflite"
)

os.makedirs(
    os.path.dirname(TFLITE_PATH),
    exist_ok=True
)

print("=" * 60)
print("ONNX to TFLite Conversion")
print("=" * 60)

print("\n[1] ONNX file:")
print(ONNX_PATH)

print("\n[2] Converting ONNX to TensorFlow SavedModel...")

subprocess.run(
    [
        "onnx2tf",
        "-i", ONNX_PATH,
        "-o", TF_DIR,
    ],
    check=True
)

print("\n[3] Converting SavedModel to TFLite FP32...")

converter = tf.lite.TFLiteConverter.from_saved_model(TF_DIR)

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)

print("\nConversion selesai")
print("SavedModel :", TF_DIR)
print("TFLite     :", TFLITE_PATH)
print("Size       :", f"{size_mb:.2f} MB")
print("DONE")