"""
WaveformTransformer ONNX -> TFLite Conversion

Catatan: onnx2tf langsung menghasilkan file .tflite tanpa melalui
SavedModel, sehingga step konversi SavedModel -> TFLite dihapus.
"""

import os
os.environ["TFLITE_DISABLE_XNNPACK"] = "1"  # harus sebelum import tf

import subprocess
import numpy as np
import tensorflow as tf

# =====================================================
# CONFIG
# =====================================================

PROJECT_DIR = "/home/indra/eq_team/earthquake-classification"

VARIANT = "teacher"

ONNX_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "transformer",
    f"waveform_transformer_{VARIANT}.onnx"
)

# Folder output onnx2tf -- file .tflite akan langsung ada di sini
TF_DIR = os.path.join(
    PROJECT_DIR,
    "models",
    "transformer",
    "TfLite"
)

# Path file .tflite float32 yang dihasilkan onnx2tf
TFLITE_PATH = os.path.join(
    TF_DIR,
    f"waveform_transformer_{VARIANT}_float32.tflite"
)

# =====================================================
# VALIDASI & PERSIAPAN
# =====================================================

if not os.path.exists(ONNX_PATH):
    raise FileNotFoundError(
        f"File ONNX tidak ditemukan: {ONNX_PATH}\n"
        f"Jalankan dulu export_transformer_onnx.py dengan VARIANT='{VARIANT}' "
        f"untuk menghasilkan file ini."
    )

os.makedirs(TF_DIR, exist_ok=True)

print("=" * 60)
print("WaveformTransformer ONNX -> TFLite Conversion")
print("=" * 60)
print(f"Variant    : {VARIANT}")
print(f"ONNX path  : {ONNX_PATH}")
print(f"Output dir : {TF_DIR}")

# =====================================================
# [1] ONNX -> TFLite (via onnx2tf)
# =====================================================
# onnx2tf langsung menghasilkan .tflite di TF_DIR,
# tidak melalui SavedModel.

print("\n[1] Converting ONNX -> TFLite via onnx2tf...")

base_cmd = [
    "onnx2tf",
    "-i", ONNX_PATH,
    "-o", TF_DIR,
]

try:
    subprocess.run(base_cmd, check=True)
    print("  Konversi default berhasil.")

except subprocess.CalledProcessError as e:
    print(f"\n  [PERINGATAN] Konversi default gagal (exit code {e.returncode}).")
    print("  Mencoba ulang dengan flag fallback...")

    fallback_cmd = base_cmd + [
        "-b", "1",
        "-nuo",
    ]
    print("  Command fallback:", " ".join(fallback_cmd))

    subprocess.run(fallback_cmd, check=True)
    print("  Konversi fallback berhasil.")

# Validasi file .tflite benar-benar ada setelah konversi
if not os.path.exists(TFLITE_PATH):
    raise FileNotFoundError(
        f"File TFLite tidak ditemukan setelah konversi: {TFLITE_PATH}\n"
        f"Cek isi folder: {TF_DIR}"
    )

size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
print(f"\nKonversi selesai.")
print(f"TFLite : {TFLITE_PATH}")
print(f"Size   : {size_mb:.2f} MB")

# =====================================================
# [2] Sanity check: jalankan TFLite interpreter sekali
# =====================================================

print("\n[2] Sanity check TFLite interpreter...")

try:
    interpreter = tf.lite.Interpreter(
        model_path=TFLITE_PATH,
        num_threads=1
    )
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("  Input details  :", [(d["name"], d["shape"], d["dtype"]) for d in input_details])
    print("  Output details :", [(d["name"], d["shape"], d["dtype"]) for d in output_details])

    dummy = np.random.randn(*input_details[0]["shape"]).astype(input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], dummy)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])

    print("  Forward pass dummy input berhasil, output shape:", out.shape)

except Exception as e:
    print(f"  [PERINGATAN] Sanity check gagal: {e}")
    print("  File .tflite tetap tersimpan, tapi JANGAN dipakai produksi "
          "sebelum masalah ini diselidiki.")

print("\n[catatan] Quantization INT8 TFLite TIDAK dilakukan di script ini.")
print("[catatan] Itu butuh representative_dataset (sample waveform asli,")
print("[catatan] bukan random noise) supaya kalibrasi range aktivasi benar.")

print("\nDONE")