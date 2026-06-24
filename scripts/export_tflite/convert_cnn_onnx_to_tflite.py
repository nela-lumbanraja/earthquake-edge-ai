"""
Convert CNN5s ONNX -> TFLite
=============================

Input  : models/cnn_v2/cnn_v2.onnx   (output dari export_cnn_to_onnx.py)
Output : models/cnn_v2/cnn_v2_float32.tflite  (+ _float16.tflite)

Perbedaan penting dari script lama:
  1) onnx2tf SUDAH menghasilkan file .tflite langsung. Tidak perlu lagi
     tf.lite.TFLiteConverter.from_saved_model(). Versi onnx2tf baru bahkan
     tidak menulis SavedModel secara default, jadi langkah lama itu bisa CRASH.
  2) Dipakai flag -kat input agar layout input TETAP NCHW (1,3,224,224),
     sama persis dengan ONNX. Tanpa flag ini, onnx2tf otomatis mengubah input
     ke NHWC (1,224,224,3) -> preprocessing NCHW kamu akan menghasilkan
     prediksi SALAH tanpa error apa pun.
  3) Ada parity check ONNX vs TFLite (di-skip otomatis kalau onnxruntime
     belum terpasang).

Preprocessing inference WAJIB sama dengan training (z-score per-trace global,
interpolate ke 224, repeat ke 224x224), input akhir bentuk (1, 3, 224, 224).
"""

import os
import shutil
import subprocess
import glob

import numpy as np
import tensorflow as tf


# =====================================================
# CONFIG
# =====================================================
PROJECT_DIR = "/home/indra/eq_team/earthquake-classification"
OUT_DIR = os.path.join(PROJECT_DIR, "models", "cnn_v2")

ONNX_PATH = os.path.join(OUT_DIR, "cnn_V2.onnx")

TFLITE_FP32 = os.path.join(OUT_DIR, "cnn_V2_float32.tflite")
TFLITE_FP16 = os.path.join(OUT_DIR, "cnn_V2_float16.tflite")

IMG_SIZE = 224

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("CNN5s ONNX -> TFLite Conversion")
print("=" * 60)

if not os.path.exists(ONNX_PATH):
    raise FileNotFoundError(f"ONNX tidak ditemukan:\n{ONNX_PATH}")
print("\n[1] ONNX file:", ONNX_PATH)


# =====================================================
# 2) ONNX -> TFLITE via onnx2tf (-kat input = pertahankan NCHW)
# =====================================================
print("\n[2] Converting ONNX -> TFLite (onnx2tf, input dipertahankan NCHW)...")

# Bersihkan output lama.
for p in (TFLITE_FP32, TFLITE_FP16):
    if os.path.exists(p):
        os.remove(p)

subprocess.run(
    [
        "onnx2tf",
        "-i", ONNX_PATH,
        "-o", OUT_DIR,
        "-kat", "input",      # keep input shape as-is (NCHW), nama input = "input"
    ],
    check=True,
)

# onnx2tf menamai file <stem>_float32.tflite / <stem>_float16.tflite.
stem = os.path.splitext(os.path.basename(ONNX_PATH))[0]   # "cnn_v2"
gen_fp32 = os.path.join(OUT_DIR, f"{stem}_float32.tflite")
gen_fp16 = os.path.join(OUT_DIR, f"{stem}_float16.tflite")

# Pastikan nama akhir konsisten (kalau stem != cnn_v2).
if gen_fp32 != TFLITE_FP32 and os.path.exists(gen_fp32):
    shutil.move(gen_fp32, TFLITE_FP32)
if gen_fp16 != TFLITE_FP16 and os.path.exists(gen_fp16):
    shutil.move(gen_fp16, TFLITE_FP16)

if not os.path.exists(TFLITE_FP32):
    raise RuntimeError("onnx2tf tidak menghasilkan float32 tflite. Cek log di atas.")


# =====================================================
# 3) CEK LAYOUT INPUT TFLITE
# =====================================================
print("\n[3] Memeriksa input/output TFLite...")
interp = tf.lite.Interpreter(model_path=TFLITE_FP32)
interp.allocate_tensors()
in_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
print("  Input  :", in_det["name"], tuple(in_det["shape"]), in_det["dtype"].__name__)
print("  Output :", out_det["name"], tuple(out_det["shape"]))

if tuple(in_det["shape"]) != (1, 3, IMG_SIZE, IMG_SIZE):
    print("  WARNING: input BUKAN (1,3,224,224). Layout mungkin berubah ke NHWC;")
    print("           sesuaikan preprocessing inference (transpose) atau cek flag -kat.")


# =====================================================
# 4) PARITY CHECK ONNX vs TFLITE (opsional)
# =====================================================
print("\n[4] Parity check ONNX vs TFLite...")
try:
    import onnxruntime as ort

    np.random.seed(0)
    x = np.random.randn(4, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["output"], {"input": x})[0]

    tfl_out = []
    for i in range(x.shape[0]):
        interp.set_tensor(in_det["index"], x[i:i + 1])
        interp.invoke()
        tfl_out.append(interp.get_tensor(out_det["index"])[0])
    tfl_out = np.array(tfl_out)

    max_diff = float(np.max(np.abs(onnx_out - tfl_out)))
    argmax_ok = bool(np.array_equal(onnx_out.argmax(1), tfl_out.argmax(1)))
    print(f"  Max abs diff : {max_diff:.3e}")
    print(f"  Argmax cocok : {argmax_ok}")
    if not argmax_ok or max_diff > 1e-3:
        print("  WARNING: TFLite TIDAK setara ONNX. Jangan dipakai sebelum diperiksa.")
    else:
        print("  PASS - TFLite setara dengan ONNX.")
except ImportError:
    print("  (di-skip) onnxruntime belum terpasang -> parity tidak diverifikasi.")


# =====================================================
# 5) INFO FILE
# =====================================================
print("\n" + "=" * 60)
print("Conversion selesai")
for p in (TFLITE_FP32, TFLITE_FP16):
    if os.path.exists(p):
        mb = os.path.getsize(p) / (1024 * 1024)
        print(f"  {os.path.basename(p):28s} {mb:6.2f} MB")
print("  Folder :", OUT_DIR)
print("DONE")