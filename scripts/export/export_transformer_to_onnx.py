"""
WaveformTransformer -> ONNX Export
Mengikuti pola dari script export MobileNetV3 (mobilenetv3_v2 -> ONNX).

Perbedaan penting vs MobileNetV3:
  - Input bukan image (1,3,224,224), tapi waveform 1D (1, C, SEQ_LEN).
  - Arsitektur WaveformTransformer punya banyak hyperparameter
    (d_model, nhead, num_layers, patch_size, dst) yang BERBEDA antara
    teacher dan student -- jadi config diambil dari checkpoint["config"],
    TIDAK di-hardcode, supaya script ini otomatis benar untuk varian mana
    pun yang dipilih lewat VARIANT di bawah.
  - seq_len/num_patches dibuat FIXED (bukan dynamic_axes) karena
    PatchEmbedding1D mensyaratkan seq_len % patch_size == 0 secara
    struktural (reshape, bukan operasi yang aman di-dynamic-kan tanpa
    re-derive num_patches). Hanya batch_size yang dynamic, sama seperti
    contoh MobileNetV3.
  - Checkpoint quantized (student_quantized.pt) TIDAK didukung oleh
    script ini. torch.ao.nn.quantized.dynamic.Linear (hasil
    torch.quantization.quantize_dynamic) tidak punya jalur export ONNX
    yang stabil di torch.onnx (non-dynamo) -- mencoba export ini akan
    gagal atau menghasilkan graph yang salah secara diam-diam. Jika
    butuh INT8 portable, alur yang benar adalah export model FP32 dulu
    (teacher/student-distilled/student-pruned) ke ONNX, baru lakukan
    quantization di sisi ONNX Runtime (onnxruntime.quantization), bukan
    membawa quantized state_dict PyTorch ke ONNX.
"""

import os
import torch
import torch.nn as nn
import onnx

# =====================================================
# CONFIG
# =====================================================

PROJECT_DIR = "/home/indra/eq_team/earthquake-classification"

# Ganti sesuai checkpoint yang ingin diexport. Pilihan yang DIDUKUNG:
#   "teacher"           -> checkpoints/best_transformer_5s.pt
#   "student_distilled"  -> checkpoints/student_distilled.pt
#   "student_pruned"     -> checkpoints/student_pruned.pt
# "student_quantized" SENGAJA tidak ada di sini -- lihat docstring di atas.
VARIANT = "teacher"

# Nama run dir tempat checkpoint berada (samakan dengan output run training).
RUN_DIR_NAME = "2026-06-22_182500_transformer_compression_nogit"

CKPT_FILENAME = {
    "teacher": "best_transformer_5s.pt",
    "student_distilled": "student_distilled.pt",
    "student_pruned": "student_pruned.pt",
}[VARIANT]

CKPT_PATH = os.path.join(
    PROJECT_DIR,
    "results", "runs", RUN_DIR_NAME,
    "checkpoints", CKPT_FILENAME
)

ONNX_DIR = os.path.join(
    PROJECT_DIR,
    "models",
    "onnx"
)

os.makedirs(
    ONNX_DIR,
    exist_ok=True
)

ONNX_PATH = os.path.join(
    ONNX_DIR,
    f"waveform_transformer_{VARIANT}.onnx"
)

OPSET_VERSION = 17  # nn.TransformerEncoderLayer butuh opset >=14 untuk
                     # beberapa op (scaled_dot_product_attention-related);
                     # 17 dipakai supaya aman, beda dari MobileNetV3 (13)
                     # yang tidak punya attention.


# =====================================================
# MODEL DEFINITION (harus identik dgn skrip training)
# =====================================================
# Disalin dari WaveformTransformer training script supaya file export ini
# tidak punya dependency import ke skrip training (yang juga set
# matplotlib backend, load data besar, dst -- tidak relevan untuk export).

import math


class PatchEmbedding1D(nn.Module):
    def __init__(self, in_channels, seq_len, patch_size, d_model):
        super().__init__()
        assert seq_len % patch_size == 0, \
            f"SEQ_LEN ({seq_len}) harus habis dibagi PATCH_SIZE ({patch_size})"
        self.num_patches = seq_len // patch_size
        self.proj = nn.Linear(in_channels * patch_size, d_model)

    def forward(self, x):
        B, C, L = x.shape
        x = x.view(B, C, self.num_patches, -1)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, self.num_patches, -1)
        return self.proj(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class WaveformTransformer(nn.Module):
    def __init__(self, in_channels, seq_len, patch_size, d_model, nhead,
                 num_layers, dim_feedforward, dropout, num_classes):
        super().__init__()
        self.patch_embed = PatchEmbedding1D(in_channels, seq_len, patch_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                              norm=nn.LayerNorm(d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B = x.size(0)
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_enc(tokens)
        out = self.encoder(tokens)
        return self.head(out[:, 0])


# =====================================================
# LOAD CHECKPOINT
# =====================================================

print("=" * 60)
print("WaveformTransformer -> ONNX Export")
print("=" * 60)
print(f"Variant : {VARIANT}")

print("\n[1] Loading checkpoint...")

checkpoint = torch.load(
    CKPT_PATH,
    map_location="cpu",
    weights_only=False
)

classes = checkpoint["classes"]
num_classes = len(classes)

# config disimpan oleh save_checkpoint() di script training sebagai dict
# gabungan run_meta["config"] + extra_config. Untuk teacher, hyperparams
# arsitektur ada langsung di cfg["teacher"]; untuk student, ada di
# cfg["student"] (lihat extra_config={"student": student_config} saat
# distillation/pruning).
cfg = checkpoint["config"]
arch_cfg = cfg["teacher"] if VARIANT == "teacher" else cfg["student"]

print("Checkpoint  :", CKPT_PATH)
print("Classes     :", classes)
print("Num classes :", num_classes)
print("Arch config :", arch_cfg)

# =====================================================
# BUILD MODEL
# =====================================================

print("\n[2] Building WaveformTransformer...")

model = WaveformTransformer(**arch_cfg)

model.load_state_dict(
    checkpoint["state_dict"]
)

model.eval()

# FIX: sama alasannya dengan torch.backends.mha.set_fastpath_enabled(False)
# di skrip training -- fastpath internal nn.TransformerEncoderLayer aktif
# saat model di .eval() dan punya kontrol jalur berbeda dari path training.
# Untuk EXPORT ONNX ini juga harus dimatikan, karena fastpath memakai
# operator native (_native_multi_head_attention) yang tracing/symbolic-nya
# tidak selalu stabil ke ONNX opset standar. Mematikannya tidak mengubah
# bobot atau hasil numerik, hanya memilih implementasi forward yang lebih
# "vanilla" dan lebih predictable saat di-trace torch.onnx.export.
torch.backends.mha.set_fastpath_enabled(False)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("Parameters :", f"{total_params:,}")
print("Seq len    :", arch_cfg["seq_len"], " Patch size:", arch_cfg["patch_size"],
      " Num tokens (+CLS):", arch_cfg["seq_len"] // arch_cfg["patch_size"] + 1)

# =====================================================
# DUMMY INPUT
# =====================================================
# Bukan image (1,3,224,224) seperti MobileNetV3 -- ini waveform 3-channel
# (umumnya Z/N/E) sepanjang seq_len sample. Shape: (batch, channels, seq_len).

dummy_input = torch.randn(
    1,
    arch_cfg["in_channels"],
    arch_cfg["seq_len"],
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
    input_names=["waveform"],
    output_names=["logits"],
    dynamic_axes={
        # Hanya batch_size yang dynamic. seq_len/in_channels FIXED karena
        # PatchEmbedding1D.forward melakukan x.view(B, C, num_patches, -1)
        # dengan num_patches = seq_len // patch_size yang sudah di-hardcode
        # saat __init__ -- mengubah seq_len di runtime akan menghasilkan
        # reshape yang salah/error, bukan didukung secara native.
        "waveform": {0: "batch_size"},
        "logits": {0: "batch_size"}
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
# SANITY CHECK: bandingkan output PyTorch vs ONNX Runtime
# =====================================================
# MobileNetV3 contoh tidak melakukan ini, tapi untuk Transformer ini
# penting ditambahkan: attention/LayerNorm/GELU punya lebih banyak celah
# untuk numerik sedikit berbeda antar opset/backend dibanding CNN biasa.
# Kalau onnxruntime tidak terpasang, langkah ini dilewati dengan pesan
# jelas (tidak menggagalkan export yang sudah berhasil di atas).

print("\n[5] Sanity check PyTorch vs ONNX Runtime...")

try:
    import onnxruntime as ort
    import numpy as np

    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"waveform": dummy_input.numpy()})[0]

    max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"  Max abs diff (logits) PyTorch vs ONNX : {max_abs_diff:.6e}")
    if max_abs_diff > 1e-3:
        print("  [PERINGATAN] Selisih > 1e-3 -- cek ulang sebelum dipakai produksi.")
    else:
        print("  OK -- output PyTorch dan ONNX konsisten.")
except ImportError:
    print("  onnxruntime tidak terpasang -- sanity check dilewati.")
    print("  Install dengan: pip install onnxruntime --break-system-packages")

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

    size_mb = (
        os.path.getsize(path)
        / (1024 * 1024)
    )

    print(
        f" - {f} ({size_mb:.2f} MB)"
    )

print("\nDONE")