import os
import json
import csv
import time
import psutil
import numpy as np
import torch
from datetime import datetime

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "/home/indra/eq_team/earthquake-classification/results/runs/20260617_201329_CNN5s/checkpoints/best_cnn_dynamic_quantized.pt"
RESULT_DIR = "results_quantized"

NUM_WARMUP = 10
NUM_RUNS = 100

os.makedirs(RESULT_DIR, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading quantized model...")

ckpt = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

model = ckpt["model"]
classes = ckpt["classes"]

model.eval()

process = psutil.Process(os.getpid())
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

print("Classes:", classes)
print("Model size:", round(model_size_mb, 4), "MB")

# =====================================================
# DUMMY INPUT
# =====================================================

x = torch.randn(1, 3, 224, 224)

# =====================================================
# WARMUP
# =====================================================

print(f"Warmup ({NUM_WARMUP}x)...")

with torch.no_grad():
    for _ in range(NUM_WARMUP):
        _ = model(x)

# =====================================================
# BENCHMARK
# =====================================================

print(f"Benchmark ({NUM_RUNS}x)...")

latencies = []
ram_usages = []
cpu_usages = []

wall_start = time.perf_counter()

with torch.no_grad():

    for _ in range(NUM_RUNS):

        psutil.cpu_percent(interval=None)

        start = time.perf_counter()

        output = model(x)

        end = time.perf_counter()

        ram_after = process.memory_info().rss / (1024 * 1024)
        cpu_after = psutil.cpu_percent(interval=None)

        latencies.append((end - start) * 1000)
        ram_usages.append(ram_after)
        cpu_usages.append(cpu_after)

wall_elapsed = time.perf_counter() - wall_start
fps = NUM_RUNS / wall_elapsed if wall_elapsed > 0 else 0

# =====================================================
# SUMMARY
# =====================================================

summary = {
    "timestamp": datetime.now().isoformat(),
    "device": "HPC CPU",
    "runtime": "PyTorch Dynamic Quantization",
    "model": "MobileNetV3 Dynamic Quantized",
    "optimization_status": "Setelah Optimasi",
    "model_path": MODEL_PATH,
    "model_size_mb": round(float(model_size_mb), 4),

    "latency_ms": {
        "mean": round(float(np.mean(latencies)), 4),
        "median": round(float(np.median(latencies)), 4),
        "min": round(float(np.min(latencies)), 4),
        "max": round(float(np.max(latencies)), 4),
        "std": round(float(np.std(latencies)), 4),
        "p95": round(float(np.percentile(latencies, 95)), 4),
        "p99": round(float(np.percentile(latencies, 99)), 4),
    },

    "throughput_fps": round(float(fps), 4),

    "memory_mb": {
        "mean_ram": round(float(np.mean(ram_usages)), 4),
        "peak_ram": round(float(np.max(ram_usages)), 4),
    },

    "cpu_percent": {
        "mean": round(float(np.mean(cpu_usages)), 4),
        "max": round(float(np.max(cpu_usages)), 4),
    }
}

# =====================================================
# SAVE
# =====================================================

json_path = os.path.join(
    RESULT_DIR,
    "mobilenetv3_dynamic_quantized_hpc.json"
)

csv_path = os.path.join(
    RESULT_DIR,
    "mobilenetv3_dynamic_quantized_hpc.csv"
)

with open(json_path, "w") as f:
    json.dump(summary, f, indent=4)

with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["metric", "value"])

    writer.writerow(["model", summary["model"]])
    writer.writerow(["optimization_status", summary["optimization_status"]])

    writer.writerow(["model_size_mb", summary["model_size_mb"]])

    writer.writerow(["latency_mean_ms", summary["latency_ms"]["mean"]])
    writer.writerow(["latency_p95_ms", summary["latency_ms"]["p95"]])

    writer.writerow(["throughput_fps", summary["throughput_fps"]])

    writer.writerow(["mean_ram_mb", summary["memory_mb"]["mean_ram"]])
    writer.writerow(["peak_ram_mb", summary["memory_mb"]["peak_ram"]])

    writer.writerow(["mean_cpu_percent", summary["cpu_percent"]["mean"]])

print("\nBenchmark selesai")
print("Model        :", summary["model"])
print("Size MB      :", summary["model_size_mb"])
print("Latency Mean :", summary["latency_ms"]["mean"])
print("Latency P95  :", summary["latency_ms"]["p95"])
print("FPS          :", summary["throughput_fps"])
print("Peak RAM     :", summary["memory_mb"]["peak_ram"])
print("Mean CPU     :", summary["cpu_percent"]["mean"])

print("\nSaved JSON:", json_path)
print("Saved CSV :", csv_path)