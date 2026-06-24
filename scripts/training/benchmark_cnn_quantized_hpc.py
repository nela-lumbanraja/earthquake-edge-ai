import os
import json
import csv
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

MODEL_PATH = "/home/indra/eq_team/earthquake-classification/results/runs/20260617_201329_CNN5s/checkpoints/best_cnn_5s_dynamic_quantized.pt"
RESULT_DIR = "results_cnn_quantized"

NUM_WARMUP = 10
NUM_RUNS = 100

os.makedirs(RESULT_DIR, exist_ok=True)


class CNN5s(nn.Module):
    def __init__(self, num_classes):
        super(CNN5s, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

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
        x = self.features(x)
        x = self.classifier(x)
        return x


ckpt = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

classes = ckpt["classes"]
num_classes = ckpt["num_classes"]

model = CNN5s(num_classes)

model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

process = psutil.Process(os.getpid())
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

x = torch.randn(1, 3, 224, 224)

print("=" * 60)
print("CNN Dynamic Quantized Benchmark")
print("Classes:", classes)
print("Model size:", round(model_size_mb, 4), "MB")
print("=" * 60)

with torch.no_grad():
    for _ in range(NUM_WARMUP):
        _ = model(x)

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

summary = {
    "timestamp": datetime.now().isoformat(),
    "device": "HPC CPU",
    "runtime": "PyTorch Dynamic Quantization",
    "model": "CNN Dynamic Quantized",
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

json_path = os.path.join(RESULT_DIR, "cnn_dynamic_quantized_hpc.json")
csv_path = os.path.join(RESULT_DIR, "cnn_dynamic_quantized_hpc.csv")

with open(json_path, "w") as f:
    json.dump(summary, f, indent=4)

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerow(["model", summary["model"]])
    writer.writerow(["optimization_status", summary["optimization_status"]])
    writer.writerow(["runtime", summary["runtime"]])
    writer.writerow(["device", summary["device"]])
    writer.writerow(["model_size_mb", summary["model_size_mb"]])
    writer.writerow(["latency_mean_ms", summary["latency_ms"]["mean"]])
    writer.writerow(["latency_p95_ms", summary["latency_ms"]["p95"]])
    writer.writerow(["throughput_fps", summary["throughput_fps"]])
    writer.writerow(["mean_ram_mb", summary["memory_mb"]["mean_ram"]])
    writer.writerow(["peak_ram_mb", summary["memory_mb"]["peak_ram"]])
    writer.writerow(["mean_cpu_percent", summary["cpu_percent"]["mean"]])
    writer.writerow(["max_cpu_percent", summary["cpu_percent"]["max"]])

print("\nBenchmark selesai")
print("Model        :", summary["model"])
print("Size MB      :", summary["model_size_mb"])
print("Latency mean :", summary["latency_ms"]["mean"], "ms")
print("Latency p95  :", summary["latency_ms"]["p95"], "ms")
print("FPS          :", summary["throughput_fps"])
print("Peak RAM     :", summary["memory_mb"]["peak_ram"], "MB")
print("Mean CPU     :", summary["cpu_percent"]["mean"], "%")
print("Saved JSON   :", json_path)
print("Saved CSV    :", csv_path)