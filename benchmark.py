
import csv
import time
from pathlib import Path

import numpy as np
import torch
import openvino as ov
from anomalib.models import Patchcore

# ===== 1. 實驗設定 =====

CKPT_PATH = "results/Patchcore/MVTecAD/bottle/v0/weights/lightning/model.ckpt"
OV_PATH = "results/weights/openvino/model.xml"

WARMUP = 10
RUNS = 30

torch.set_num_threads(4)

# ===== 2. 載入 OpenVINO =====

core = ov.Core()
ov_model = core.compile_model(OV_PATH, "CPU")

input_shape = list(ov_model.input(0).shape)
batch_size = input_shape[0]

# 兩個模型使用相同的測試資料
rng = np.random.default_rng(42)
input_data = rng.random(input_shape).astype(np.float32)

# ===== 3. 載入 PyTorch 模型 =====

torch_model = Patchcore.load_from_checkpoint(
    CKPT_PATH,
    map_location="cpu",
    weights_only=True,
)
torch_model.eval()

pt_model = torch_model.model
pt_model.eval()

torch_input = torch.from_numpy(input_data.copy())

# ===== 4. 定義測速函式 =====

def benchmark(name, inference_fn):
    print(f"\nTesting {name}...")

    # 預熱
    for _ in range(WARMUP):
        inference_fn()

    times = []

    # 正式測速
    for _ in range(RUNS):
        start = time.perf_counter()
        inference_fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_latency = float(np.mean(times) * 1000)
    throughput = batch_size / float(np.mean(times))

    print(f"Latency: {avg_latency:.2f} ms/batch")
    print(f"Throughput: {throughput:.2f} images/sec")

    return {
        "runtime": name,
        "batch_size": batch_size,
        "input_size": f"{input_shape[2]}x{input_shape[3]}",
        "latency_ms": round(avg_latency, 2),
        "throughput_fps": round(throughput, 2),
    }


# ===== 5. 執行 Benchmark =====

with torch.inference_mode():
    pt_result = benchmark(
        "PyTorch",
        lambda: pt_model(torch_input)
    )

ov_result = benchmark(
    "OpenVINO",
    lambda: ov_model(input_data)
)

# ===== 6. 自動儲存 CSV =====

Path("reports").mkdir(exist_ok=True)

with open(
    "reports/benchmark.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=pt_result.keys())
    writer.writeheader()
    writer.writerows([pt_result, ov_result])

# ===== 7. 比較效能 =====

speedup = (
    pt_result["latency_ms"] /
    ov_result["latency_ms"]
)

print("\n===== Benchmark Result =====")
print(f"PyTorch: {pt_result['throughput_fps']} FPS")
print(f"OpenVINO: {ov_result['throughput_fps']} FPS")
print(f"OpenVINO speed ratio: {speedup:.2f}x")
print("Saved: reports/benchmark.csv")
