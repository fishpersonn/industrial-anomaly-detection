# Industrial Anomaly Detection with Anomalib & OpenVINO

Industrial visual anomaly detection using PatchCore, with CPU inference benchmarking between PyTorch and OpenVINO.

## Project Overview

- **Task:** Industrial visual anomaly detection
- **Dataset:** MVTec AD (Bottle)
- **Model:** PatchCore
- **Frameworks:** Anomalib, PyTorch, OpenVINO
- **Hardware:** CPU

## Model Evaluation

| Metric | Score |
|---|---:|
| Image AUROC | 1.000 |
| Image F1-score | 0.992 |
| Pixel AUROC | 0.986 |
| Pixel F1-score | 0.726 |

## Preliminary CPU Benchmark

| Runtime | Latency (ms/batch) | Throughput (images/sec) |
|---|---:|---:|
| PyTorch | 989.74 | 2.02 |
| OpenVINO | 556.03 | 3.60 |

**Observed speed ratio: 1.78×**

Test configuration:
- Batch size: 2
- Input shape: 2 × 3 × 256 × 256
- Warm-up iterations: 10
- Benchmark iterations: 30

These are preliminary results using synthetic input. CPU thread configurations and inference pipelines have not yet been fully standardized, so the result should not be interpreted as a controlled deployment speedup.

## Installation

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Benchmark

```bash
python benchmark.py
```

The benchmark requires a trained PatchCore checkpoint and an exported OpenVINO model at the paths configured in `benchmark.py`.

## Future Work

- Validate PyTorch and OpenVINO prediction consistency
- Standardize CPU benchmarking conditions
- Evaluate inference with real industrial images
- Improve deployment reproducibility