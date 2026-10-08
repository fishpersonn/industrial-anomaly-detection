 <div align="center">

# 🔍 Industrial Anomaly Detection

### Industrial Visual Inspection with Anomalib, PatchCore & OpenVINO

**An industrial anomaly detection experiment featuring visual defect localization and CPU inference benchmarking.**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-orange?logo=pytorch)
![Anomalib](https://img.shields.io/badge/Anomalib-2.7.0-green)
![OpenVINO](https://img.shields.io/badge/OpenVINO-2026.4.1-purple)
![Status](https://img.shields.io/badge/Status-Experimental-yellow)

**[Overview](#-overview) • [Results](#-model-evaluation) • [Benchmark](#-inference-benchmark) • [Installation](#-installation) • [Roadmap](#-roadmap)**

</div>

---

## 👋 Overview

This project explores **industrial visual anomaly detection** using Anomalib and PatchCore, with an emphasis on CPU inference performance and deployment optimization using Intel OpenVINO.

Industrial quality inspection often faces challenges such as limited defect samples, inconsistent manual inspection, and computational constraints on edge devices.

The project investigates two questions:

1. Can PatchCore effectively detect and localize defects using primarily normal training images?
2. How does OpenVINO inference performance compare with PyTorch on CPU?

### Project Highlights

- Industrial anomaly detection using PatchCore
- Evaluation on the MVTec AD Bottle dataset
- Pixel-level anomaly visualization
- Model export from PyTorch to OpenVINO
- CPU inference benchmarking
- Automated benchmark results exported to CSV

## 🏗️ System Architecture

```text
       Industrial Images
               |
               v
        MVTec AD Dataset
               |
               v
      Anomalib + PatchCore
               |
       +-------+-------+
       |               |
       v               v
   Evaluation      Model Export
       |               |
       v               v
  AUROC / F1         OpenVINO
                       |
                       v
                CPU Inference
                       |
                       v
              Benchmark Results
                       |
                       v
                 CSV Reports
```

## 🧠 Model & Dataset

| Component | Description |
|---|---|
| Model | PatchCore |
| Framework | Anomalib 2.7.0 |
| Deep Learning | PyTorch 2.14.1 (CPU) |
| Deployment Runtime | OpenVINO 2026.4.1 |
| Dataset | MVTec AD |
| Category | Bottle |
| Input Shape | 2 × 3 × 256 × 256 (exported model) |

PatchCore uses pretrained visual representations and a memory bank of normal image features to identify anomalous regions.

The MVTec AD dataset contains industrial inspection images with normal and defective samples and pixel-level ground truth annotations.

## 🖼️ Anomaly Detection Visualization

The following visualization components were generated during model evaluation:

| Visualization | Description |
|---|---|
| Original Image | Industrial bottle inspection image |
| Ground Truth Mask | Annotated defect region |
| Anomaly Map | Model-generated anomaly scores |
| Predicted Mask | Model-predicted defect region |

> A sample visualization will be added under `assets/` after preparing a GitHub-friendly image.

## 📊 Model Evaluation

PatchCore was evaluated on the MVTec AD Bottle dataset.

| Metric | Score |
|---|---:|
| Image AUROC | **1.0000** |
| Image F1-score | **0.9920** |
| Pixel AUROC | **0.9856** |
| Pixel F1-score | **0.7264** |

The model demonstrated strong image-level anomaly discrimination. Pixel-level localization results indicate room for improvement in accurately outlining defect boundaries.

## ⚡ Inference Benchmark

### PyTorch vs OpenVINO (CPU)

| Metric | PyTorch | OpenVINO |
|---|---:|---:|
| Average Batch Latency | 989.74 ms | **556.03 ms** |
| Throughput | 2.02 images/sec | **3.60 images/sec** |
| Batch Size | 2 | 2 |
| Input Resolution | 256 × 256 | 256 × 256 |
| Device | CPU | CPU |

### Observed Speed Ratio: 1.78×

Under the measured test configuration, OpenVINO achieved approximately **1.78× the inference throughput of PyTorch**.

### Benchmark Configuration

- Warm-up iterations: 10
- Measurement iterations: 30
- Input: Synthetic random tensors
- Device: CPU
- PyTorch intra-op threads: 4
- OpenVINO: Default CPU configuration

**Limitations:** These results are preliminary. CPU thread settings and exact model preprocessing/postprocessing equivalence have not been fully validated. The speed ratio should not be interpreted as a controlled production-deployment speedup.

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/industrial-anomaly-detection.git
cd industrial-anomaly-detection
```

Replace `YOUR_USERNAME` with your GitHub username after creating the repository.

### 2. Create Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## 🚀 Usage

### 1. Train PatchCore

```bash
anomalib train --model Patchcore --data anomalib.data.MVTecAD --data.category bottle
```

### 2. Export to OpenVINO

Replace the checkpoint path if your experiment directory differs:

```powershell
anomalib export --model Patchcore --export_type openvino --ckpt_path "results/Patchcore/MVTecAD/bottle/v0/weights/lightning/model.ckpt" --input_size "[256,256]" --compression_type FP16
```

### 3. Run Benchmark

```bash
python benchmark.py
```

The benchmark uses checkpoint and OpenVINO model paths configured in `benchmark.py`. Both model files must exist before running it.

Benchmark results are automatically saved to:

```text
reports/benchmark.csv
```

## 📁 Project Structure

```text
industrial-anomaly-detection/
├── benchmark.py
├── requirements.txt
├── README.md
├── .gitignore
├── reports/
│   ├── baseline.csv
│   └── benchmark.csv
├── datasets/         # Local dataset (Git ignored)
└── results/          # Model artifacts (Git ignored)
```

## 🗺️ Roadmap

- [x] Set up Anomalib and OpenVINO
- [x] Train PatchCore using MVTec AD Bottle
- [x] Evaluate image-level and pixel-level metrics
- [x] Generate defect heatmaps
- [x] Export PatchCore to OpenVINO
- [x] Run preliminary CPU inference benchmarking
- [x] Automate benchmark CSV generation
- [ ] Validate PyTorch and OpenVINO output equivalence
- [ ] Standardize CPU benchmarking configurations
- [ ] Benchmark using real industrial images
- [ ] Add reproducible visualization assets
- [ ] Build an interactive inference demo
- [ ] Explore real-world edge deployment

## 📚 References

- [Anomalib — Official Repository](https://github.com/open-edge-platform/anomalib)
- [OpenVINO — Official Documentation](https://docs.openvino.ai/)
- [MVTec AD — Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [PatchCore — Original Paper](https://arxiv.org/abs/2106.08265)

## ⚖️ License & Dataset Notice

This project uses third-party open-source frameworks. Their respective licenses remain applicable.

MVTec AD is distributed under its own terms and is not included in this repository. Review the dataset's usage restrictions before redistribution or commercial use.

---

<div align="center">

**Built with Python · PyTorch · Anomalib · OpenVINO**

*Exploring industrial AI, anomaly detection, and efficient edge inference.*

</div>
