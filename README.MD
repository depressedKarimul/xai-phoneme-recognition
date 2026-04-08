# 🎙️ XAI Phoneme Recognition Evaluation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![XAI](https://img.shields.io/badge/Focus-Explainable%20AI-green.svg)](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository evaluates Explainable AI (XAI) methods for Automatic Speech Recognition (ASR), specifically phoneme recognition on the TIMIT dataset. It reproduces and analyzes the methodology from the paper: **"Can We Trust Explainable AI Methods on ASR? An Evaluation on Phoneme Recognition"** (ArXiv: [2305.18011](https://arxiv.org/abs/2305.18011)).

---

## 🔥 Key Features

- **XAI Methodology**: Implementation of three LIME variants:
  - **Base LIME**: Standard Local Interpretable Model-agnostic Explanations.
  - **LIME-WS (Window-Segment)**: Localized explanations focusing on neighboring phoneme windows.
  - **LIME-TS (Time-Segment)**: Explanation based on fixed time-duration segments (e.g., 70ms).
- **ASR Pipeline**: 
  - Utilizes **Wav2Vec2-base** fine-tuned on TIMIT.
  - Optimized inference with **ONNX Runtime GPU** (`CUDAExecutionProvider`) for high performance.
  - 61-to-39 Phoneme Mapping (Lee & Hon mapping).
- **Rigorous Evaluation**:
  - **Validity Metrics**: V@1, V@3, and V@5 comparison across gender cohorts.
  - **Statistical Analysis**: Pairwise Tukey HSD and OLS regression to compare XAI method performance.
  - Comparison against **Random Baselines**.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU (RTX 50-series recommended for Blackwell-specific optimizations used in the notebook)
- CUDA-enabled environment

### Installation

```bash
pip install torch torchaudio pandas numpy matplotlib seaborn tqdm transformers optimum onnxruntime-gpu statsmodels
```

### Usage

The core logic and analysis are contained within the Jupyter Notebook:
1. Open `XAI_ASR_Phoneme_Recognition.ipynb`.
2. Follow the cells to load the TIMIT dataset, run ASR inference, and execute the XAI evaluation pipeline.
3. Results to be saved as `.pkl` files in the specified paths for caching and further analysis.

## 📊 Summary of Results

Initial evaluations indicate the following Validity scores (gender-neutral averages):

| Method   | V@1  | V@3  | V@5  |
|----------|------|------|------|
| Base LIME| ~0.45| ~0.70| ~0.77|
| LIME-WS  | ~0.45| ~0.85| ~0.92|
| LIME-TS  | ~0.47| ~0.80| ~0.90|

*LIME-WS and LIME-TS significantly outperform the Base LIME and Random baselines in locating salient audio regions.*

## 📂 Project Structure

```text
├── Dataset/             # TIMIT dataset (Local path defined in notebook)
├── Paper/               # Reference Paper PDF
│   └── 2305.18011v1.pdf
├── XAI_ASR_Phoneme_Recognition.ipynb  # Primary Research Notebook
└── README.md            # Project Documentation
```

## 🛠️ Optimizations

- **ONNX Inference**: To bypass PyTorch/CUDA kernel version mismatches (e.g., on newer Blackwell GPUs), the notebook uses ONNX Runtime as the backend while keeping PyTorch for tensor manipulation.
- **Caching**: Validity analysis results are cached to allow resuming long-running evaluations.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information (if applicable).

---
*Built for research in Explainable AI for Speech Processing.*
