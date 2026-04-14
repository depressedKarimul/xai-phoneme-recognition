# 🎙️ XAI Phoneme Recognition 🧠

![Banner](assets/banner.png)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-Wav2Vec2--Base-green.svg)](https://huggingface.co/patrickvonplaten/wav2vec2-base-timit-fine-tuned)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An academically rigorous reproduction and enhancement of the research paper:  
**"Can We Trust Explainable AI Methods on ASR? An Evaluation on Phoneme Recognition"** (Wu et al., 2023).

---

## 🚀 Overview

This project provides a comprehensive evaluation of Local Interpretable Model-agnostic Explanations (LIME) applied to Automatic Speech Recognition (ASR). We analyze the validity of three LIME variants—**Base LIME**, **LIME-WS** (Window-Segment), and **LIME-TS** (Time-Segment)—using the **TIMIT dataset**.

### 🌟 Key Enhancements over the Original Paper
*   **Modern Backend**: Replaced the legacy **Kaldi** toolbox with **HuggingFace's Wav2Vec2** architecture for state-of-the-art CTC-based recognition.
*   **Hardware Optimization**: Fully optimized for **NVIDIA RTX 5050 (Blackwell)** using **ONNX Runtime** (`CUDAExecutionProvider`) to overcome legacy CUDA kernel incompatibilities.
*   **Performance Leap**: Achieved a **+47%** improvement in Base LIME Validity@3 (0.91 vs 0.62) and a **+21%** improvement in LIME-WS Validity@3.
*   **Optimal Hyperparameters**: Identified that a **90ms** segment duration significantly outperforms the paper's original 70ms recommendation.

---

## 📊 Evaluation Results

*Evaluated with 500 samples, 90ms segment duration, and 300ms window size.*

| Methodology | V@1 | V@3 | V@5 | vs Paper (V@3) |
| :--- | :---: | :---: | :---: | :---: |
| **Base LIME** | 0.50 | 0.91 | 0.95 | +47% 🔥 |
| **LIME-WS** | 0.48 | 0.92 | 0.96 | +21% 🔥 |
| **LIME-TS** | 0.59 | 0.92 | 0.97 | Equal |
| **Random Baseline** | 0.02 | 0.07 | 0.12 | — |

### 🚻 Gender-Based Validity Analysis

| Gender | Method | V@1 | V@3 | V@5 |
| :--- | :--- | :---: | :---: | :---: |
| **Female** | LIME-TS | 0.584 | 0.922 | 0.964 |
| **Male** | LIME-TS | 0.592 | 0.926 | 0.971 |

---

## 🛠️ Project Configuration

### Requirements
- **OS**: Windows 11
- **GPU**: NVIDIA RTX 5000 Series (Blackwell Architecture)
- **Python**: 3.10.x
- **Inference Stack**: ONNX Runtime GPU (Optimum)

### Installation
```bash
pip install transformers torchaudio scikit-learn scipy matplotlib seaborn pandas numpy tqdm statsmodels optimum onnxruntime-gpu
```

---

## 📂 Repository Structure

```
├── assets/                     # Media and banners
├── Paper/                      # Reference research papers
├── TIMIT dataset/              # Dataset directory (symlinked or local)
├── XAI_ASR_Phoneme_Recognition.ipynb  # Primary Research Notebook
└── README.md                   # Project Documentation
```

---

## 📖 Methodology

The project implements a "black-box" evaluation pipeline:
1.  **Preprocessing**: Standardizing TIMIT audio to 16kHz and applying Lee & Hon 61-to-39 phoneme mapping.
2.  **Perturbation**: Systematic masking of audio segments to generate local surrogate datasets.
3.  **Validity Metrics**: Calculating the probability mass shifts to determine if the XAI method correctly identifies the influential frame.
4.  **Statistical Analysis**: Rigorous significance testing using Tukey's HSD to compare XAI method performance.

---

## 🔗 References

> Wu, J., et al. "Can We Trust Explainable AI Methods on ASR? An Evaluation on Phoneme Recognition." *Proceedings of Interspeech*, 2023.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.