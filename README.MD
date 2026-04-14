# XAI Phoneme Recognition

Reproduction and improvement of **"Can We Trust Explainable AI Methods on ASR? An Evaluation on Phoneme Recognition"** using modern deep learning tools.

## Overview
This project evaluates three LIME-based XAI methods (Base LIME, LIME-WS, LIME-TS) for explaining phoneme recognition on the TIMIT dataset, replacing the original Kaldi backend with Wav2Vec2.

## Key Improvements over Paper
- **Base LIME V@3: 0.91 vs 0.62** (+47%) 🔥
- **LIME-WS V@3: 0.92 vs 0.76** (+21%) 🔥
- **LIME-TS V@5: 0.97 = 0.97** (equal)
- **New finding:** 90ms segment duration outperforms paper's 70ms

## Setup
- OS: Windows 11
- GPU: NVIDIA RTX 5050 (Blackwell)
- Python: 3.10.20
- PyTorch: 2.5.1+cu121
- ONNX Runtime GPU for Blackwell compatibility

## Installation
```bash
pip install transformers torchaudio scikit-learn scipy matplotlib seaborn pandas numpy tqdm statsmodels optimum onnxruntime-gpu
```

## Dataset
TIMIT dataset — place at `D:\xai-phoneme-recognition\TIMIT dataset\data\`

## Model
`patrickvonplaten/wav2vec2-base-timit-fine-tuned` via HuggingFace

## Results (n_samples=500, dur_ms=90ms)

| Method | V@1 | V@3 | V@5 |
|--------|-----|-----|-----|
| Base LIME | 0.50 | 0.91 | 0.95 |
| LIME-WS | 0.48 | 0.92 | 0.96 |
| LIME-TS | 0.59 | 0.92 | 0.97 |
| Random | 0.02 | 0.07 | 0.12 |

## Reference
Wu et al., "Can We Trust Explainable AI Methods on ASR? An Evaluation on Phoneme Recognition", 2023