# XAI Phoneme Recognition

![Banner](assets/banner.png)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-Wav2Vec2--Base-green.svg)](https://huggingface.co/patrickvonplaten/wav2vec2-base-timit-fine-tuned)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproduction and extension of:
**"Can We Trust Explainable AI Methods on ASR? An Evaluation on Phoneme Recognition"** (Wu et al., Interspeech 2023)

> Goal: stress-test LIME-style explanations for phoneme-level ASR and push beyond the original setup with stronger baselines, new variants, and deeper analysis.

## Why This Project Is Different

- Replaces legacy Kaldi stack with Hugging Face Wav2Vec2 (CTC) (using `wav2vec2-large-960h-lv60-self` for extended datasets)
- Expands evaluation beyond TIMIT to include **Mozilla Common Voice** and **LibriSpeech (dev-clean)**
- Implements robust forced alignment using `torchaudio.pipelines.MMS_FA`
- Tuned for modern hardware via ONNX Runtime workflow
- Adds new method: `LIME-TS-Overlap` (50% overlap)
- Adds SHAP comparison against LIME variants
- Adds gender and phoneme-category (vowel vs consonant) analysis
- Reports duration sensitivity (50/70/90 ms)

## Results At A Glance

Evaluation setting: `n_samples=500`, `dur_ms=90`, `TIMIT speakers=630`

| Method | Paper V@1 | Ours V@1 | Paper V@3 | Ours V@3 | Paper V@5 | Ours V@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Base LIME | 0.40 | 0.50 | 0.62 | 0.91 | 0.67 | 0.95 |
| LIME-WS | 0.49 | 0.48 | 0.76 | 0.92 | 0.83 | 0.96 |
| LIME-TS | 0.86 | 0.59 | 0.96 | 0.92 | 0.97 | 0.97 |
| Random | 0.03 | 0.02 | 0.09 | 0.07 | 0.15 | 0.12 |

### Additional Contributions

| Method | V@1 | V@3 | V@5 | Notes |
| :--- | :---: | :---: | :---: | :--- |
| LIME-TS-Overlap | 0.74 | 0.98 | 1.00 | 50% segment overlap |
| SHAP | - | 10.0% | - | Baseline comparison |
| LIME-TS (90 ms) | - | 92.5% | - | Strongest non-overlap setting |

### Duration Sensitivity

| Segment Duration | LIME-TS Validity@3 |
| :--- | :---: |
| 50 ms | 0.76 |
| 70 ms (paper setting) | 0.85 |
| 90 ms (this work) | 0.91 |

## Quickstart

### 1) Install Dependencies

```bash
pip install transformers torchaudio scikit-learn scipy matplotlib seaborn pandas numpy tqdm statsmodels optimum onnxruntime-gpu shap
```

### 2) Prepare Dataset

- Download datasets: TIMIT, Mozilla Common Voice, and/or LibriSpeech
- Set dataset root paths in the respective notebook configuration sections
- Ensure audio is available and correctly formatted (e.g., `.wav`, `.mp3`, or `.flac` for 16 kHz processing pipeline)

### 3) Run Notebook

Open and run the notebook for your target dataset:
- `XAI_ASR_Phoneme_Recognition.ipynb` (TIMIT Baseline)
- `XAI_ASR_Phoneme_Recognition_Updated.ipynb` (TIMIT with SHAP & Overlap)
- `xai_asr_evaluation_for_CommonVoice.ipynb` (Common Voice Evaluation)
- `xai_asr_evaluation_for_librispeech.ipynb` (LibriSpeech Evaluation)

## Method Overview

1. Preprocess audio and apply forced alignment (using Lee & Hon `61 -> 39` mapping for TIMIT, or `torchaudio` MMS Forced Aligner for other datasets)
2. Run Wav2Vec2 inference (via ONNX Runtime backend or PyTorch)
3. Generate perturbations using segment masking (Base LIME, LIME-WS, LIME-TS)
4. Compute validity metrics (`V@1`, `V@3`, `V@5`)
5. Perform statistical tests (ANOVA, Tukey HSD, Wilcoxon) and generate qualitative heatmaps

## Repository Map

```text
assets/                                      # Visual assets
Paper/                                       # Reference paper(s)
XAI_ASR_Phoneme_Recognition.ipynb            # Original TIMIT research notebook
XAI_ASR_Phoneme_Recognition_Updated.ipynb    # Updated TIMIT notebook (SHAP, Overlap)
xai_asr_evaluation_for_CommonVoice.ipynb     # Common Voice evaluation notebook
xai_asr_evaluation_for_librispeech.ipynb     # LibriSpeech evaluation notebook
README.MD
LICENSE
```

## Notebook Map

| Cell | Focus |
| :--- | :--- |
| 1 | Package installation |
| 2 | Imports and config |
| 3 | TIMIT loading and phoneme mapping |
| 4 | Wav2Vec2 ASR inference |
| 5 | Base LIME |
| 6 | LIME-WS |
| 7 | LIME-TS |
| 8 | Validity evaluation |
| 9 | Paper-style results table |
| 10 | Gender analysis |
| 11 | Duration sensitivity |
| 12 | Waveform heatmap |
| 13 | Model saving |
| 14 | SHAP vs LIME |
| 15 | LIME-TS-Overlap |
| 16 | Vowel vs consonant analysis |
| 17 | Attention visualization |
| 18 | Combined summary table |

## Reference

Wu, X., Bell, P., and Rajan, A. "Can We Trust Explainable AI Methods on ASR? An Evaluation on Phoneme Recognition." Interspeech, 2023.

## License

MIT License. See [LICENSE](LICENSE).
