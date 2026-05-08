#!/usr/bin/env python
# coding: utf-8

# # Phoneme Recognition XAI Evaluation (LIME/LIME-WS/LIME-TS)
# This notebook evaluates Explainable AI (XAI) methods for Automatic Speech Recognition (ASR), specifically phoneme recognition on the TIMIT dataset. We reproduce and analyze the methodology from:
# **"Can We Trust Explainable AI Methods on ASR? An Evaluation on Phoneme Recognition"**.
# 
# ### Critical System Configuration:
# - **RTX 5050 (Blackwell)**: Utilizing **ONNX Runtime GPU** (`CUDAExecutionProvider`) for inference to bypass PyTorch CUDA kernel compatibility issues.
# - **Strict CPU Tensors**: All PyTorch tensors remain on CPU; ONNX Runtime manages GPU acceleration internally.
# 

# In[1]:


import sys
# Install missing critical packages for ONNX/XAI pipeline
get_ipython().system('{sys.executable} -m pip install statsmodels optimum onnxruntime-gpu')


# ## 2. Imports and Environment Verification

# In[2]:


import torch
import torchaudio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import os, glob, pickle, warnings, sys
from transformers import Wav2Vec2Processor
from optimum.onnxruntime import ORTModelForCTC
from sklearn.linear_model import Ridge
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings('ignore')

# Workspace Paths
DATA_PATH = r"D:\Allah vorosha\TIMIT dataset\data"
SAVE_PATH = r"D:\Allah vorosha"

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")


# ## 3. TIMIT Dataset Loading (61-to-39 Phoneme Mapping)

# In[3]:


# Lee and Hon 61-to-39 Mapping
map39 = {
    'iy':'iy', 'ih':'ih', 'eh':'eh', 'ae':'ae', 'ix':'ih', 'ax':'ah', 'ah':'ah', 'ax-h':'ah', 'uw':'uw', 'ux':'uw',
    'uh':'uh', 'ao':'aa', 'aa':'aa', 'ey':'ey', 'ay':'ay', 'oy':'oy', 'aw':'aw', 'ow':'ow', 'er':'er', 'axr':'er',
    'l':'l', 'el':'l', 'r':'r', 'w':'w', 'y':'y', 'm':'m', 'em':'m', 'n':'n', 'en':'n', 'nx':'n', 'ng':'ng', 'eng':'ng',
    'ch':'ch', 'jh':'jh', 'dh':'dh', 'b':'b', 'd':'d', 'dx':'d', 'g':'g', 'p':'p', 't':'t', 'k':'k', 'z':'z', 'zh':'sh',
    'v':'v', 'f':'f', 'th':'th', 's':'s', 'sh':'sh', 'hh':'hh', 'hv':'hh', 'pcl':'sil', 'tcl':'sil', 'kcl':'sil',
    'qcl':'sil', 'bcl':'sil', 'dcl':'sil', 'gcl':'sil', 'h#':'sil', '#h':'sil', 'pau':'sil', 'epi':'sil', 'q':'sil'
}

def load_sa1_dataset():
    data = []
    # All speakers SA1.WAV files in TRAIN and TEST folders
    pattern = os.path.join(DATA_PATH, "*", "**", "*SA1.WAV")
    files = glob.glob(pattern, recursive=True)

    for f in tqdm(files, desc="Parsing TIMIT SA1"):
        try:
            path_segments = f.split(os.sep)
            speaker_dir = path_segments[-2]
            gender = 'M' if speaker_dir.startswith('M') else 'F'

            # Load phn
            phn_file = f.replace(".WAV", ".PHN")
            phns = []
            if os.path.exists(phn_file):
                with open(phn_file, 'r') as pf:
                    for line in pf:
                        start, end, p = line.strip().split()
                        phns.append({'start': int(start), 'end': int(end), 'phoneme': map39.get(p, p)})

            data.append({'path': f, 'speaker': speaker_dir, 'gender': gender, 'phns': phns})
        except Exception as e:
            print(f"Error parsing {f}: {e}")

    return data

dataset_info = load_sa1_dataset()
print(f"Total SA1 samples loaded: {len(dataset_info)}")


# ## 4. ONNX Model Inference (RTX 5050 Optimized)

# In[4]:


import onnxruntime as ort
from optimum.onnxruntime import ORTModelForCTC
from transformers import Wav2Vec2Processor

# ── Provider Selection ──────────────────────────────────────────
available_providers = ort.get_available_providers()
print("Available ONNX providers:", available_providers)

if "CUDAExecutionProvider" in available_providers:
    provider = "CUDAExecutionProvider"
    print("GPU (CUDA) found — using CUDAExecutionProvider")
else:
    provider = "CPUExecutionProvider"
    print("CUDA not available — falling back to CPUExecutionProvider")

# ── Model Loading ───────────────────────────────────────────────
model_name = "patrickvonplaten/wav2vec2-base-timit-fine-tuned"
processor = Wav2Vec2Processor.from_pretrained(model_name)

ort_model = ORTModelForCTC.from_pretrained(
    model_name,
    export=True,
    provider=provider,
    use_io_binding=False
)

print(f"\n Model loaded on: {provider}")

# ── Inference Function ──────────────────────────────────────────
def get_predictions(wav_path):
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)

    inputs = processor(wav.squeeze(), sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = ort_model(**inputs).logits

    probs = torch.softmax(
        logits if isinstance(logits, torch.Tensor) else torch.tensor(logits),
        dim=-1
    ).squeeze(0)
    ids = torch.argmax(
        logits if isinstance(logits, torch.Tensor) else torch.tensor(logits),
        dim=-1
    ).squeeze(0)
    tokens = processor.tokenizer.convert_ids_to_tokens(ids)

    return wav, probs, tokens

# ── Cache Check ─────────────────────────────────────────────────
pkl_path = os.path.join(SAVE_PATH, "asr_results.pkl")

if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
        asr_results = pickle.load(f)
    print(f"Loaded from cache: {len(asr_results)} speakers")
else:
    asr_results = []
    for item in tqdm(dataset_info, desc="ASR Inference"):
        try:
            wav, probs, tokens = get_predictions(item['path'])
            ratio = wav.shape[1] / probs.shape[0]

            eval_list = []
            for phn in item['phns']:
                sf = int(phn['start'] / ratio)
                ef = int(phn['end'] / ratio)
                seg_tokens = tokens[sf:ef+1]
                pred_clean = [t for t in seg_tokens if t not in ['[PAD]', 'sil', '<pad>']]
                correct = 1 if phn['phoneme'] in pred_clean else 0
                eval_list.append({**phn, 'is_correct': correct, 'pred': pred_clean[0] if pred_clean else 'sil'})

            asr_results.append({**item, 'wav': wav, 'probs': probs, 'tokens': tokens, 'eval': eval_list})
        except Exception as e:
            print(f"Error for {item['speaker']}: {e}")

    with open(pkl_path, 'wb') as f:
        pickle.dump(asr_results, f)
    print(f"Done! {len(asr_results)} speakers processed")


# ## 5. Base LIME Implementation

# In[5]:


class LimeASR:
    def __init__(self, model, processor, n_samples=100):
        self.model = model
        self.processor = processor
        self.n_samples = n_samples

    def perturb(self, wav, segments, target_idx, target_phn):
        num_segs = len(segments)
        mutants = np.random.randint(0, 2, (self.n_samples, num_segs))
        mutants[0, :] = 1
        labels, weights = [], []
        target_token_id = self.processor.tokenizer.convert_tokens_to_ids(target_phn)

        for i in range(self.n_samples):
            wav_mut = wav.clone()
            for s_idx, is_active in enumerate(mutants[i]):
                if not is_active:
                    wav_mut[:, segments[s_idx]['start']:segments[s_idx]['end']] = 0
            inputs = processor(wav_mut.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                l = ort_model(**inputs).logits
            # CPU-তে convert করে softmax
            if isinstance(l, torch.Tensor):
                l = l.cpu()
            else:
                l = torch.tensor(l)
            l = torch.softmax(l, dim=-1).squeeze(0)
            ratio = wav.shape[1] / l.shape[0]
            sf = int(segments[target_idx]['start'] / ratio)
            ef = int(segments[target_idx]['end'] / ratio)
            p = l[sf:ef+1, target_token_id].mean().item()
            labels.append(p)
            w = np.dot(mutants[i], mutants[0]) / (np.linalg.norm(mutants[i]) * np.linalg.norm(mutants[0]) + 1e-9)
            weights.append(w)

        ridge = Ridge(alpha=1.0)
        ridge.fit(mutants, labels, sample_weight=weights)
        return ridge.coef_

lime_base = LimeASR(ort_model, processor, n_samples=100)


# ## 6. LIME-WS (Window-Segment) Implementation

# In[6]:


class LimeWS(LimeASR):
    def explain(self, wav, segments, target_idx, window=5):
        num_segs = len(segments)
        start_w = max(0, target_idx - window)
        end_w = min(num_segs, target_idx + window + 1)
        active_indices = list(range(start_w, end_w))

        mutants = np.random.randint(0, 2, (self.n_samples, len(active_indices)))
        mutants[0, :] = 1
        labels, weights = [], []
        target_phn = segments[target_idx]['phoneme']
        target_id = self.processor.tokenizer.convert_tokens_to_ids(target_phn)

        for i in range(self.n_samples):
            wav_mut = wav.clone()
            for j, is_active in enumerate(mutants[i]):
                if not is_active:
                    idx = active_indices[j]
                    wav_mut[:, segments[idx]['start']:segments[idx]['end']] = 0
            inputs = processor(wav_mut.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                l = ort_model(**inputs).logits
            if isinstance(l, torch.Tensor):
                l = l.cpu()
            else:
                l = torch.tensor(l)
            l = torch.softmax(l, dim=-1).squeeze(0)
            ratio = wav.shape[1] / l.shape[0]
            p = l[int(segments[target_idx]['start']/ratio):int(segments[target_idx]['end']/ratio)+1, target_id].mean().item()
            labels.append(p)
            w = np.dot(mutants[i], mutants[0]) / (np.linalg.norm(mutants[i]) * np.linalg.norm(mutants[0]) + 1e-9)
            weights.append(w)

        rdg = Ridge(alpha=1.0)
        rdg.fit(mutants, labels, sample_weight=weights)
        full_coef = np.zeros(num_segs)
        for j, val in enumerate(active_indices):
            full_coef[val] = rdg.coef_[j]
        return full_coef

lime_ws = LimeWS(ort_model, processor, n_samples=100)


# ## 7. LIME-TS (Time-Segment) Implementation

# In[7]:


class LimeTS(LimeASR):
    def get_time_segs(self, wav, dur_ms=90):
        step = int(dur_ms * 16)
        return [{'start': i, 'end': min(i+step, wav.shape[1])} for i in range(0, wav.shape[1], step)]

    def explain(self, wav, phoneme_seg, dur_ms=90, window_ms=300):
        ts_segs = self.get_time_segs(wav, dur_ms)
        target_center = (phoneme_seg['start'] + phoneme_seg['end']) / 2
        active_indices = [i for i, s in enumerate(ts_segs) if abs(((s['start']+s['end'])/2) - target_center) < (window_ms * 16)]

        mutants = np.random.randint(0, 2, (self.n_samples, len(active_indices)))
        mutants[0, :] = 1
        labels, weights = [], []
        target_id = self.processor.tokenizer.convert_tokens_to_ids(phoneme_seg['phoneme'])

        for i in range(self.n_samples):
            wav_mut = wav.clone()
            for j, is_active in enumerate(mutants[i]):
                if not is_active:
                    idx = active_indices[j]
                    wav_mut[:, ts_segs[idx]['start']:ts_segs[idx]['end']] = 0
            inputs = processor(wav_mut.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                l = ort_model(**inputs).logits
            if isinstance(l, torch.Tensor):
                l = l.cpu()
            else:
                l = torch.tensor(l)
            l = torch.softmax(l, dim=-1).squeeze(0)
            ratio = wav.shape[1] / l.shape[0]
            p = l[int(phoneme_seg['start']/ratio):int(phoneme_seg['end']/ratio)+1, target_id].mean().item()
            labels.append(p)
            w = np.dot(mutants[i], mutants[0]) / (np.linalg.norm(mutants[i]) * np.linalg.norm(mutants[0]) + 1e-9)
            weights.append(w)

        rdg = Ridge()
        rdg.fit(mutants, labels, sample_weight=weights)
        return rdg.coef_, active_indices, ts_segs

lime_ts = LimeTS(ort_model, processor, n_samples=200)


# In[8]:


import time, os, pickle, numpy as np, pandas as pd
from tqdm.auto import tqdm

SAVE_PATH = r"D:\Allah vorosha"

pkl_path = os.path.join(SAVE_PATH, "validity_results.pkl")
checkpoint_path = os.path.join(SAVE_PATH, "validity_checkpoint.pkl")


if os.path.exists(pkl_path):
    os.remove(pkl_path)
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)

N_SAMPLES = 500
SAVE_EVERY = 10
GPU_REST_EVERY = 50

lime_base = LimeASR(ort_model, processor, n_samples=N_SAMPLES)
lime_ws   = LimeWS(ort_model, processor, n_samples=N_SAMPLES)
lime_ts   = LimeTS(ort_model, processor, n_samples=N_SAMPLES)


# ## 8. Validity Evaluation with Caching

# In[9]:


def run_random_baseline(num_elements, target_idx, ks=[1, 3, 5]):
    shuffled = np.random.permutation(num_elements)
    return {f'v{k}_rand': 1 if target_idx in shuffled[:k] else 0 for k in ks}

if os.path.exists(pkl_path):
    df_v = pd.read_pickle(pkl_path)
    print(f" Final results loaded from cache: {len(df_v)} phonemes")

else:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        all_validity = checkpoint['data']
        start_idx = checkpoint['speaker_idx']
        print(f" Resuming from speaker {start_idx}/{len(asr_results)}, {len(all_validity)} phonemes done so far")
    else:
        all_validity = []
        start_idx = 0
        print(f" Starting fresh — {len(asr_results)} speakers to process")

    pbar = tqdm(total=len(asr_results), initial=start_idx, desc="Validity Analysis")

    for speaker_idx in range(start_idx, len(asr_results)):
        item = asr_results[speaker_idx]

        for idx, p_seg in enumerate(item['eval']):
            if p_seg['is_correct'] == 0:
                continue

            try:
                coef_b = lime_base.perturb(item['wav'], item['eval'], idx, p_seg['phoneme'])
                rank_b = np.argsort(coef_b)[::-1]

                coef_ws = lime_ws.explain(item['wav'], item['eval'], idx)
                rank_ws = np.argsort(coef_ws)[::-1]

                coef_ts, active, ts_segs = lime_ts.explain(item['wav'], p_seg, dur_ms=90)
                rank_ts = np.argsort(coef_ts)[::-1]

                target_mid = (p_seg['start'] + p_seg['end']) / 2
                ts_v = {}
                for k in [1, 3, 5]:
                    v_k = 0
                    for r_idx in rank_ts[:k]:
                        real_idx = active[r_idx]
                        if ts_segs[real_idx]['start'] <= target_mid <= ts_segs[real_idx]['end']:
                            v_k = 1; break
                    ts_v[f'v{k}_ts'] = v_k

                all_validity.append({
                    'gender': item['gender'],
                    'phoneme': p_seg['phoneme'],
                    'v1_base': 1 if idx in rank_b[:1] else 0,
                    'v3_base': 1 if idx in rank_b[:3] else 0,
                    'v5_base': 1 if idx in rank_b[:5] else 0,
                    'v1_ws':   1 if idx in rank_ws[:1] else 0,
                    'v3_ws':   1 if idx in rank_ws[:3] else 0,
                    'v5_ws':   1 if idx in rank_ws[:5] else 0,
                    **ts_v,
                    **run_random_baseline(len(item['eval']), idx)
                })

            except Exception as e:
                print(f" Error at speaker {item['speaker']}, phoneme {p_seg['phoneme']}: {e}")
                continue

        pbar.update(1)

        if (speaker_idx + 1) % SAVE_EVERY == 0:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'data': all_validity, 'speaker_idx': speaker_idx + 1}, f)
            pbar.set_postfix({'saved': speaker                               _idx + 1, 'phonemes': len(all_validity)})

        if (speaker_idx + 1) % GPU_REST_EVERY == 0:
            time.sleep(2)

    pbar.close()

    df_v = pd.DataFrame(all_validity)
    df_v.to_pickle(pkl_path)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\n Done! Total phonemes evaluated: {len(df_v)}")
    print(f"LIME-TS Validity@3: {df_v['v3_ts'].mean():.4f}")


# In[10]:


import pandas as pd
import os 

SAVE_PATH = r"D:\Allah vorosha"          
df_v = pd.read_pickle(os.path.join(SAVE_PATH, "validity_results.pkl"))
print(df_v.columns.tolist())
print(df_v.head(2)


# ## 9. Table 1 (Paper Format) and Statistical Significance

# In[11]:


def build_table_1(df):
    metrics = ['v1', 'v3', 'v5']
    methods = ['base', 'ws', 'ts', 'rand']
    cols = [f"{met}_{m}" for m in methods for met in metrics]

    summary = df.groupby('gender')[cols].mean()
    summary.loc['All'] = df[cols].mean()

    summary.columns = pd.MultiIndex.from_product(
        [['Base LIME', 'LIME-WS', 'LIME-TS', 'Random'], ['V@1', 'V@3', 'V@5']]
    )
    return summary

display(build_table_1(df_v).round(4))

# ANOVA + Tukey HSD for TS-V3
print("\n--- ANOVA: LIME-TS Validity@3 by Gender ---")
from statsmodels.formula.api import ols
import statsmodels.api as sm

model_aov = ols('v3_ts ~ gender', data=df_v).fit()
display(sm.stats.anova_lm(model_aov, typ=2))

# Wilcoxon
from scipy import stats
print("\nWilcoxon Signed-Rank Test (LIME-WS vs LIME-TS V@3):")
stat, p_val = stats.wilcoxon(df_v['v3_ws'], df_v['v3_ts'])
print(f"Statistic: {stat}, P-value: {p_val:.4e}")


# ## 10. Phoneme Error Analysis (Figure 3 & Table 2)

# In[12]:


err_list = []
for r in asr_results:
    for e in r['eval']:
        if e['is_correct'] == 0:
            err_list.append({'gender': r['gender'], 'truth': e['phoneme'], 'pred': e['pred'], 'pair': f"{e['phoneme']}->{e['pred']}"})

df_err = pd.DataFrame(err_list)

# Figure 3: Error bar chart
plt.figure(figsize=(12, 5))
sns.countplot(data=df_err, x='pair', hue='gender', order=df_err['pair'].value_counts().index[:10])
plt.title("Figure 3: Top 10 Transcription Errors by Gender")
plt.xticks(rotation=45); plt.show()

# Table 2: er->uw substitution
print("Table 2: Top 3 important regions for 'er->uw' substitution error (gender sensitive)")
display(df_err[df_err['pair'] == 'er->uw'].groupby('gender').size().to_frame(name='Count'))


# ## 11. Custom Improvement: Duration Sensitivity (Actual Computation)

# In[13]:


def compute_v3_sens(dur_ms, limit=100):
    hits, count = 0, 0
    pbar = tqdm(total=limit, desc=f"TS {dur_ms}ms")
    for item in asr_results:
        for idx, p_seg in enumerate(item['eval']):
            if p_seg['is_correct'] == 0: continue

            c, act, tss = lime_ts.explain(item['wav'], p_seg, dur_ms=dur_ms)
            ranks = np.argsort(c)[::-1]
            target_mid = (p_seg['start'] + p_seg['end']) / 2

            for r_idx in ranks[:3]:
                if tss[act[r_idx]]['start'] <= target_mid <= tss[act[r_idx]]['end']:
                    hits += 1; break
            count += 1; pbar.update(1)
            if count >= limit: break
        if count >= limit: break
    pbar.close()
    return hits / count

res_50 = compute_v3_sens(50)
res_70 = compute_v3_sens(70) # Or use existing from df_v if aligned
res_90 = compute_v3_sens(90)

sens_df = pd.DataFrame({'Duration': ['50ms', '70ms', '90ms'], 'Validity@3': [res_50, res_70, res_90]})
display(sens_df)

sns.barplot(data=sens_df, x='Duration', y='Validity@3', palette='magma')
plt.title("LIME-TS Sensitivity to Segment Duration"); plt.show()


# ## 12. Visualization: Waveform Heatmap Explanations

# In[14]:


def plot_xai(r_idx, p_idx):
    item = asr_results[r_idx]
    target = item['eval'][p_idx]

    coef, active, ts_segs = lime_ts.explain(item['wav'], target)
    wav_np = item['wav'].numpy().squeeze()

    plt.figure(figsize=(15, 3))
    plt.plot(wav_np, color='lightgray', alpha=0.5)

    # Scale coef for color intensity
    v_min, v_max = coef.min(), coef.max()
    norm = (coef - v_min) / (v_max - v_min + 1e-9)

    for i, a_idx in enumerate(active):
        s = ts_segs[a_idx]
        plt.axvspan(s['start'], s['end'], color='red', alpha=norm[i]*0.6)

    plt.axvspan(target['start'], target['end'], color='green', alpha=0.4, label='Ground Truth')
    plt.title(f"XAI for '{target['phoneme']}' in speaker {item['speaker']} (Red=Importance)")
    plt.legend(); plt.show()

plot_xai(0, 5)


# In[ ]:





# ## Cell 13 — Save Model

# In[15]:


# ============================================================
# CELL 13: Save Model
# ============================================================
import os
import shutil
import json

SAVE_MODEL_PATH = r"D:\Allah vorosha\saved_model"
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

# Save processor
processor_path = os.path.join(SAVE_MODEL_PATH, "processor")
processor.save_pretrained(processor_path)
print(f"✅ Processor saved → {processor_path}")

# Save ONNX model
onnx_src = None
for possible in [
    os.path.expanduser("~/.cache/huggingface/hub"),
    os.path.join(os.environ.get("USERPROFILE", ""), ".cache", "huggingface", "hub")
]:
    if os.path.exists(possible):
        for root, dirs, files in os.walk(possible):
            for f in files:
                if f.endswith(".onnx") and "wav2vec2" in root.lower():
                    onnx_src = os.path.join(root, f)
                    break

if onnx_src:
    onnx_dst = os.path.join(SAVE_MODEL_PATH, "model.onnx")
    shutil.copy2(onnx_src, onnx_dst)
    size_mb = os.path.getsize(onnx_dst) / (1024*1024)
    print(f" ONNX model saved → {onnx_dst} ({size_mb:.1f} MB)")
else:
    # Direct export
    ort_model.save_pretrained(SAVE_MODEL_PATH)
    print(f" Model saved → {SAVE_MODEL_PATH}")

# Save config
config = {
    "model_id": "patrickvonplaten/wav2vec2-base-timit-fine-tuned",
    "provider": "CPUExecutionProvider",
    "target_sr": 16000,
    "n_samples": 500,
    "ts_dur_ms": 90
}
with open(os.path.join(SAVE_MODEL_PATH, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
print(f" Config saved → {SAVE_MODEL_PATH}\\config.json")
print(f"\n All files in {SAVE_MODEL_PATH}:")
for f in os.listdir(SAVE_MODEL_PATH):
    size = os.path.getsize(os.path.join(SAVE_MODEL_PATH, f))
    print(f"   {f} — {size/1024:.1f} KB")


# In[ ]:





# ## Cell 14 — SHAP Analysis

# In[22]:


# ============================================================
# CELL 14: SHAP Analysis
# ============================================================
import subprocess
subprocess.run(["pip", "install", "shap", "-q"], check=True)
import shap

print("Running SHAP analysis on first 20 correct phonemes...")

# Collect 20 correct phonemes
shap_samples = []
for item in asr_results:
    for idx, p_seg in enumerate(item['eval']):
        if p_seg['is_correct'] == 1:
            shap_samples.append({'item': item, 'idx': idx, 'p_seg': p_seg})
        if len(shap_samples) >= 20:
            break
    if len(shap_samples) >= 20:
        break

print(f"Collected {len(shap_samples)} correct phonemes for SHAP")

def predict_fn(masks, item, p_idx, p_seg):
    results = []
    target_id = processor.tokenizer.convert_tokens_to_ids(p_seg['phoneme'])
    if target_id is None or target_id == processor.tokenizer.unk_token_id:
        return np.zeros(len(masks))

    for mask in masks:
        try:
            wav_mut = item['wav'].clone()
            for s_idx, is_active in enumerate(mask):
                if not is_active and s_idx < len(item['eval']):
                    seg = item['eval'][s_idx]
                    wav_mut[:, int(seg['start']):int(seg['end'])] = 0

            inputs = processor(
                wav_mut.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            )
            with torch.no_grad():
                logits = ort_model(**inputs).logits
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            ratio = item['wav'].shape[1] / probs.shape[0]
            sf = max(0, int(p_seg['start'] / ratio))
            ef = min(probs.shape[0]-1, int(p_seg['end'] / ratio))
            p = probs[sf:ef+1, target_id].mean().item()
            results.append(float(p))
        except:
            results.append(0.0)
    return np.array(results)

# Run SHAP and compare with LIME-TS
agreement_count = 0
total_count = 0

for sample in tqdm(shap_samples, desc="SHAP Analysis"):
    item  = sample['item']
    idx   = sample['idx']
    p_seg = sample['p_seg']
    n_segs = len(item['eval'])

    if n_segs < 3:
        continue

    try:
        background = np.ones((1, n_segs))
        explainer  = shap.KernelExplainer(
            lambda m: predict_fn(m, item, idx, p_seg),
            background
        )
        shap_vals = explainer.shap_values(
            np.ones((1, n_segs)), nsamples=50, silent=True
        )
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals = np.array(shap_vals).flatten()
        shap_top3 = set(np.argsort(shap_vals)[::-1][:3])

        # LIME-TS top3
        coef_ts, active, ts_segs = lime_ts.explain(item['wav'], p_seg, dur_ms=90)
        rank_ts = np.argsort(coef_ts)[::-1]
        target_mid = (p_seg['start'] + p_seg['end']) / 2

        gt_in_lime = 0
        for r in rank_ts[:3]:
            real_idx = active[r]
            seg = ts_segs[real_idx]
            if seg['start'] <= target_mid <= seg['end']:
                gt_in_lime = 1; break

        gt_in_shap = 1 if idx in shap_top3 else 0

        if gt_in_shap == gt_in_lime:
            agreement_count += 1
        total_count += 1

    except Exception as e:
        print(f" Skipping: {e}")
        continue

agreement_pct = agreement_count / total_count * 100 if total_count > 0 else 0
lime_ts_pct   = df_v['v3_ts'].mean() * 100

print(f"\n SHAP vs LIME-TS Agreement: {agreement_pct:.1f}%")
print(f"   ({agreement_count}/{total_count} phonemes agree on ground truth)")

# Bar chart
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(
    ['LIME-TS\nValidity@3', 'SHAP\nValidity@3', 'SHAP vs LIME-TS\nAgreement'],
    [lime_ts_pct, agreement_pct, agreement_pct],
    color=['#e67e22', '#9b59b6', '#2ecc71']
)
ax.set_ylabel('Score (%)')
ax.set_title('SHAP vs LIME-TS Comparison')
ax.set_ylim(0, 110)
for bar, val in zip(bars, [lime_ts_pct, agreement_pct, agreement_pct]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'shap_comparison.png'), dpi=150)
plt.show()


# In[ ]:





# ## Cell 15 — LIME-TS-Overlap

# In[23]:


# ============================================================
# CELL 15: LIME-TS-Overlap (Original Contribution)
# ============================================================

class LimeTSOverlap(LimeTS):
    def get_time_segs(self, wav, dur_ms=90):
        step    = int(dur_ms * 16)
        overlap = step // 2  # 50% overlap
        return [
            {'start': i, 'end': min(i+step, wav.shape[1])}
            for i in range(0, wav.shape[1]-overlap, overlap)
        ]

lime_ts_overlap = LimeTSOverlap(ort_model, processor, n_samples=200)
print(" LIME-TS-Overlap defined (50% overlapping 90ms segments)")

# Run on first 100 correct phonemes
overlap_results = []
count = 0
for item in tqdm(asr_results, desc="LIME-TS-Overlap"):
    for idx, p_seg in enumerate(item['eval']):
        if p_seg['is_correct'] == 0:
            continue
        try:
            coef_ts, active, ts_segs = lime_ts_overlap.explain(
                item['wav'], p_seg, dur_ms=90
            )
            rank_ts = np.argsort(coef_ts)[::-1]
            target_mid = (p_seg['start'] + p_seg['end']) / 2

            row = {'gender': item['gender'], 'phoneme': p_seg['phoneme']}
            for k in [1, 3, 5]:
                v_k = 0
                for r in rank_ts[:k]:
                    real_idx = active[r]
                    seg = ts_segs[real_idx]
                    if seg['start'] <= target_mid <= seg['end']:
                        v_k = 1; break
                row[f'v{k}_ts_overlap'] = v_k
            overlap_results.append(row)
            count += 1
        except:
            continue
        if count >= 100:
            break
    if count >= 100:
        break

df_overlap = pd.DataFrame(overlap_results)
print(f"\n LIME-TS-Overlap Results (n={len(df_overlap)}):")
print(f"   V@1: {df_overlap['v1_ts_overlap'].mean():.4f}")
print(f"   V@3: {df_overlap['v3_ts_overlap'].mean():.4f}")
print(f"   V@5: {df_overlap['v5_ts_overlap'].mean():.4f}")

# Compare with LIME-TS
lime_ts_v3    = df_v['v3_ts'].mean()
overlap_v3    = df_overlap['v3_ts_overlap'].mean()

fig, ax = plt.subplots(figsize=(7, 4))
methods = ['LIME-TS\n(70ms, paper)', 'LIME-TS\n(90ms, ours)', 'LIME-TS-Overlap\n(90ms+50%, ours)']
values  = [0.96, lime_ts_v3, overlap_v3]
colors  = ['#95a5a6', '#e67e22', '#e74c3c']
bars    = ax.bar(methods, values, color=colors, width=0.5)
ax.axhline(0.96, color='gray', linestyle='--', label='Paper baseline (0.96)')
ax.set_ylabel('Validity@3')
ax.set_title('LIME-TS Variants Comparison (Original Contribution)')
ax.set_ylim(0, 1.1)
for bar, val in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'lime_ts_overlap_comparison.png'), dpi=150)
plt.show()


# In[ ]:





# ## Cell 16 — Vowel vs Consonant

# In[24]:


# ============================================================
# CELL 16: Vowel vs Consonant Analysis
# ============================================================

VOWELS = ['aa','ae','ah','ao','aw','ay','eh','er',
          'ey','ih','iy','ow','oy','uh','uw']

df_v['type'] = df_v['phoneme'].apply(
    lambda p: 'Vowel' if p in VOWELS else 'Consonant'
)

print("Vowel vs Consonant LIME-TS Validity@3:")
print(df_v.groupby('type')['v3_ts'].mean())
print("\nBy Gender:")
print(df_v.groupby(['type','gender'])['v3_ts'].mean())

# Bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, metric, k in zip(axes, ['v1_ts','v3_ts','v5_ts'], [1,3,5]):
    data = df_v.groupby(['type','gender'])[metric].mean().unstack()
    if data.empty:
        continue
    data.plot(kind='bar', ax=ax, color=['#3498db','#e74c3c','#2ecc71'])
    ax.set_title(f'LIME-TS Validity@{k}')
    ax.set_ylabel(f'Validity@{k}')
    ax.set_xlabel('')
    ax.set_xticklabels(data.index.tolist(), rotation=0)
    ax.legend(title='Gender')
    ax.set_ylim(0, 1.1)

plt.suptitle('Vowel vs Consonant Analysis by Gender',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'vowel_consonant_analysis.png'), dpi=150)
plt.show()


# In[ ]:





# ## Cell 17 — Attention Visualization

# In[25]:


# ============================================================
# CELL 17: Attention Visualization
# ============================================================
from transformers import Wav2Vec2ForCTC as Wav2Vec2Full

print("Loading model with attention output...")
attn_model = Wav2Vec2Full.from_pretrained(
    "patrickvonplaten/wav2vec2-base-timit-fine-tuned",
    output_attentions=True
)
attn_model.eval()

for sample_idx in range(3):
    item = asr_results[sample_idx]
    inputs = processor(
        item['wav'].squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = attn_model(**inputs, output_attentions=True)

    # Last layer attention — average over heads
    last_attn = outputs.attentions[-1][0]  # [heads, T, T]
    avg_attn  = last_attn.mean(0).mean(0).numpy()  # [T]
    avg_attn  = (avg_attn - avg_attn.min()) / (avg_attn.max() - avg_attn.min() + 1e-9)

    wav_np = item['wav'].squeeze().numpy()
    t_axis = np.arange(len(wav_np)) / 16000
    ratio  = len(wav_np) / len(avg_attn)
    attn_upsampled = np.interp(
        np.arange(len(wav_np)),
        np.arange(len(avg_attn)) * ratio,
        avg_attn
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.plot(t_axis, wav_np, color='gray', alpha=0.6, linewidth=0.5)
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Waveform — Speaker: {item["speaker"]}')

    ax2.fill_between(t_axis, attn_upsampled, alpha=0.6, color='#e74c3c')
    for p in item['eval']:
        ax2.axvspan(p['start']/16000, p['end']/16000,
                   alpha=0.2, color='blue')
        mid = (p['start'] + p['end']) / 2 / 16000
        ax2.text(mid, 0.9, p['phoneme'], ha='center',
                fontsize=7, color='darkblue')
    ax2.set_ylabel('Attention Weight')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Model Attention (red) vs Phoneme Boundaries (blue)')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f'attention_{sample_idx}.png'), dpi=150)
    plt.show()


# In[ ]:





# ## Cell 18 — Combined Results Table

# In[26]:


# ============================================================
# CELL 18: Combined Results Table
# ============================================================
from IPython.display import display, HTML

paper_results = {
    'Base LIME':  {'V@1': 0.40, 'V@3': 0.62, 'V@5': 0.67},
    'LIME-WS':    {'V@1': 0.49, 'V@3': 0.76, 'V@5': 0.83},
    'LIME-TS':    {'V@1': 0.86, 'V@3': 0.96, 'V@5': 0.97},
}

your_results = {
    'Base LIME':       {'V@1': df_v['v1_base'].mean(), 'V@3': df_v['v3_base'].mean(), 'V@5': df_v['v5_base'].mean()},
    'LIME-WS':         {'V@1': df_v['v1_ws'].mean(),   'V@3': df_v['v3_ws'].mean(),   'V@5': df_v['v5_ws'].mean()},
    'LIME-TS':         {'V@1': df_v['v1_ts'].mean(),   'V@3': df_v['v3_ts'].mean(),   'V@5': df_v['v5_ts'].mean()},
    'LIME-TS-Overlap': {'V@1': df_overlap['v1_ts_overlap'].mean(), 'V@3': df_overlap['v3_ts_overlap'].mean(), 'V@5': df_overlap['v5_ts_overlap'].mean()},
}

html = """
<style>
.final-table { border-collapse: collapse; font-family: serif; font-size: 13px; }
.final-table th, .final-table td { border: 1px solid black; padding: 6px 12px; text-align: center; }
.final-table .paper-bg { background-color: #f8d7da; }
.final-table .ours-bg  { background-color: #d4edda; }
.final-table .new-bg   { background-color: #cce5ff; }
.final-table .better   { color: #155724; font-weight: bold; }
.final-table .worse    { color: #721c24; }
</style>
<table class="final-table">
<tr>
    <th rowspan="2">Method</th>
    <th colspan="3" class="paper-bg">Paper (Kaldi)</th>
    <th colspan="3" class="ours-bg">Ours (Wav2Vec2)</th>
    <th colspan="3" class="new-bg">Improvement</th>
</tr>
<tr>
    <th class="paper-bg">V@1</th><th class="paper-bg">V@3</th><th class="paper-bg">V@5</th>
    <th class="ours-bg">V@1</th><th class="ours-bg">V@3</th><th class="ours-bg">V@5</th>
    <th class="new-bg">V@1</th><th class="new-bg">V@3</th><th class="new-bg">V@5</th>
</tr>
"""

for method in ['Base LIME', 'LIME-WS', 'LIME-TS']:
    p = paper_results[method]
    o = your_results[method]
    html += f"<tr><td><b>{method}</b></td>"
    for k in ['V@1','V@3','V@5']:
        html += f'<td class="paper-bg">{p[k]:.2f}</td>'
    for k in ['V@1','V@3','V@5']:
        html += f'<td class="ours-bg">{o[k]:.2f}</td>'
    for k in ['V@1','V@3','V@5']:
        diff = o[k] - p[k]
        cls = 'better' if diff >= 0 else 'worse'
        sign = '+' if diff >= 0 else ''
        html += f'<td class="new-bg {cls}">{sign}{diff:.2f}</td>'
    html += "</tr>"

# LIME-TS-Overlap (new)
o = your_results['LIME-TS-Overlap']
html += f"<tr><td><b>LIME-TS-Overlap </b></td>"
for k in ['V@1','V@3','V@5']:
    html += f'<td class="paper-bg">—</td>'
for k in ['V@1','V@3','V@5']:
    html += f'<td class="new-bg">{o[k]:.2f}</td>'
for k in ['V@1','V@3','V@5']:
    html += f'<td class="new-bg">New</td>'
html += "</tr></table>"
html += "<br><small> = Original contribution not in paper</small>"

display(HTML(html))

# Save
with open(os.path.join(SAVE_PATH, "final_results.html"), "w") as f:
    f.write(html)
print(f" Saved → {SAVE_PATH}final_results.html")


# In[ ]:





# In[ ]:




