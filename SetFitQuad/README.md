# SetFitQuad — Few-Shot Aspect Sentiment Quad Prediction
**A Few-Shot Framework for ASQP with Sampling Strategies**
*(Submitted to IEEE Access)*

This project implements **SetFitQuad**, a few-shot learning framework for
Aspect-Based Sentiment Analysis (ABSA) quad prediction using SetFit
(Sentence Transformers + contrastive fine-tuning).

---

## What This Project Does

Extracts sentiment quadruplets from restaurant reviews:

| Field | Description | Example |
|---|---|---|
| Aspect Term (AT) | The thing being discussed | *"pasta"* |
| Aspect Category (AC) | Category from fixed list | *"food quality"* |
| Opinion Term (OT) | The opinion expressed | *"amazing"* |
| Sentiment Polarity (SP) | positive / negative / neutral | *"positive"* |

---

## Dataset

- **Name:** `JaquanTW/fewshot-absaquad` (HuggingFace)
- **Source:** Rest15 + Rest16 (restaurant review benchmarks)
- **Few-shot setting:** 50 training examples per experiment
- **Evaluation:** 5-fold cross-validation on 778 test reviews
- Downloaded automatically — no manual download needed

---

## Project Structure

```
SetFitQuad-Code-main/
  exp1/
    exp1_samplestrategy.py      # 19 sampling strategies comparison
  exp2/
    exp2_pretrainmodel_cs.py    # 5 models with Cluster Sampling
    exp2_pretrainmodel_mes.py   # 5 models with Max Entropy Sampling
    exp2_pretrainmodel_rf.py    # 5 models with Random Forest Sampling
  exp3/
    exp3_samplesize.py          # Training size effect (1 to 200 examples)
  exp4/
    exp4_comparision_cs.py      # SetFitQuad vs baselines (Cluster Sampling)
    exp4_comparision_dbs.py     # SetFitQuad vs baselines (Density-based)
    exp4_comparision_rf.py      # SetFitQuad vs baselines (Random Forest)
  results/
    *_5fold_average_results.json    # Final averaged results (19 strategies)
  exp2/results/                     # EXP2 fold results
  exp3/results/                     # EXP3 fold results
  analyze_results.py            # Generate analysis charts
  experiment-docs.md            # Detailed experiment descriptions
  env-setup.md                  # Full environment setup guide
```

---

## Requirements & Installation

### Step 1 — Create Conda environment
```bash
conda create -n setfit python=3.9
conda activate setfit
```

### Step 2 — Install PyTorch
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Step 3 — Install required packages
```bash
pip install setfit transformers datasets scikit-learn torchmetrics tqdm protobuf==3.20.* scipy spacy seaborn matplotlib GPUtil
```

### Step 4 — Install Spacy language models
```bash
set KMP_DUPLICATE_LIB_OK=TRUE
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_sm
```

> **No Ollama or LLM needed.** SetFitQuad uses small Sentence Transformer models
> (~90MB–1GB) downloaded automatically from HuggingFace.

---

## Pretrained Models Used (auto-downloaded)

| Model | Size |
|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB |
| `sentence-transformers/all-mpnet-base-v2` | ~420 MB |
| `sentence-transformers/multi-qa-mpnet-base-cos-v1` | ~420 MB |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | ~1 GB |
| `sentence-transformers/paraphrase-TinyBERT-L6-v2` | ~240 MB |

---

## How to Run

```bash
# EXP1: All 19 sampling strategies (~2-3 hours)
cd exp1
python exp1_samplestrategy.py

# EXP2: Pretrained model comparison
cd exp2
python exp2_pretrainmodel_cs.py

# EXP3: Sample size effect (learning curve)
cd exp3
python exp3_samplesize.py

# EXP4: Full comparison vs baselines
cd exp4
python exp4_comparision_cs.py

# Analyze and visualize all results
python analyze_results.py
```

---

## Experiments Overview

### EXP1 — Sampling Strategies (19 strategies, 50 examples each)

| Type | Strategies |
|---|---|
| Basic (6) | Random Seed, Grid, Max-Min Distance, Density-based, Max Entropy, Cluster |
| Ensemble (13) | Lasso/Ridge/ElasticNet/Random Forest (sizes 20,30,40) + Equal Proportion |

**Best:** Cluster Sampling — Partial F1 = **0.4213**

### EXP2 — Pretrained Model Comparison

**Best:** multi-qa-mpnet-base-cos-v1 — Partial F1 = **0.4250**

### EXP3 — Sample Size Effect

| Training Size | Partial F1 |
|---|---|
| 1 | 0.2375 |
| 10 | 0.4037 |
| 50 | 0.4260 |
| 100 | 0.4380 |
| **150** | **0.4468** |
| 200 | 0.4400 |

### EXP4 — Comparison vs Baselines

| System | Partial F1 | Training Time |
|---|---|---|
| **SetFitQuad (paper)** | **0.532** | 461 sec |
| T5-ASQP baseline | 0.590 | 1334 sec |
| SetFitABSA | — | 661 sec |

---

## Evaluation Metrics

**Exact Match F1** — all 4 fields must match exactly (very strict).

**Partial Match F1** — main metric, same formula as paper:
```
score = (category_exact + sentiment_exact + aspect_LCS + opinion_LCS) / 4
```
Uses Hungarian algorithm for optimal quad assignment.

---

## How SetFitQuad Works

```
50 Training Examples
    |
    v
Sampling Strategy (selects most informative examples)
    |
    v
SetFit Contrastive Fine-tuning
(Sentence Transformer learns quad-level representations)
    |
    v
Logistic Regression Head
    |
    v
Test Review → "span|category|opinion|sentiment" string
    |
    v
Parser → structured quad → Evaluation
```

---

## Results vs LLM Approach

| System | Partial F1 | Training Cost |
|---|---|---|
| Our LLM 8b Few-Shot | **0.619** | 0 (no training) |
| T5-ASQP (paper) | 0.590 | Full fine-tuning |
| SetFitQuad (paper) | 0.532 | 50 examples + fine-tuning |
| SetFitQuad (our runs) | 0.425 | 50 examples + fine-tuning |

---

## Related Projects

- `C:\ABSA_LLM` — LLM-based approach with llama3.2:3b
- `C:\ABSA_LLM_8b` — LLM-based approach with llama3.1:8b (best overall results)
