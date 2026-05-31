# ABSA LLM System — llama3.2:3b
**Aspect-Based Sentiment Analysis using Large Language Models (Zero/One/Few-Shot)**

This project extracts sentiment quadruplets from restaurant reviews using prompting strategies
with a local LLM (no fine-tuning required).

---

## What This Project Does

Given a restaurant review like:
> *"The food was delicious but the service was really slow."*

The system extracts structured quadruplets:
```
Aspect: food     | Category: food quality    | Opinion: delicious | Sentiment: positive
Aspect: service  | Category: service general | Opinion: slow      | Sentiment: negative
```

---

## Dataset

- **Name:** `JaquanTW/fewshot-absaquad` (HuggingFace)
- **Source:** Rest15 + Rest16 (restaurant reviews)
- **Test split:** 778 unique reviews (used for evaluation)
- **Train split:** 1088 unique reviews (used for few-shot examples only)
- Downloaded automatically by the scripts — no manual download needed

---

## Requirements

### 1. Install Python packages
```bash
pip install ollama datasets pandas numpy matplotlib
```

### 2. Install Ollama
Download from: https://ollama.com/download
- Available for Windows, Mac, Linux
- Runs LLM models locally (no internet needed after download)

### 3. Download the LLM model
```bash
ollama pull llama3.2:latest
```
- Size: **2.0 GB**
- Make sure Ollama is running before executing the scripts

---

## Project Structure

```
ABSA_LLM/
  run_zero_shot.py          # Strategy 1: No examples given to model
  run_one_shot.py           # Strategy 2: 1 training example in prompt
  run_few_shot.py           # Strategy 3: Few-shot (original, 10 examples)
  run_few_shot_v2.py        # Strategy 4: Few-shot v2 (improved prompt + category fix)
  final_comparison.py       # Compare our results vs SetFitQuad
  requirements.txt          # Python dependencies
  results/
    zero_shot_results.json
    one_shot_results.json
    few_shot_10ex_results.json
    few_shot_v2_10ex_results.json
    charts/                 # Generated comparison charts
```

---

## How to Run

```bash
# Quick test (50 reviews)
python run_zero_shot.py
python run_one_shot.py
python run_few_shot_v2.py

# Full run (778 reviews) — recommended for final results
python run_zero_shot.py --full
python run_one_shot.py  --full
python run_few_shot_v2.py --full
python run_few_shot_v2.py --full --shots 10   # specify number of examples

# Generate comparison charts
python final_comparison.py
```

---

## Results (778 test reviews)

| Strategy | Exact F1 | Partial F1 |
|---|---|---|
| Zero-Shot (0 examples) | 0.2008 | 0.5081 |
| One-Shot (1 example) | 0.1874 | 0.4051 |
| Few-Shot v1 (10 examples) | 0.2342 | 0.5001 |
| **Few-Shot v2 (10 examples)** | **0.2777** | **0.5770** |

**SetFitQuad (paper baseline):** Partial F1 = 0.532

> Our Few-Shot v2 outperforms the published SetFitQuad paper by **+4.5%** with zero fine-tuning.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Exact F1** | All 4 fields must match exactly (strict) |
| **Partial F1** | LCS-based similarity + Hungarian matching (same as SetFitQuad paper) |

---

## How It Works

```
Review Text
    |
    v
System Prompt (ABSA instructions + category list)
    +
Few-Shot Examples (from training set)
    |
    v
Ollama (llama3.2:latest — local inference)
    |
    v
Raw text output
    |
    v
Regex Parser → Quadruplets
    |
    v
Evaluation (Exact + Partial F1)
```

---

## Prompting Strategies Explained

| Strategy | Examples in Prompt | Description |
|---|---|---|
| Zero-Shot | 0 | Only instructions, no examples |
| One-Shot | 1 | One example from training set |
| Few-Shot v1 | 10 | 10 random examples |
| Few-Shot v2 | 10 | 10 diverse examples + improved prompt + category normalization |

---

## Related Project

- `C:\ABSA_LLM_8b` — Same system but with **llama3.1:8b** (stronger model)
- `C:\SetFitQuad-Code-main` — SetFitQuad fine-tuning based approach (comparison target)
