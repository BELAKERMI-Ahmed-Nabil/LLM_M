# ABSA LLM System — llama3.1:8b
**Aspect-Based Sentiment Analysis using Large Language Models (Zero/One/Few-Shot)**

This is the **upgraded version** of the ABSA LLM system using `llama3.1:8b` (8 billion parameters)
instead of `llama3.2:3b`. The larger model achieves significantly better results,
especially with few-shot prompting.

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
- Downloaded automatically — no manual download needed

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
- Make sure Ollama is **running** before executing any script

### 3. Download the LLM model
```bash
ollama pull llama3.1:8b
```
- Size: **4.9 GB**
- Requires at least **8 GB RAM** (16 GB recommended)
- GPU optional but speeds up inference significantly

---

## Project Structure

```
ABSA_LLM_8b/
  run_zero_shot.py          # Strategy 1: No examples given to model
  run_one_shot.py           # Strategy 2: 1 training example in prompt
  run_few_shot.py           # Strategy 3: Few-shot (10 or 50 examples)
  compare_8b_vs_3b.py       # Compare 8b vs 3b vs SetFitQuad
  results/
    zero_shot_results.json
    one_shot_results.json
    few_shot_v2_10ex_results.json
    charts/                 # Generated comparison charts
```

---

## How to Run

```bash
# Quick test (50 reviews, fast)
python run_zero_shot.py
python run_one_shot.py
python run_few_shot.py

# Full run (778 reviews) — for final results
python run_zero_shot.py --full
python run_one_shot.py  --full
python run_few_shot.py  --full --shots 10    # 10 examples (recommended)
python run_few_shot.py  --full --shots 50    # 50 examples (NOT recommended — see note)

# Generate comparison charts (vs 3b and SetFitQuad)
python compare_8b_vs_3b.py
```

> **Note on 50 shots:** Even with 8b, using 50 examples causes the model to lose focus
> and produce incorrect output format ("Lost in the Middle" problem).
> **10 examples is the optimal setting.**

---

## Results (778 test reviews)

| Strategy | Exact F1 | Partial F1 |
|---|---|---|
| Zero-Shot (0 examples) | 0.1121 | 0.3842 |
| One-Shot (1 example) | 0.2417 | 0.4745 |
| **Few-Shot (10 examples)** | **0.3371** | **0.6186** |
| Few-Shot (50 examples) | 0.0744 | 0.1576 (failed) |

### Comparison with baselines

| System | Partial F1 | Training Cost |
|---|---|---|
| **Our Few-Shot 8b (10ex)** | **0.619** | 0 (no training) |
| T5-ASQP (paper baseline) | 0.590 | Full fine-tuning |
| SetFitQuad (paper) | 0.532 | 50 labeled examples |
| Our Few-Shot 3b (10ex) | 0.577 | 0 (no training) |
| SetFitQuad (our runs) | 0.425 | 50 labeled examples |

> **Our 8b Few-Shot outperforms T5-ASQP by +2.9% with ZERO training cost.**

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
System Prompt (ABSA instructions + exact category list)
    +
10 Diverse Few-Shot Examples (auto-selected from training set)
    |
    v
Ollama (llama3.1:8b — local inference, no internet)
    |
    v
Raw text output
    |
    v
Regex Parser + Category Normalization → Quadruplets
    |
    v
Evaluation (Exact F1 + Partial F1)
```

---

## Why 8b Beats 3b in Few-Shot

| Aspect | llama3.2:3b | llama3.1:8b |
|---|---|---|
| Parameters | 3 billion | 8 billion |
| Context handling | Loses focus with many examples | Better instruction following |
| Few-Shot 10ex | Partial F1 = 0.577 | Partial F1 = 0.619 |
| Zero-Shot | Partial F1 = 0.508 | Partial F1 = 0.384 |
| Model size | 2.0 GB | 4.9 GB |

> Interesting: 3b is better at Zero-Shot (it guesses more aggressively),
> while 8b is better at Few-Shot (it follows instructions more precisely).

---

## Related Projects

- `C:\ABSA_LLM` — Same system with **llama3.2:3b** (lighter, faster)
- `C:\SetFitQuad-Code-main` — SetFitQuad fine-tuning approach (comparison target)
