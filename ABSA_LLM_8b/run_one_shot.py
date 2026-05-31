"""
run_one_shot.py
===============
ABSA Quad Extraction — ONE-SHOT strategy.

Input : restaurant reviews (text column) from Rest15 + Rest16
Output: quadruplets (Aspect, Category, Opinion, Sentiment)

ONE example from the training set is shown to the model before each review.

Usage:
    python run_one_shot.py                  # runs on 50 reviews (default)
    python run_one_shot.py --full           # all 778 reviews
    python run_one_shot.py --samples 20     # custom count
"""

import argparse, json, os, sys, time, re, random
sys.path.insert(0, os.path.dirname(__file__))

import ollama
from datasets import load_dataset
from collections import Counter

# ─────────────────────────── CONFIG ──────────────────────────────────────────
OLLAMA_MODEL  = "llama3.1:8b"
DATASET_NAME  = "JaquanTW/fewshot-absaquad"
RESULTS_FILE  = "results/one_shot_results.json"
DEFAULT_N     = 50
TEMPERATURE   = 0.1
RANDOM_SEED   = 42

CATEGORIES = [
    "food quality", "food prices", "food style options",
    "service general", "ambience general",
    "restaurant general", "restaurant prices", "restaurant misc",
    "drinks quality", "drinks prices", "drinks style options",
    "location general",
]

# ─────────────────────────── PROMPT ──────────────────────────────────────────
SYSTEM_PROMPT = f"""You are an expert in Aspect-Based Sentiment Analysis (ABSA).

Extract ALL sentiment quadruplets from the restaurant review.

Each quadruplet contains:
  - Aspect   : the specific thing being discussed (e.g. food, service, price)
  - Category : one of: {", ".join(CATEGORIES)}
  - Opinion  : the opinion word/phrase (e.g. delicious, slow, expensive)
  - Sentiment: exactly one of: positive | negative | neutral

Output format — one quadruplet per line:
  Aspect: <text> | Category: <text> | Opinion: <text> | Sentiment: <text>

If nothing can be extracted, write: NONE
Do NOT add any explanation or extra text."""


def format_example(example: dict) -> str:
    """Format a training example into the expected output format."""
    lines = [f'Review: "{example["text"]}"', "Quadruplets:"]
    for q in example["quads"]:
        lines.append(
            f"Aspect: {q['aspect']} | Category: {q['category']} | "
            f"Opinion: {q['opinion']} | Sentiment: {q['sentiment']}"
        )
    return "\n".join(lines)


def build_user_prompt(review: str, example: dict) -> str:
    """Build the user message with ONE training example followed by the test review."""
    example_block = format_example(example)
    return (
        f"Here is one example:\n\n"
        f"{example_block}\n\n"
        f"---\n"
        f"Now extract quadruplets for this review:\n\n"
        f'Review: "{review}"\n\nExtract all quadruplets:'
    )


# ─────────────────────────── DATA ────────────────────────────────────────────
import ast

def clean_text(v):
    s = str(v).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            p = ast.literal_eval(s)
            return str(p[0]).strip() if isinstance(p, list) and p else s
        except Exception:
            pass
    return s


def load_split(split_name):
    """Load one split and group rows by review text."""
    raw = load_dataset(DATASET_NAME)[split_name].to_pandas()
    records = {}
    for _, row in raw.iterrows():
        text = clean_text(row["text"])
        if text not in records:
            records[text] = []
        records[text].append({
            "aspect":    clean_text(row["span"]).lower(),
            "category":  clean_text(row["ac"]).lower(),
            "opinion":   clean_text(row["ot"]).lower(),
            "sentiment": clean_text(row["label"]).lower(),
        })
    return [{"text": t, "quads": q} for t, q in records.items()]


def load_data(n_samples=None):
    print(f"Loading dataset: {DATASET_NAME} ...")
    train_reviews = load_split("train")
    test_reviews  = load_split("test")

    print(f"Train: {len(train_reviews)} unique reviews")
    print(f"Test : {len(test_reviews)} unique reviews")

    if n_samples:
        random.seed(RANDOM_SEED)
        test_reviews = random.sample(test_reviews, min(n_samples, len(test_reviews)))
        print(f"Sampled {n_samples} test reviews")

    # Pick ONE example from training set (same example used for all reviews)
    random.seed(RANDOM_SEED)
    one_shot_example = random.choice(train_reviews)
    print(f"\nOne-shot example: \"{one_shot_example['text'][:70]}...\"")
    print(f"  Quads: {one_shot_example['quads']}")

    return test_reviews, one_shot_example


# ─────────────────────────── PARSER ──────────────────────────────────────────
QUAD_RE = re.compile(
    r"[Aa]spect\s*:\s*(?P<aspect>.+?)\s*\|\s*"
    r"[Cc]ategory\s*:\s*(?P<category>.+?)\s*\|\s*"
    r"[Oo]pinion\s*:\s*(?P<opinion>.+?)\s*\|\s*"
    r"[Ss]entiment\s*:\s*(?P<sentiment>\w+)", re.IGNORECASE
)

def parse_output(text: str) -> list:
    if not text or text.strip().upper() == "NONE":
        return []
    quads = []
    for line in text.splitlines():
        line = re.sub(r"^\d+[\.\)]\s*|^[-*]\s*", "", line.strip())
        m = QUAD_RE.search(line)
        if m:
            quads.append({
                "aspect":    m.group("aspect").strip().lower(),
                "category":  m.group("category").strip().lower(),
                "opinion":   m.group("opinion").strip().lower(),
                "sentiment": m.group("sentiment").strip().lower(),
            })
    return quads


# ─────────────────────────── EVAL ────────────────────────────────────────────
def quad_f1(all_preds, all_golds):
    fields_map = {
        "quad":      ["aspect", "category", "opinion", "sentiment"],
        "aspect":    ["aspect"],
        "category":  ["category"],
        "opinion":   ["opinion"],
        "sentiment": ["sentiment"],
    }
    totals = {k: {"tp": 0, "pred": 0, "gold": 0} for k in fields_map}
    for preds, golds in zip(all_preds, all_golds):
        for name, fields in fields_map.items():
            def to_t(q): return tuple(q.get(f, "").strip().lower() for f in fields)
            ps = Counter(to_t(q) for q in preds)
            gs = Counter(to_t(q) for q in golds)
            tp = sum((ps & gs).values())
            totals[name]["tp"]   += tp
            totals[name]["pred"] += sum(ps.values())
            totals[name]["gold"] += sum(gs.values())
    results = {}
    for name, c in totals.items():
        p  = c["tp"] / c["pred"] if c["pred"] else 0
        r  = c["tp"] / c["gold"] if c["gold"] else 0
        f1 = 2*p*r/(p+r) if (p+r) else 0
        results[name] = {"precision": round(p,4), "recall": round(r,4), "f1": round(f1,4)}
    return results


def print_metrics(metrics, title="Results"):
    print(f"\n{'='*52}")
    print(f"  {title}")
    print(f"{'='*52}")
    print(f"  {'Metric':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*42}")
    for name, s in metrics.items():
        print(f"  {name:<12} {s['precision']:>10.4f} {s['recall']:>10.4f} {s['f1']:>10.4f}")
    print(f"{'='*52}")


# ─────────────────────────── MAIN ────────────────────────────────────────────
def run(n_samples=None):
    test_reviews, one_shot_example = load_data(n_samples)

    try:
        available = [m.model for m in ollama.list().models]
        if OLLAMA_MODEL not in available:
            print(f"ERROR: '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama.\n{e}")
        sys.exit(1)

    print(f"\nModel   : {OLLAMA_MODEL}")
    print(f"Strategy: ONE-SHOT (1 example shown to model)")
    print(f"Reviews : {len(test_reviews)}")
    print("-" * 52)

    all_preds, all_golds, output_records = [], [], []

    for i, item in enumerate(test_reviews):
        review     = item["text"]
        gold_quads = item["quads"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(review, one_shot_example)},
        ]

        t0 = time.time()
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": TEMPERATURE, "num_predict": 512},
        )
        elapsed = time.time() - t0
        raw_output = resp.message.content.strip()
        pred_quads = parse_output(raw_output)

        all_preds.append(pred_quads)
        all_golds.append(gold_quads)

        pct = (i + 1) / len(test_reviews)
        bar = "#" * int(pct * 30)
        print(f"\r  [{bar:<30}] {i+1}/{len(test_reviews)}  ({elapsed:.1f}s)", end="", flush=True)

        output_records.append({
            "review":     review,
            "gold":       gold_quads,
            "predicted":  pred_quads,
            "raw_output": raw_output,
            "elapsed_s":  round(elapsed, 2),
        })

    print()

    metrics = quad_f1(all_preds, all_golds)
    print_metrics(metrics, title=f"ONE-SHOT  |  {OLLAMA_MODEL}  |  {len(test_reviews)} reviews")

    os.makedirs("results", exist_ok=True)
    result = {
        "strategy":        "one-shot",
        "model":           OLLAMA_MODEL,
        "n_reviews":       len(test_reviews),
        "one_shot_example": one_shot_example,
        "metrics":         metrics,
        "records":         output_records,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {RESULTS_FILE}")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=DEFAULT_N)
    p.add_argument("--full",    action="store_true", help="Run on all test reviews")
    args = p.parse_args()
    run(n_samples=None if args.full else args.samples)
