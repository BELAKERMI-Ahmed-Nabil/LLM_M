"""
run_few_shot_v2.py
==================
ABSA Quad Extraction -- FEW-SHOT (improved version).

Improvements over v1:
  1. Fixed category names to match gold labels exactly
  2. Category normalization in parser (handles model variations)
  3. Improved prompt with category examples and implicit aspect guidance
  4. Partial Match F1 evaluation (same metric as SetFitQuad)
  5. Smarter few-shot example selection (covers all categories)

Usage:
    python run_few_shot_v2.py               # 50 reviews
    python run_few_shot_v2.py --full        # all 778 reviews
    python run_few_shot_v2.py --shots 15
"""

import argparse, json, os, sys, time, re, random
from collections import Counter
from difflib import SequenceMatcher
import ast

import ollama
from datasets import load_dataset

# ─────────────────────────── CONFIG ──────────────────────────────────────────
OLLAMA_MODEL  = "llama3.1:8b"
DATASET_NAME  = "JaquanTW/fewshot-absaquad"
DEFAULT_N     = 50
DEFAULT_SHOTS = 10
TEMPERATURE   = 0.1
RANDOM_SEED   = 42

# FIXED: exact category names matching the gold labels
CATEGORIES = [
    "food quality",
    "food prices",
    "food style_options",
    "service general",
    "ambience general",
    "restaurant general",
    "restaurant prices",
    "restaurant miscellaneous",
    "drinks quality",
    "drinks prices",
    "drinks style_options",
    "location general",
]

# Map common model output variations -> canonical gold labels
CATEGORY_MAP = {
    "food style options":         "food style_options",
    "food style option":          "food style_options",
    "food styles options":        "food style_options",
    "food options":               "food style_options",
    "restaurant misc":            "restaurant miscellaneous",
    "restaurant miscellaneous":   "restaurant miscellaneous",
    "drinks style options":       "drinks style_options",
    "drinks style option":        "drinks style_options",
    "drink quality":              "drinks quality",
    "drink prices":               "drinks prices",
    "drink style_options":        "drinks style_options",
    "food price":                 "food prices",
    "restaurant price":           "restaurant prices",
    "drinks price":               "drinks prices",
    "service":                    "service general",
    "ambience":                   "ambience general",
    "atmosphere":                 "ambience general",
    "location":                   "location general",
}

def normalize_category(cat: str) -> str:
    c = cat.strip().lower()
    if c in CATEGORY_MAP:
        return CATEGORY_MAP[c]
    # fuzzy: find closest canonical
    best, best_score = c, 0.0
    for canonical in CATEGORIES:
        s = SequenceMatcher(None, c, canonical).ratio()
        if s > best_score:
            best_score, best = s, canonical
    return best if best_score > 0.6 else c

# ─────────────────────────── PROMPT ──────────────────────────────────────────
CAT_STR = "\n".join(f"  - {c}" for c in CATEGORIES)

SYSTEM_PROMPT = f"""You are an expert in Aspect-Based Sentiment Analysis (ABSA) for restaurant reviews.

Your task: Extract ALL sentiment quadruplets from the review.

Each quadruplet has exactly 4 fields:
  Aspect   : the specific item/thing being evaluated (use "restaurant" if no explicit mention)
  Category : MUST be one of these exact strings:
{CAT_STR}
  Opinion  : the exact opinion word or phrase from the text
  Sentiment: MUST be exactly one of: positive | negative | neutral

Output format (one quadruplet per line, nothing else):
  Aspect: <text> | Category: <category> | Opinion: <text> | Sentiment: <label>

Rules:
  - Use the EXACT category strings listed above (e.g. "food style_options" not "food style options")
  - If no aspect is mentioned explicitly, write the general topic (e.g. "restaurant", "food", "service")
  - Output NONE if the review has no sentiment
  - Do NOT add explanations, numbers, or extra text"""


def format_example(example: dict) -> str:
    lines = [f'Review: "{example["text"]}"', "Output:"]
    for q in example["quads"]:
        lines.append(
            f"Aspect: {q['aspect']} | Category: {q['category']} | "
            f"Opinion: {q['opinion']} | Sentiment: {q['sentiment']}"
        )
    return "\n".join(lines)


def build_user_prompt(review: str, examples: list) -> str:
    blocks = [format_example(ex) for ex in examples]
    examples_text = "\n\n".join(blocks)
    return (
        f"Here are {len(examples)} labeled examples:\n\n"
        f"{examples_text}\n\n"
        f"---\n"
        f"Now extract all quadruplets for this review:\n\n"
        f'Review: "{review}"\nOutput:'
    )


# ─────────────────────────── DATA ────────────────────────────────────────────
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


def select_diverse_examples(train_reviews, n_shots):
    """Select examples that cover as many categories as possible."""
    random.seed(RANDOM_SEED)
    covered = set()
    selected = []
    shuffled = train_reviews[:]
    random.shuffle(shuffled)

    for item in shuffled:
        cats = {q["category"] for q in item["quads"]}
        new_cats = cats - covered
        if new_cats or len(selected) < n_shots // 2:
            selected.append(item)
            covered.update(cats)
        if len(selected) >= n_shots:
            break

    # fill remaining slots randomly
    remaining = [x for x in shuffled if x not in selected]
    while len(selected) < n_shots and remaining:
        selected.append(remaining.pop(0))

    return selected[:n_shots]


def load_data(n_samples=None, n_shots=DEFAULT_SHOTS):
    print(f"Loading {DATASET_NAME} ...")
    train_reviews = load_split("train")
    test_reviews  = load_split("test")
    print(f"Train: {len(train_reviews)} unique reviews")
    print(f"Test : {len(test_reviews)} unique reviews")

    if n_samples:
        random.seed(RANDOM_SEED)
        test_reviews = random.sample(test_reviews, min(n_samples, len(test_reviews)))
        print(f"Sampled {len(test_reviews)} test reviews")

    examples = select_diverse_examples(train_reviews, n_shots)
    cats_covered = {q["category"] for ex in examples for q in ex["quads"]}
    print(f"\nFew-shot examples: {len(examples)} | Categories covered: {len(cats_covered)}/{len(CATEGORIES)}")
    return test_reviews, examples


# ─────────────────────────── PARSER ──────────────────────────────────────────
QUAD_RE = re.compile(
    r"[Aa]spect\s*:\s*(?P<aspect>.+?)\s*\|\s*"
    r"[Cc]ategory\s*:\s*(?P<category>.+?)\s*\|\s*"
    r"[Oo]pinion\s*:\s*(?P<opinion>.+?)\s*\|\s*"
    r"[Ss]entiment\s*:\s*(?P<sentiment>\w+)",
    re.IGNORECASE,
)

SENTIMENT_MAP = {
    "pos": "positive", "neg": "negative", "neu": "neutral",
    "1": "positive", "-1": "negative", "0": "neutral",
    "great": "positive", "bad": "negative",
}

def normalize_sentiment(s: str) -> str:
    s = s.strip().lower()
    return SENTIMENT_MAP.get(s, s)


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
                "category":  normalize_category(m.group("category")),
                "opinion":   m.group("opinion").strip().lower(),
                "sentiment": normalize_sentiment(m.group("sentiment")),
            })
    return quads


# ─────────────────────────── EVAL — EXACT MATCH ──────────────────────────────
def quad_f1_exact(all_preds, all_golds):
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


# ─────────────────────────── EVAL — PARTIAL MATCH ────────────────────────────
def lcs_score(a: str, b: str) -> float:
    """LCS-based similarity (same as SetFitQuad)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    m = SequenceMatcher(None, a.lower().split(), b.lower().split())
    lcs_len = sum(block.size for block in m.get_matching_blocks())
    return 2 * lcs_len / (len(a.split()) + len(b.split()))


def quad_partial_score(pred: dict, gold: dict) -> float:
    """Score one pred-gold pair using partial match (same formula as SetFitQuad)."""
    cat_s  = 1.0 if pred["category"]  == gold["category"]  else 0.0
    sent_s = 1.0 if pred["sentiment"] == gold["sentiment"] else 0.0
    asp_s  = lcs_score(pred["aspect"],   gold["aspect"])
    op_s   = lcs_score(pred["opinion"],  gold["opinion"])
    return (cat_s + sent_s + asp_s + op_s) / 4.0


def hungarian_match(preds, golds):
    """Greedy Hungarian-style matching to maximise total score."""
    if not preds or not golds:
        return 0.0, len(preds), len(golds)

    scores = [[quad_partial_score(p, g) for g in golds] for p in preds]
    matched_p, matched_g = set(), set()
    total_score = 0.0

    flat = sorted(
        [(scores[i][j], i, j) for i in range(len(preds)) for j in range(len(golds))],
        reverse=True,
    )
    for score, i, j in flat:
        if i not in matched_p and j not in matched_g:
            total_score += score
            matched_p.add(i)
            matched_g.add(j)

    return total_score, len(preds), len(golds)


def quad_f1_partial(all_preds, all_golds):
    total_score, total_pred, total_gold = 0.0, 0, 0
    for preds, golds in zip(all_preds, all_golds):
        s, np_, ng = hungarian_match(preds, golds)
        total_score += s
        total_pred  += np_
        total_gold  += ng
    p  = total_score / total_pred if total_pred else 0
    r  = total_score / total_gold if total_gold else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return {"precision": round(p,4), "recall": round(r,4), "f1": round(f1,4)}


# ─────────────────────────── PRINT ───────────────────────────────────────────
def print_metrics(exact, partial, title="Results"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'Metric':<14} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*44}")
    for name, s in exact.items():
        print(f"  {name:<14} {s['precision']:>10.4f} {s['recall']:>10.4f} {s['f1']:>10.4f}")
    print(f"  {'-'*44}")
    pm = partial
    print(f"  {'partial_quad':<14} {pm['precision']:>10.4f} {pm['recall']:>10.4f} {pm['f1']:>10.4f}")
    print(f"{'='*60}")
    print(f"  SetFitQuad best Partial F1 = 0.4250  (for reference)")
    print(f"{'='*60}")


# ─────────────────────────── MAIN ────────────────────────────────────────────
def run(n_samples=None, n_shots=DEFAULT_SHOTS):
    test_reviews, examples = load_data(n_samples, n_shots)

    try:
        available = [m.model for m in ollama.list().models]
        if OLLAMA_MODEL not in available:
            print(f"ERROR: '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama.\n{e}")
        sys.exit(1)

    print(f"\nModel   : {OLLAMA_MODEL}")
    print(f"Strategy: FEW-SHOT v2 ({n_shots} examples)")
    print(f"Reviews : {len(test_reviews)}")
    print("-" * 60)

    all_preds, all_golds, output_records = [], [], []

    for i, item in enumerate(test_reviews):
        review     = item["text"]
        gold_quads = item["quads"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(review, examples)},
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

    exact   = quad_f1_exact(all_preds, all_golds)
    partial = quad_f1_partial(all_preds, all_golds)

    title = f"FEW-SHOT v2 ({n_shots} ex) | {OLLAMA_MODEL} | {len(test_reviews)} reviews"
    print_metrics(exact, partial, title=title)

    results_file = f"results/few_shot_v2_{n_shots}ex_results.json"
    os.makedirs("results", exist_ok=True)
    result = {
        "strategy":      f"few-shot-v2-{n_shots}",
        "model":         OLLAMA_MODEL,
        "n_reviews":     len(test_reviews),
        "n_shots":       n_shots,
        "metrics_exact": exact,
        "metrics_partial": {"quad": partial},
        "records":       output_records,
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {results_file}")
    return exact, partial


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=DEFAULT_N)
    p.add_argument("--shots",   type=int, default=DEFAULT_SHOTS)
    p.add_argument("--full",    action="store_true")
    args = p.parse_args()
    run(n_samples=None if args.full else args.samples, n_shots=args.shots)
