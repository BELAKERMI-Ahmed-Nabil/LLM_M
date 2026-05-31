"""
run_zero_shot.py — llama3.1:8b
================================
ABSA Quad Extraction — ZERO-SHOT strategy.
No examples shown to the model.

Usage:
    python run_zero_shot.py              # 50 reviews
    python run_zero_shot.py --full       # all 778 reviews
    python run_zero_shot.py --samples 20
"""

import argparse, json, os, sys, time, re, ast, random
from collections import Counter
from difflib import SequenceMatcher

import ollama
from datasets import load_dataset

# ─────────────────────────── CONFIG ──────────────────────────────────────────
OLLAMA_MODEL  = "llama3.1:8b"
DATASET_NAME  = "JaquanTW/fewshot-absaquad"
DEFAULT_N     = 50
TEMPERATURE   = 0.1
RANDOM_SEED   = 42

CATEGORIES = [
    "food quality", "food prices", "food style_options",
    "service general", "ambience general",
    "restaurant general", "restaurant prices", "restaurant miscellaneous",
    "drinks quality", "drinks prices", "drinks style_options",
    "location general",
]

CATEGORY_MAP = {
    "food style options": "food style_options",
    "food style option":  "food style_options",
    "restaurant misc":    "restaurant miscellaneous",
    "drinks style options":"drinks style_options",
    "drink quality":      "drinks quality",
    "drink prices":       "drinks prices",
    "service":            "service general",
    "ambience":           "ambience general",
    "atmosphere":         "ambience general",
    "location":           "location general",
    "food price":         "food prices",
    "restaurant price":   "restaurant prices",
}

def normalize_category(cat):
    c = cat.strip().lower()
    if c in CATEGORY_MAP:
        return CATEGORY_MAP[c]
    best, best_score = c, 0.0
    for canonical in CATEGORIES:
        s = SequenceMatcher(None, c, canonical).ratio()
        if s > best_score:
            best_score, best = s, canonical
    return best if best_score > 0.6 else c

CAT_STR = "\n".join(f"  - {c}" for c in CATEGORIES)

SYSTEM_PROMPT = f"""You are an expert in Aspect-Based Sentiment Analysis (ABSA) for restaurant reviews.

Your task: Extract ALL sentiment quadruplets from the review.

Each quadruplet has exactly 4 fields:
  Aspect   : the specific item being evaluated (use general topic if implicit, e.g. "food", "service")
  Category : MUST be one of these exact strings:
{CAT_STR}
  Opinion  : the exact opinion word or phrase from the text
  Sentiment: MUST be exactly one of: positive | negative | neutral

Output format (one quadruplet per line, nothing else):
  Aspect: <text> | Category: <category> | Opinion: <text> | Sentiment: <label>

Rules:
  - Use EXACT category strings (e.g. "food style_options" not "food style options")
  - Output NONE if the review has no sentiment
  - Do NOT add explanations, numbers, or extra text"""

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

def load_data(n_samples=None):
    print(f"Loading {DATASET_NAME} ...")
    raw = load_dataset(DATASET_NAME)["test"].to_pandas()
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
    reviews = [{"text": t, "quads": q} for t, q in records.items()]
    print(f"Test: {len(reviews)} unique reviews")
    if n_samples:
        random.seed(RANDOM_SEED)
        reviews = random.sample(reviews, min(n_samples, len(reviews)))
        print(f"Sampled {len(reviews)} reviews")
    return reviews

# ─────────────────────────── PARSER ──────────────────────────────────────────
QUAD_RE = re.compile(
    r"[Aa]spect\s*:\s*(?P<aspect>.+?)\s*\|\s*"
    r"[Cc]ategory\s*:\s*(?P<category>.+?)\s*\|\s*"
    r"[Oo]pinion\s*:\s*(?P<opinion>.+?)\s*\|\s*"
    r"[Ss]entiment\s*:\s*(?P<sentiment>\w+)", re.IGNORECASE,
)

SENTIMENT_MAP = {"pos": "positive", "neg": "negative", "neu": "neutral"}

def parse_output(text):
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
                "sentiment": SENTIMENT_MAP.get(m.group("sentiment").strip().lower(),
                                               m.group("sentiment").strip().lower()),
            })
    return quads

# ─────────────────────────── EVAL ────────────────────────────────────────────
def lcs_score(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    m = SequenceMatcher(None, a.lower().split(), b.lower().split())
    l = sum(blk.size for blk in m.get_matching_blocks())
    return 2 * l / (len(a.split()) + len(b.split()))

def partial_score(p, g):
    return (
        (1.0 if normalize_category(p["category"]) == normalize_category(g["category"]) else 0.0) +
        (1.0 if p["sentiment"] == g["sentiment"] else 0.0) +
        lcs_score(p["aspect"],  g["aspect"]) +
        lcs_score(p["opinion"], g["opinion"])
    ) / 4.0

def hungarian(preds, golds):
    if not preds or not golds: return 0.0, len(preds), len(golds)
    scores = [[partial_score(p, g) for g in golds] for p in preds]
    mp, mg, total = set(), set(), 0.0
    for s, i, j in sorted([(scores[i][j], i, j)
                            for i in range(len(preds))
                            for j in range(len(golds))], reverse=True):
        if i not in mp and j not in mg:
            total += s; mp.add(i); mg.add(j)
    return total, len(preds), len(golds)

def quad_f1_exact(all_preds, all_golds):
    fields_map = {
        "quad":      ["aspect", "category", "opinion", "sentiment"],
        "aspect":    ["aspect"], "category": ["category"],
        "opinion":   ["opinion"], "sentiment": ["sentiment"],
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

def quad_f1_partial(all_preds, all_golds):
    ts, tp, tg = 0.0, 0, 0
    for preds, golds in zip(all_preds, all_golds):
        s, np_, ng = hungarian(preds, golds)
        ts += s; tp += np_; tg += ng
    p  = ts/tp if tp else 0
    r  = ts/tg if tg else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return {"precision": round(p,4), "recall": round(r,4), "f1": round(f1,4)}

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
    print(f"  llama3.2 Zero-Shot Partial F1 = 0.5081  (for reference)")
    print(f"  SetFitQuad best Partial F1    = 0.4250  (for reference)")
    print(f"{'='*60}")

# ─────────────────────────── MAIN ────────────────────────────────────────────
def run(n_samples=None):
    reviews = load_data(n_samples)

    try:
        available = [m.model for m in ollama.list().models]
        if OLLAMA_MODEL not in available:
            print(f"ERROR: '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama.\n{e}"); sys.exit(1)

    print(f"\nModel   : {OLLAMA_MODEL}")
    print(f"Strategy: ZERO-SHOT")
    print(f"Reviews : {len(reviews)}")
    print("-" * 60)

    all_preds, all_golds, output_records = [], [], []

    for i, item in enumerate(reviews):
        review     = item["text"]
        gold_quads = item["quads"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f'Review: "{review}"\nExtract all quadruplets:'},
        ]

        t0 = time.time()
        resp = ollama.chat(model=OLLAMA_MODEL, messages=messages,
                           options={"temperature": TEMPERATURE, "num_predict": 512})
        elapsed = time.time() - t0
        raw_output = resp.message.content.strip()
        pred_quads = parse_output(raw_output)

        all_preds.append(pred_quads)
        all_golds.append(gold_quads)

        pct = (i + 1) / len(reviews)
        bar = "#" * int(pct * 30)
        print(f"\r  [{bar:<30}] {i+1}/{len(reviews)}  ({elapsed:.1f}s)", end="", flush=True)

        output_records.append({
            "review": review, "gold": gold_quads,
            "predicted": pred_quads, "raw_output": raw_output,
            "elapsed_s": round(elapsed, 2),
        })

    print()
    exact   = quad_f1_exact(all_preds, all_golds)
    partial = quad_f1_partial(all_preds, all_golds)
    print_metrics(exact, partial, title=f"ZERO-SHOT | {OLLAMA_MODEL} | {len(reviews)} reviews")

    os.makedirs("results", exist_ok=True)
    result = {
        "strategy": "zero-shot", "model": OLLAMA_MODEL,
        "n_reviews": len(reviews),
        "metrics_exact":   exact,
        "metrics_partial": {"quad": partial},
        "records": output_records,
    }
    with open("results/zero_shot_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> results/zero_shot_results.json")
    return exact, partial

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=DEFAULT_N)
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    run(n_samples=None if args.full else args.samples)
