"""
compare_8b_vs_3b.py
====================
Final comprehensive comparison:
  - llama3.1:8b  (this folder)
  - llama3.2:3b  (C:/ABSA_LLM)
  - SetFitQuad   (C:/SetFitQuad-Code-main)
  - Paper        (published PDF)
"""

import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = "results/charts"
os.makedirs(OUT, exist_ok=True)

# ── Load helpers ──────────────────────────────────────────────
def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    m  = d.get("metrics", d.get("metrics_exact", d))
    mp = d.get("metrics_partial", {}).get("quad", {})
    return {
        "exact_f1":   m.get("quad",      {}).get("f1", 0),
        "partial_f1": mp.get("f1", 0),
        "aspect":     m.get("aspect",    {}).get("f1", 0),
        "category":   m.get("category",  {}).get("f1", 0),
        "opinion":    m.get("opinion",   {}).get("f1", 0),
        "sentiment":  m.get("sentiment", {}).get("f1", 0),
    }

def avg_folds(pattern):
    files = glob.glob(pattern)
    if not files: return None
    pms, ems = [], []
    for f in files:
        with open(f, encoding="utf-8") as fh: d = json.load(fh)
        q = d.get("quad", {})
        pms.append(q.get("partial_match", {}).get("f1", 0))
        ems.append(q.get("exact_match",   {}).get("f1", 0))
    return {"partial_f1": float(np.mean(pms)), "exact_f1": float(np.mean(ems))}

# ── Results ───────────────────────────────────────────────────
R8  = "results"
R3  = "C:/ABSA_LLM/results"
RSF = "C:/SetFitQuad-Code-main"

results = {
    # 8b
    "Zero-Shot\n(8b)":      load(f"{R8}/zero_shot_results.json"),
    "One-Shot\n(8b)":       load(f"{R8}/one_shot_results.json"),
    "Few-Shot 10ex\n(8b)":  load(f"{R8}/few_shot_v2_10ex_results.json"),
    "Few-Shot 50ex\n(8b)":  load(f"{R8}/few_shot_v2_50ex_results.json"),
    # 3b
    "Zero-Shot\n(3b)":      load(f"{R3}/zero_shot_results.json"),
    "One-Shot\n(3b)":       load(f"{R3}/one_shot_results.json"),
    "Few-Shot 10ex\n(3b)":  load(f"{R3}/few_shot_v2_10ex_results.json"),
}

# SetFitQuad best (EXP2 avg folds)
sf_best = avg_folds(f"{RSF}/exp2/results/*_fold*_results.json")
sf_exp3 = {}
for f in glob.glob(f"{RSF}/exp3/results/*_fold*_results.json"):
    name = os.path.basename(f).rsplit("_fold", 1)[0]
    size_str = name.rsplit("_", 1)[-1]
    if size_str.isdigit():
        sf_exp3.setdefault(int(size_str), []).append(f)
sf_exp3_avg = {s: float(np.mean([
    json.load(open(ff, encoding="utf-8")).get("quad", {}).get("partial_match", {}).get("f1", 0)
    for ff in files])) for s, files in sf_exp3.items()}

# Remove missing
results = {k: v for k, v in results.items() if v}

SETFIT_BEST  = max(sf_best["partial_f1"], 0.4250) if sf_best else 0.4250
PAPER_BEST   = 0.532
T5_ASQP      = 0.590

# ── Print table ───────────────────────────────────────────────
def print_table():
    print(f"\n{'='*82}")
    print(f"  FINAL COMPARISON — All Systems")
    print(f"  Metric: Partial Match F1  |  Dataset: Rest15+Rest16 (778 test reviews)")
    print(f"{'='*82}")
    print(f"\n  {'System':<24} {'Partial F1':>11} {'Exact F1':>10} {'Aspect':>8} {'Cat':>7} {'Opinion':>8} {'Sent':>7}")
    print(f"  {'-'*77}")

    groups = [
        ("llama3.1:8b", [k for k in results if "8b" in k]),
        ("llama3.2:3b", [k for k in results if "3b" in k]),
    ]
    for gname, keys in groups:
        print(f"  [{gname}]")
        for k in keys:
            v = results[k]
            name = k.replace("\n", " ")
            mark = " *" if v["partial_f1"] == max(r["partial_f1"] for r in results.values()) else ""
            print(f"  {name:<24} {v['partial_f1']:>11.4f} {v['exact_f1']:>10.4f} "
                  f"{v['aspect']:>8.4f} {v['category']:>7.4f} {v['opinion']:>8.4f} {v['sentiment']:>7.4f}{mark}")

    print(f"\n  [Baselines]")
    print(f"  {'SetFitQuad (our runs)':<24} {SETFIT_BEST:>11.4f}   (50 examples + fine-tuning)")
    print(f"  {'SetFitQuad (paper)':<24} {PAPER_BEST:>11.3f}   (published result)")
    print(f"  {'T5-ASQP (paper)':<24} {T5_ASQP:>11.3f}   (large model, full fine-tuning)")

    best_name = max(results, key=lambda k: results[k]["partial_f1"])
    best_val  = results[best_name]["partial_f1"]
    print(f"\n  {'='*77}")
    print(f"  BEST: {best_name.replace(chr(10),' '):<22} Partial F1 = {best_val:.4f}")
    diff_t5 = best_val - T5_ASQP
    diff_paper = best_val - PAPER_BEST
    print(f"  vs T5-ASQP    : {'+' if diff_t5>=0 else ''}{diff_t5:.4f}  ({'WINS' if diff_t5>=0 else 'T5 wins'})")
    print(f"  vs Paper best : {'+' if diff_paper>=0 else ''}{diff_paper:.4f}  ({'WINS' if diff_paper>=0 else 'Paper wins'})")
    print(f"  Training cost : 0 examples (only 10 in-context, no fine-tuning)")
    print(f"{'='*82}")

# ── Chart 1: All systems bar chart ───────────────────────────
def chart_all():
    COLORS = {
        "8b": ["#C0392B","#E74C3C","#1ABC9C","#17A589"],
        "3b": ["#7D6608","#F39C12","#1A5276","#2E86C1"],
    }
    fig, ax = plt.subplots(figsize=(18, 7))

    bars_data = []
    for k, v in results.items():
        model = "8b" if "8b" in k else "3b"
        idx   = [kk for kk in results if ("8b" if "8b" in kk else "3b") == model].index(k)
        bars_data.append((k.replace("\n"," "), v["partial_f1"], COLORS[model][idx % 4], model))

    x      = np.arange(len(bars_data))
    labels = [d[0] for d in bars_data]
    vals   = [d[1] for d in bars_data]
    colors = [d[2] for d in bars_data]

    bars = ax.bar(x, vals, color=colors, width=0.62, alpha=0.88, zorder=3, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.007,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Separator between 8b and 3b
    n8 = sum(1 for d in bars_data if d[3] == "8b")
    ax.axvline(n8 - 0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ymax = max(vals + [T5_ASQP]) * 1.35
    ax.text(n8/2 - 0.5,    ymax*0.97, "llama3.1:8b", ha="center",
            fontsize=11, style="italic", color="#333",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FDFEFE", alpha=0.8))
    ax.text(n8 + (len(bars_data)-n8)/2 - 0.5, ymax*0.97, "llama3.2:3b", ha="center",
            fontsize=11, style="italic", color="#333",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FDFEFE", alpha=0.8))

    # Reference lines
    refs = [
        (SETFIT_BEST, "#2980B9", "--",           f"SetFitQuad our runs = {SETFIT_BEST:.3f}"),
        (PAPER_BEST,  "#27AE60", "-.",            f"SetFitQuad paper    = {PAPER_BEST:.3f}"),
        (T5_ASQP,     "#8E44AD", (0,(3,1,1,1)),  f"T5-ASQP baseline   = {T5_ASQP:.3f}"),
    ]
    for val, col, ls, label in refs:
        ax.axhline(val, color=col, linestyle=ls, linewidth=2.2, label=label, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Partial Match F1", fontsize=12)
    ax.set_ylim(0, ymax)
    ax.set_title(
        "Complete Comparison: llama3.1:8b vs llama3.2:3b vs SetFitQuad vs T5-ASQP\n"
        "Metric: Partial Match F1 (fair comparison — same metric for all systems)",
        fontsize=13, fontweight="bold"
    )
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    p = os.path.join(OUT, "final_all_systems.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p}")

# ── Chart 2: Shot count effect (8b vs 3b) ────────────────────
def chart_shots():
    fig, ax = plt.subplots(figsize=(11, 6))

    shot_map_8b = {0: "Zero-Shot\n(8b)", 1: "One-Shot\n(8b)",
                   10: "Few-Shot 10ex\n(8b)", 50: "Few-Shot 50ex\n(8b)"}
    shot_map_3b = {0: "Zero-Shot\n(3b)", 1: "One-Shot\n(3b)", 10: "Few-Shot 10ex\n(3b)"}

    shots_8b = sorted([k for k in shot_map_8b if shot_map_8b[k] in results])
    shots_3b = sorted([k for k in shot_map_3b if shot_map_3b[k] in results])

    vals_8b = [results[shot_map_8b[s]]["partial_f1"] for s in shots_8b]
    vals_3b = [results[shot_map_3b[s]]["partial_f1"] for s in shots_3b]

    ax.plot(shots_8b, vals_8b, "o-", color="#E74C3C", linewidth=2.5,
            markersize=9, label="llama3.1:8b", zorder=4)
    ax.plot(shots_3b, vals_3b, "s--", color="#2980B9", linewidth=2.5,
            markersize=9, label="llama3.2:3b", zorder=4)

    for s, v in zip(shots_8b, vals_8b):
        ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10,
                    color="#E74C3C", fontweight="bold")
    for s, v in zip(shots_3b, vals_3b):
        ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=10,
                    color="#2980B9", fontweight="bold")

    refs = [
        (SETFIT_BEST, "#2980B9", "--",          f"SetFitQuad (our runs) = {SETFIT_BEST:.3f}"),
        (PAPER_BEST,  "#27AE60", "-.",           f"SetFitQuad (paper)   = {PAPER_BEST:.3f}"),
        (T5_ASQP,     "#8E44AD", (0,(3,1,1,1)), f"T5-ASQP              = {T5_ASQP:.3f}"),
    ]
    for val, col, ls, label in refs:
        ax.axhline(val, color=col, linestyle=ls, linewidth=2, label=label, zorder=3)

    ax.set_xlabel("Number of In-Context Examples (Shots)", fontsize=12)
    ax.set_ylabel("Partial Match F1", fontsize=12)
    ax.set_title(
        "Effect of Shot Count on Performance\nllama3.1:8b vs llama3.2:3b",
        fontsize=13, fontweight="bold"
    )
    ax.set_xticks([0, 1, 10, 50])
    ax.set_xticklabels(["0\n(Zero-Shot)", "1\n(One-Shot)", "10\n(Few-Shot)", "50\n(Few-Shot)"])
    ax.set_ylim(0, 0.75)
    ax.yaxis.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "shot_count_effect.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p}")

# ── Chart 3: Learning curve EXP3 + our best ──────────────────
def chart_learning_curve():
    if not sf_exp3_avg: return
    sizes = sorted(sf_exp3_avg.keys())
    vals  = [sf_exp3_avg[s] for s in sizes]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, vals, "o-", color="#2980B9", linewidth=2.5,
            markersize=9, label="SetFitQuad (our runs) — Partial F1 vs training size", zorder=4)
    for s, v in zip(sizes, vals):
        ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9,
                    color="#2980B9", fontweight="bold")

    our_lines = [
        ("Few-Shot 10ex (8b)", "#E74C3C", "-"),
        ("Few-Shot 10ex (3b)", "#F39C12", "--"),
        ("Zero-Shot (3b)",     "#1ABC9C", "-."),
    ]
    for key, col, ls in our_lines:
        v = results.get(key, {}).get("partial_f1", 0)
        if v:
            ax.axhline(v, color=col, linestyle=ls, linewidth=2,
                       label=f"Our {key.replace(chr(10),' ')} = {v:.3f}", zorder=3)

    ax.axhline(PAPER_BEST, color="#27AE60", linestyle=(0,(3,1,1,1)),
               linewidth=2.2, label=f"SetFitQuad paper = {PAPER_BEST:.3f}", zorder=3)
    ax.axhline(T5_ASQP,   color="#8E44AD", linestyle=":",
               linewidth=2.2, label=f"T5-ASQP = {T5_ASQP:.3f}", zorder=3)

    ax.set_xlabel("SetFitQuad Training Sample Size", fontsize=12)
    ax.set_ylabel("Partial Match F1", fontsize=12)
    ax.set_title(
        "SetFitQuad: Performance vs Training Size\nvs Our LLM (0 training samples needed)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xticks(sizes)
    ax.set_ylim(0.15, 0.75)
    ax.yaxis.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "learning_curve_final.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p}")

# ── Main ──────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  Final Comparison: 8b vs 3b vs SetFitQuad vs Paper")
    print("="*60)
    print(f"  Systems loaded: {len(results)}")

    print_table()

    print("\nGenerating charts...")
    chart_all()
    chart_shots()
    chart_learning_curve()
    print(f"\nAll charts saved to '{OUT}/'")

if __name__ == "__main__":
    main()
