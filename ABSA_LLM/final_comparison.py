"""
final_comparison.py
===================
Complete comparison between:
  1. Our LLM System (Zero/One/Few-Shot) - C:\ABSA_LLM
  2. SetFitQuad (our runs)              - C:\SetFitQuad-Code-main
  3. SetFitQuad Paper results           - from published PDF

All systems evaluated on Rest15+Rest16 dataset (778 test reviews).
Metric used for comparison: Partial Match F1 (LCS-based, Hungarian matching)
"""

import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

OUT        = "results/charts"
SETFIT_DIR = "C:/SetFitQuad-Code-main"
os.makedirs(OUT, exist_ok=True)


# ════════════════════════════════════════════════════════════
# 1. LOAD OUR LLM RESULTS
# ════════════════════════════════════════════════════════════
def load_llm():
    files = {
        "Zero-Shot":     "results/zero_shot_results.json",
        "One-Shot":      "results/one_shot_results.json",
        "Few-Shot v1":   "results/few_shot_10ex_results.json",
        "Few-Shot v2":   "results/few_shot_v2_10ex_results.json",
    }
    out = {}
    for label, path in files.items():
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        m  = d.get("metrics", d.get("metrics_exact", d))
        mp = d.get("metrics_partial", {}).get("quad", {})
        out[label] = {
            "exact_f1":   m["quad"]["f1"],
            "partial_f1": mp.get("f1", 0.0),
            "aspect":     m["aspect"]["f1"],
            "category":   m["category"]["f1"],
            "opinion":    m["opinion"]["f1"],
            "sentiment":  m["sentiment"]["f1"],
        }
    return out


# ════════════════════════════════════════════════════════════
# 2. LOAD SETFITQUAD-CODE-MAIN (OUR RUNS)
# ════════════════════════════════════════════════════════════
def load_sf_exp1():
    """EXP1: 19 sampling strategies — test results (single run)."""
    res = {}
    # Try 5fold average files first
    for f in glob.glob(f"{SETFIT_DIR}/results/*_5fold_average_results.json"):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        name = os.path.basename(f).replace("_5fold_average_results.json", "")
        q = d.get("quad", {})
        res[name] = {
            "exact_f1":   q.get("exact_match",  {}).get("f1", 0),
            "partial_f1": q.get("partial_match", {}).get("f1", 0),
            "source": "5fold_avg",
        }
    # Fallback: fold files averaged
    if not res:
        fold_map = {}
        for f in glob.glob(f"{SETFIT_DIR}/results/*_fold*_results.json"):
            key = os.path.basename(f).rsplit("_fold", 1)[0]
            fold_map.setdefault(key, []).append(f)
        for name, files in fold_map.items():
            ems, pms = [], []
            for ff in files:
                with open(ff, encoding="utf-8") as fh:
                    d = json.load(fh)
                q = d.get("quad", {})
                ems.append(q.get("exact_match",  {}).get("f1", 0))
                pms.append(q.get("partial_match", {}).get("f1", 0))
            res[name] = {
                "exact_f1":   float(np.mean(ems)),
                "partial_f1": float(np.mean(pms)),
                "source": "fold_avg",
            }
    return res


def load_sf_exp2():
    """EXP2: 5 pretrained models — 5-fold averaged."""
    fold_map = {}
    for f in glob.glob(f"{SETFIT_DIR}/exp2/results/*_fold*_results.json"):
        model = os.path.basename(f).rsplit("_fold", 1)[0]
        fold_map.setdefault(model, []).append(f)

    MODEL_LABELS = {
        "all-MiniLM-L6-v2":                   "MiniLM-L6",
        "all-mpnet-base-v2":                   "mpnet-base",
        "multi-qa-mpnet-base-cos-v1":          "multi-qa-mpnet",
        "paraphrase-multilingual-mpnet-base-v2": "multilingual-mpnet",
        "paraphrase-TinyBERT-L6-v2":           "TinyBERT-L6",
    }

    res = {}
    for model, files in fold_map.items():
        ems, pms = [], []
        for ff in files:
            with open(ff, encoding="utf-8") as fh:
                d = json.load(fh)
            q = d.get("quad", {})
            ems.append(q.get("exact_match",  {}).get("f1", 0))
            pms.append(q.get("partial_match", {}).get("f1", 0))
        label = MODEL_LABELS.get(model, model[:22])
        res[label] = {
            "exact_f1":   float(np.mean(ems)),
            "partial_f1": float(np.mean(pms)),
        }
    return res


def load_sf_exp3():
    """EXP3: sample size effect — 5-fold averaged per size."""
    fold_map = {}
    for f in glob.glob(f"{SETFIT_DIR}/exp3/results/*_fold*_results.json"):
        name = os.path.basename(f).rsplit("_fold", 1)[0]
        fold_map.setdefault(name, []).append(f)
    res = {}
    for name, files in fold_map.items():
        pms = []
        for ff in files:
            with open(ff, encoding="utf-8") as fh:
                d = json.load(fh)
            pms.append(d.get("quad", {}).get("partial_match", {}).get("f1", 0))
        size_str = name.rsplit("_", 1)[-1]
        if size_str.isdigit():
            res[int(size_str)] = float(np.mean(pms))
    return res


# ════════════════════════════════════════════════════════════
# 3. PAPER RESULTS (SetFitQuad-others PDF)
# ════════════════════════════════════════════════════════════
PAPER_RESULTS = {
    # From Table 5 in the published paper (Partial Match F1, best configuration)
    "SetFitQuad\n(paper, CS)":    {"partial_f1": 0.532, "exact_f1": 0.083},
    "T5-ASQP\n(paper baseline)": {"partial_f1": 0.590, "exact_f1": 0.188},
}

# Paper subtask F1 (Table 6): ACD, ASC, ATE, OTE
PAPER_SUBTASKS = {
    "SetFitQuad (paper)": {"ACD": 0.575, "ASC": 0.826, "ATE": 0.533, "OTE": 0.333},
    "T5-ASQP (baseline)": {"ACD": 0.379, "ASC": 0.514, "ATE": 0.589, "OTE": 0.280},
    "SetFitABSA":          {"ACD": 0.000, "ASC": 0.774, "ATE": 0.000, "OTE": 0.000},
}


# ════════════════════════════════════════════════════════════
# CHART 1: Grand Overview — All Systems, Same Metric
# ════════════════════════════════════════════════════════════
def chart1_grand_overview(llm, sf1, sf2):
    fig, ax = plt.subplots(figsize=(20, 8))

    groups = []

    # --- Our LLM ---
    LLM_COLORS = ["#E74C3C", "#F39C12", "#2ECC71", "#1ABC9C"]
    for i, (name, v) in enumerate(llm.items()):
        groups.append({
            "label": name, "partial": v["partial_f1"],
            "exact": v["exact_f1"], "color": LLM_COLORS[i],
            "hatch": "", "group": "LLM"
        })

    # --- SetFitQuad our runs (top 5 EXP1 + top 3 EXP2) ---
    top5 = sorted(sf1.items(), key=lambda x: -x[1]["partial_f1"])[:5]
    for name, v in top5:
        groups.append({
            "label": name[:18], "partial": v["partial_f1"],
            "exact": v["exact_f1"], "color": "#2980B9",
            "hatch": "//", "group": "SetFit-EXP1"
        })
    top3 = sorted(sf2.items(), key=lambda x: -x[1]["partial_f1"])[:3]
    for name, v in top3:
        groups.append({
            "label": name, "partial": v["partial_f1"],
            "exact": v["exact_f1"], "color": "#7D3C98",
            "hatch": "xx", "group": "SetFit-EXP2"
        })

    # --- Paper results ---
    for name, v in PAPER_RESULTS.items():
        groups.append({
            "label": name, "partial": v["partial_f1"],
            "exact": v["exact_f1"], "color": "#27AE60",
            "hatch": "**", "group": "Paper"
        })

    x      = np.arange(len(groups))
    labels = [g["label"] for g in groups]
    partials = [g["partial"] for g in groups]
    exacts   = [g["exact"]   for g in groups]
    colors   = [g["color"]   for g in groups]
    hatches  = [g["hatch"]   for g in groups]

    w = 0.38
    bars_p = ax.bar(x - w/2, partials, w, color=colors, alpha=0.88, zorder=3, label="_nolegend_")
    bars_e = ax.bar(x + w/2, exacts,   w, color=colors, alpha=0.40, zorder=3, label="_nolegend_")

    for bar, h in zip(bars_p, hatches):
        if h:
            bar.set_hatch(h); bar.set_edgecolor("#333")
    for bar, h in zip(bars_e, hatches):
        if h:
            bar.set_hatch(h); bar.set_edgecolor("#333")

    for bar, val in zip(bars_p, partials):
        if val > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.007,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(bars_e, exacts):
        if val > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.007,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color="#555")

    # Group separators
    boundaries = []
    prev = groups[0]["group"]
    for i, g in enumerate(groups):
        if g["group"] != prev:
            boundaries.append(i - 0.5)
            prev = g["group"]
    for b in boundaries:
        ax.axvline(b, color="gray", linestyle="--", alpha=0.5, linewidth=1.2)

    # Group labels
    group_ranges = {}
    for i, g in enumerate(groups):
        grp = g["group"]
        if grp not in group_ranges:
            group_ranges[grp] = [i, i]
        group_ranges[grp][1] = i

    group_names = {
        "LLM": "Our LLM System\n(llama3.2, no fine-tuning)",
        "SetFit-EXP1": "SetFitQuad EXP1\n(our runs, top-5 strategies)",
        "SetFit-EXP2": "SetFitQuad EXP2\n(our runs, top-3 models)",
        "Paper": "Published Paper\n(SetFitQuad-others PDF)",
    }
    ymax = max(partials + exacts) * 1.45
    for grp, (start, end) in group_ranges.items():
        mid = (start + end) / 2
        ax.text(mid, ymax * 0.97, group_names.get(grp, grp),
                ha="center", va="top", fontsize=9, style="italic",
                color="#333", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0, ymax)
    ax.set_title(
        "Complete Comparison: Our LLM vs SetFitQuad (Our Runs) vs Published Paper\n"
        "Dark bars = Partial Match F1  |  Light bars = Exact Match F1",
        fontsize=13, fontweight="bold"
    )
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    legend = [
        mpatches.Patch(color="#E74C3C", label="Zero-Shot (0 examples)"),
        mpatches.Patch(color="#F39C12", label="One-Shot  (1 example)"),
        mpatches.Patch(color="#2ECC71", label="Few-Shot v1 (10 examples)"),
        mpatches.Patch(color="#1ABC9C", label="Few-Shot v2 (improved, 10 examples)"),
        mpatches.Patch(color="#2980B9", hatch="//", label="SetFitQuad EXP1 (our runs)"),
        mpatches.Patch(color="#7D3C98", hatch="xx", label="SetFitQuad EXP2 (our runs)"),
        mpatches.Patch(color="#27AE60", hatch="**", label="Paper published results"),
        mpatches.Patch(color="gray",    alpha=0.9, label="Dark = Partial F1"),
        mpatches.Patch(color="gray",    alpha=0.3, label="Light = Exact F1"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.95, ncol=2)
    fig.tight_layout()
    p = os.path.join(OUT, "cmp1_overview.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p}")


# ════════════════════════════════════════════════════════════
# CHART 2: LLM Component Breakdown (Exact per-field F1)
# ════════════════════════════════════════════════════════════
def chart2_llm_components(llm):
    metrics  = ["exact_f1", "aspect", "category", "opinion", "sentiment"]
    m_labels = ["Quad\n(all 4)", "Aspect", "Category", "Opinion", "Sentiment"]
    colors   = ["#E74C3C", "#F39C12", "#2ECC71", "#1ABC9C"]

    n = len(llm)
    x = np.arange(len(metrics))
    w = 0.72 / n

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (label, vals) in enumerate(llm.items()):
        vlist  = [vals.get(m, 0) for m in metrics]
        offset = (i - n/2 + 0.5) * w
        bars   = ax.bar(x + offset, vlist, w * 0.9,
                        label=label, color=colors[i], alpha=0.88, zorder=3)
        for bar, v in zip(bars, vlist):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

    # Paper subtask lines for reference
    paper_colors = {"ACD": "#27AE60", "ASC": "#8E44AD", "ATE": "#E67E22", "OTE": "#C0392B"}
    subtask_x    = {"ACD": 2, "ASC": 4, "ATE": 1, "OTE": 3}
    paper_vals   = PAPER_SUBTASKS["SetFitQuad (paper)"]
    for sub, xi in subtask_x.items():
        val = paper_vals[sub]
        ax.plot([xi - 0.42, xi + 0.42], [val, val],
                color=paper_colors[sub], linewidth=2.5, linestyle="--",
                label=f"Paper {sub}={val:.3f}", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(m_labels, fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0, 0.92)
    ax.set_title(
        "Our LLM System — F1 per Component (Exact Match)\n"
        "Dashed lines = SetFitQuad paper subtask F1 (ACD, ASC, ATE, OTE)",
        fontsize=12, fontweight="bold"
    )
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.legend(fontsize=8.5, loc="upper right", ncol=2, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "cmp2_llm_components.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p}")


# ════════════════════════════════════════════════════════════
# CHART 3: All 19 SetFitQuad Strategies + Our LLM lines
# ════════════════════════════════════════════════════════════
def chart3_setfit_strategies(sf1, llm):
    sorted_sf = sorted(sf1.items(), key=lambda x: -x[1]["partial_f1"])
    names  = [s[0] for s in sorted_sf]
    pm_vals = [s[1]["partial_f1"] for s in sorted_sf]
    em_vals = [s[1]["exact_f1"]   for s in sorted_sf]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x - 0.2, pm_vals, 0.37, label="Partial Match F1 (our runs)",
           color="#2980B9", alpha=0.85, zorder=3)
    ax.bar(x + 0.2, em_vals, 0.37, label="Exact Match F1 (our runs)",
           color="#AED6F1", alpha=0.85, zorder=3)

    # Our LLM lines
    line_cfg = [
        ("Zero-Shot",   "#E74C3C", "--",  "Our Zero-Shot"),
        ("One-Shot",    "#F39C12", ":",   "Our One-Shot"),
        ("Few-Shot v1", "#2ECC71", "-.",  "Our Few-Shot v1"),
        ("Few-Shot v2", "#1ABC9C", "-",   "Our Few-Shot v2 (best)"),
    ]
    for key, col, ls, desc in line_cfg:
        val = llm.get(key, {}).get("partial_f1", 0)
        if val:
            ax.axhline(val, color=col, linestyle=ls, linewidth=2.2,
                       label=f"{desc} Partial F1={val:.3f}", zorder=4)

    # Paper line
    ax.axhline(0.532, color="#27AE60", linestyle=(0, (3, 1, 1, 1)),
               linewidth=2.5, label="Paper best Partial F1=0.532", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0, 0.72)
    ax.set_title(
        "SetFitQuad EXP1 — All 19 Sampling Strategies (Our Runs)\n"
        "Horizontal lines = Our LLM Partial F1 & Paper reference",
        fontsize=12, fontweight="bold"
    )
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.legend(fontsize=8.5, loc="upper right", ncol=2, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "cmp3_setfit_strategies.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p}")


# ════════════════════════════════════════════════════════════
# CHART 4: Learning Curve (EXP3) + Our Lines + Paper
# ════════════════════════════════════════════════════════════
def chart4_learning_curve(sf3, llm):
    if not sf3:
        print("  [skip] chart4 — no EXP3 data")
        return
    sizes    = sorted(sf3.keys())
    pm_vals  = [sf3[s] for s in sizes]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, pm_vals, "o-", color="#2980B9", linewidth=2.5,
            markersize=9, label="SetFitQuad (our runs) — Partial F1 vs training size",
            zorder=4)
    for s, v in zip(sizes, pm_vals):
        ax.annotate(f"{v:.3f}", (s, v),
                    textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=9, color="#2980B9", fontweight="bold")

    # Our LLM lines
    line_cfg = [
        ("Zero-Shot",   "#E74C3C", "--"),
        ("One-Shot",    "#F39C12", ":"),
        ("Few-Shot v1", "#2ECC71", "-."),
        ("Few-Shot v2", "#1ABC9C", "-"),
    ]
    for key, col, ls in line_cfg:
        val = llm.get(key, {}).get("partial_f1", 0)
        if val:
            ax.axhline(val, color=col, linestyle=ls, linewidth=2,
                       label=f"Our {key} = {val:.3f}")

    # Paper line
    ax.axhline(0.532, color="#27AE60", linestyle=(0, (3,1,1,1)),
               linewidth=2.5, label="SetFitQuad paper best = 0.532", zorder=3)

    ax.set_xlabel("SetFitQuad Training Sample Size", fontsize=12)
    ax.set_ylabel("Partial Match F1", fontsize=12)
    ax.set_title(
        "EXP3: SetFitQuad Performance vs Training Size\n"
        "vs Our LLM (which uses 0 training samples for inference)",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylim(0.15, 0.72)
    ax.set_xticks(sizes)
    ax.yaxis.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "cmp4_learning_curve.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p}")


# ════════════════════════════════════════════════════════════
# CHART 5: Paper Subtask Comparison
# ════════════════════════════════════════════════════════════
def chart5_subtasks(llm):
    tasks   = ["ACD", "ASC", "ATE", "OTE"]
    t_labels = ["Aspect Category\nDetection (ACD)",
                "Aspect Sentiment\nClassification (ASC)",
                "Aspect Term\nExtraction (ATE)",
                "Opinion Term\nExtraction (OTE)"]

    # Map our metrics to subtasks (approximate)
    llm_map = {
        "ACD": "category",
        "ASC": "sentiment",
        "ATE": "aspect",
        "OTE": "opinion",
    }

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(tasks))
    w = 0.18

    # Paper systems
    paper_colors = {"SetFitQuad (paper)": "#27AE60",
                    "T5-ASQP (baseline)": "#C0392B",
                    "SetFitABSA":          "#8E44AD"}
    for i, (sys_name, vals) in enumerate(PAPER_SUBTASKS.items()):
        ys = [vals.get(t, 0) for t in tasks]
        offset = (i - 2) * w
        bars = ax.bar(x + offset, ys, w * 0.9,
                      label=sys_name, color=paper_colors[sys_name],
                      alpha=0.85, zorder=3)
        for bar, v in zip(bars, ys):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # Our LLM (few-shot v2 — best)
    our_colors = {"Few-Shot v2": "#1ABC9C"}
    best_llm = llm.get("Few-Shot v2", {})
    if best_llm:
        ys = [best_llm.get(llm_map[t], 0) for t in tasks]
        offset = 1 * w
        bars = ax.bar(x + offset, ys, w * 0.9,
                      label="Our Few-Shot v2 (LLM)", color="#1ABC9C",
                      alpha=0.88, zorder=3)
        for bar, v in zip(bars, ys):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                        fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(t_labels, fontsize=10)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Subtask F1 Comparison: Paper Systems vs Our LLM\n"
        "(ACD=Category, ASC=Sentiment, ATE=Aspect, OTE=Opinion)",
        fontsize=12, fontweight="bold"
    )
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "cmp5_subtasks.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p}")


# ════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════
def print_summary(llm, sf1, sf2, sf3):
    sep = "=" * 78

    print(f"\n{sep}")
    print("  FINAL COMPARISON SUMMARY")
    print(f"  Metric: Partial Match F1 (LCS-based, Hungarian matching)")
    print(f"  Dataset: Rest15 + Rest16  |  778 test reviews")
    print(sep)

    # --- Our LLM ---
    print(f"\n  [A] OUR LLM SYSTEM (llama3.2:latest, no fine-tuning)")
    print(f"  {'Strategy':<18} {'Partial F1':>11} {'Exact F1':>10} {'Aspect':>8} {'Cat':>8} {'Opinion':>8} {'Sent':>8}")
    print(f"  {'-'*73}")
    for name, v in llm.items():
        print(f"  {name:<18} {v['partial_f1']:>11.4f} {v['exact_f1']:>10.4f} "
              f"{v['aspect']:>8.4f} {v['category']:>8.4f} {v['opinion']:>8.4f} {v['sentiment']:>8.4f}")

    # --- SetFitQuad our runs ---
    print(f"\n  [B] SETFITQUAD — OUR RUNS (C:\\SetFitQuad-Code-main)")
    print(f"  {'System':<30} {'Partial F1':>11} {'Exact F1':>10}")
    print(f"  {'-'*53}")
    all_sf = {}
    for k, v in sorted(sf1.items(), key=lambda x: -x[1]["partial_f1"])[:8]:
        print(f"  EXP1 {k[:25]:<25} {v['partial_f1']:>11.4f} {v['exact_f1']:>10.4f}")
        all_sf[k] = v["partial_f1"]
    print(f"  {'---':30}")
    for k, v in sorted(sf2.items(), key=lambda x: -x[1]["partial_f1"]):
        print(f"  EXP2 {k[:25]:<25} {v['partial_f1']:>11.4f} {v['exact_f1']:>10.4f}")

    # --- EXP3 ---
    if sf3:
        print(f"\n  [C] SETFITQUAD EXP3 — Sample Size Effect (our runs)")
        print(f"  {'Size':<8} {'Partial F1':>11}")
        print(f"  {'-'*20}")
        for size in sorted(sf3.keys()):
            marker = " <-- same as our Few-Shot" if size == 10 else ""
            print(f"  {size:<8} {sf3[size]:>11.4f}{marker}")

    # --- Paper ---
    print(f"\n  [D] PUBLISHED PAPER (SetFitQuad-others PDF)")
    print(f"  {'System':<30} {'Partial F1':>11} {'Exact F1':>10}")
    print(f"  {'-'*53}")
    for name, v in PAPER_RESULTS.items():
        n = name.replace('\n', ' ')
        print(f"  {n:<30} {v['partial_f1']:>11.3f} {v['exact_f1']:>10.3f}")

    # --- Key Insights ---
    best_llm_name = max(llm, key=lambda k: llm[k]["partial_f1"])
    best_llm_pf   = llm[best_llm_name]["partial_f1"]
    best_sf_pf    = max(v["partial_f1"] for v in {**sf1, **sf2}.values())
    paper_pf      = PAPER_RESULTS["SetFitQuad\n(paper, CS)"]["partial_f1"]
    t5_pf         = PAPER_RESULTS["T5-ASQP\n(paper baseline)"]["partial_f1"]

    print(f"\n{sep}")
    print(f"  KEY FINDINGS:")
    print(f"  {'='*74}")
    print(f"  Our best LLM   ({best_llm_name:<16}) Partial F1 = {best_llm_pf:.4f}")
    print(f"  SetFitQuad best (our runs)          Partial F1 = {best_sf_pf:.4f}  (requires 50 examples + training)")
    print(f"  SetFitQuad best (paper)              Partial F1 = {paper_pf:.3f}  (published result)")
    print(f"  T5-ASQP baseline (paper)             Partial F1 = {t5_pf:.3f}  (large model, full fine-tune)")
    print(f"")
    diff_sf = best_llm_pf - best_sf_pf
    diff_paper = best_llm_pf - paper_pf
    diff_t5    = best_llm_pf - t5_pf
    print(f"  vs SetFitQuad (our runs): {'+' if diff_sf>=0 else ''}{diff_sf:.4f}  ({'LLM WINS' if diff_sf>=0 else 'SetFit wins'})")
    print(f"  vs SetFitQuad (paper)   : {'+' if diff_paper>=0 else ''}{diff_paper:.4f}  ({'LLM WINS' if diff_paper>=0 else 'SetFit paper wins'})")
    print(f"  vs T5-ASQP baseline     : {'+' if diff_t5>=0 else ''}{diff_t5:.4f}  ({'LLM WINS' if diff_t5>=0 else 'T5 wins'})")
    print(f"")
    print(f"  CONCLUSION:")
    print(f"  Our LLM (llama3.2, Few-Shot v2) achieves Partial F1 = {best_llm_pf:.4f}")
    if diff_sf >= 0:
        print(f"  - Outperforms SetFitQuad (our runs) by {diff_sf*100:.1f}% with NO fine-tuning")
    if diff_paper >= 0:
        print(f"  - Outperforms the published paper result by {diff_paper*100:.1f}%")
    else:
        print(f"  - Paper result is {abs(diff_paper)*100:.1f}% higher (different eval setup possible)")
    if diff_t5 < 0:
        print(f"  - T5-ASQP still leads by {abs(diff_t5)*100:.1f}% but requires full fine-tuning on all data")
    print(f"  - Our approach: only 10 in-context examples, zero training cost")
    print(sep)


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60)
    print("  Final Comparison: LLM vs SetFitQuad vs Paper")
    print("=" * 60)

    llm = load_llm()
    sf1 = load_sf_exp1()
    sf2 = load_sf_exp2()
    sf3 = load_sf_exp3()

    print(f"  Our LLM strategies  : {len(llm)}")
    print(f"  SetFitQuad EXP1     : {len(sf1)} sampling strategies")
    print(f"  SetFitQuad EXP2     : {len(sf2)} pretrained models")
    print(f"  SetFitQuad EXP3     : {len(sf3)} sample sizes")
    print(f"  Paper results       : {len(PAPER_RESULTS)} systems (from PDF)")

    print_summary(llm, sf1, sf2, sf3)

    print("\nGenerating 5 charts...")
    chart1_grand_overview(llm, sf1, sf2)
    chart2_llm_components(llm)
    chart3_setfit_strategies(sf1, llm)
    chart4_learning_curve(sf3, llm)
    chart5_subtasks(llm)

    print(f"\nAll charts saved to '{OUT}/'")


if __name__ == "__main__":
    main()
