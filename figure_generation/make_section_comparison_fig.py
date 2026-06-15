import pathlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import final_dataset, gold_final, random_state


labels = ["afwijkend", "niet-afwijkend"]
few_shot_example_case_ids = {11, 374, 437, 455}

gold_rename = {
    "definitief_handmatig_label": "handmatig_label",
    "definitief_artrose_aanwezig": "handmatig_artrose_aanwezig",
    "definitief_artrose_graad": "handmatig_artrose_graad",
    "definitief_nhg": "handmatig_nhg",
    "herkomst": "gold_herkomst",
}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, centre - half), min(1, centre + half)


def bootstrap_f1(y_true, y_pred, n=1000):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(set(y_true[idx])) < 2:
            continue
        scores.append(f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def load_gold_with_primary_labels():
    primary = pd.read_excel(final_dataset)
    gold = pd.read_excel(gold_final).rename(columns=gold_rename)
    gold_cols = [
        "case_id", "handmatig_label", "handmatig_artrose_aanwezig",
        "handmatig_artrose_graad", "handmatig_nhg", "label_martijn",
        "label_heleen", "label_artrose_martijn", "label_artrose_heleen",
        "artrose_graad_martijn", "artrose_graad_heleen", "nhg_martijn",
        "nhg_heleen", "gold_herkomst",
    ]
    gold = gold[[c for c in gold_cols if c in gold.columns]].copy()
    primary_cols = [c for c in primary.columns if c == "case_id" or c not in gold.columns]
    return gold.merge(primary[primary_cols], on="case_id", how="left", validate="one_to_one")


def metric_row(df, pred_col):
    sub = df[df["handmatig_label"].isin(labels) & df[pred_col].isin(labels)].copy()
    y_true = sub["handmatig_label"].to_numpy()
    y_pred = sub[pred_col].to_numpy()
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_lo, f1_hi = bootstrap_f1(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp, fn = int(cm[0, 0]), int(cm[0, 1])
    sens = tp / (tp + fn)
    sens_lo, sens_hi = wilson_ci(tp, tp + fn)
    return {
        "f1": f1,
        "ci_lo": f1_lo,
        "ci_hi": f1_hi,
        "sens": sens,
        "sens_ci_lo": sens_lo,
        "sens_ci_hi": sens_hi,
    }


gold = load_gold_with_primary_labels()

data = {}
for variant, suffix in [("zero-shot", ""), ("few-shot", "_fewshot"), ("consensus", "_consensus")]:
    eval_gold = gold[~gold["case_id"].isin(few_shot_example_case_ids)].copy() if variant == "few-shot" else gold
    data[variant] = {}
    for section in ["bevindingen", "conclusie", "gecombineerd"]:
        data[variant][section] = metric_row(eval_gold, f"label_{section}{suffix}")


sections = ["bevindingen", "conclusie", "gecombineerd"]
sec_labels = ["Findings", "Conclusion", "Combined"]
variants = ["zero-shot", "few-shot", "consensus"]
var_labels = ["Zero-shot", "Few-shot", "Consensus"]
colors = ["#56B4E9", "#E69F00", "#009E73"]

x = np.arange(len(sections))
width = 0.22
offset = [-width, 0, width]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=False)
fig.subplots_adjust(wspace=0.32)

panel_cfg = [
    (
        axes[0], "f1", "Macro F1", (0.70, 0.895),
        lambda v: ([v["f1"] - v["ci_lo"]], [v["ci_hi"] - v["f1"]]),
    ),
    (
        axes[1], "sens", "Sensitivity (abnormal class)", (0.78, 1.01),
        lambda v: ([v["sens"] - v["sens_ci_lo"]], [v["sens_ci_hi"] - v["sens"]]),
    ),
]

for ax, metric, ylabel, ylim, get_err in panel_cfg:
    for vi, (variant, label, color) in enumerate(zip(variants, var_labels, colors)):
        vals = [data[variant][s][metric] for s in sections]
        xpos = x + offset[vi]
        ax.bar(xpos, vals, width=width * 0.92, color=color, alpha=0.88, label=label, zorder=3)
        errs = [get_err(data[variant][s]) for s in sections]
        err_lo = [e[0][0] for e in errs]
        err_hi = [e[1][0] for e in errs]
        ax.errorbar(
            xpos, vals, yerr=[err_lo, err_hi], fmt="none",
            color="#333333", capsize=3.5, linewidth=1.0, zorder=4,
        )
        for xi, v, ehi in zip(xpos, vals, err_hi):
            ax.text(xi, v + ehi + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=6.8)

    ax.set_xticks(x)
    ax.set_xticklabels(sec_labels, fontsize=10)
    ax.set_title(ylabel, fontsize=11, fontweight="bold", pad=7)
    ax.set_ylim(ylim)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.tick_params(axis="y", labelsize=8.5, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

handles = [mpatches.Patch(color=c, alpha=0.88, label=l) for c, l in zip(colors, var_labels)]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=[0, 0.07, 1, 1])

out = pathlib.Path(__file__).parent / "figures" / "fig_section_comparison"
out.parent.mkdir(exist_ok=True)
fig.savefig(str(out) + ".pdf", bbox_inches="tight")
fig.savefig(str(out) + ".png", dpi=180, bbox_inches="tight")
print(f"Saved: {out}.pdf and {out}.png")
