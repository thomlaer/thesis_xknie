import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact
import pathlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import final_dataset

df   = pd.read_excel(final_dataset)
gold = df[df["handmatig_label"].isin(["afwijkend", "niet-afwijkend"])].copy()
full = df[df["label_gecombineerd"].isin(["afwijkend", "niet-afwijkend"])].copy()




def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    d = 1 + z**2/n
    c = (p + z**2/(2*n)) / d
    h = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return p*100, max(0, c-h)*100, min(1, c+h)*100

def crosstab_pct(data, group_col, group_vals, label_col="handmatig_label", pos="afwijkend"):
    rows = []
    for gv in group_vals:
        sub = data[data[group_col] == gv]
        k = (sub[label_col] == pos).sum()
        n = len(sub)
        pct, lo, hi = wilson(k, n)
        rows.append({"group": gv, "pct": pct, "lo": lo, "hi": hi, "n": n, "k": k})
    return rows


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.subplots_adjust(wspace=0.38)

colors = ["#009E73", "#56B4E9"]  

def clean_ax(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.5, zorder=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))




ax = axes[0]

groups = [
    ("Artrose in\nvraagstelling: Ja",  full[full["artrose_vraagstelling"]=="ja"],  "#009E73"),
    ("Artrose in\nvraagstelling: Nee", full[full["artrose_vraagstelling"]=="nee"], "#56B4E9"),
    ("Artrose in\nradiologie: Ja",     full[full["artrose_radiologie"]=="ja"],     "#E69F00"),
    ("Artrose in\nradiologie: Nee",    full[full["artrose_radiologie"]=="nee"],    "#D55E00"),
]

x = np.arange(len(groups))
for xi, (label, sub, color) in enumerate(groups):
    k = (sub["label_gecombineerd"] == "afwijkend").sum()
    n = len(sub)
    pct, lo, hi = wilson(k, n)
    ax.bar(xi, pct, width=0.55, color=color, alpha=0.88, zorder=3)
    ax.errorbar(xi, pct, yerr=[[pct-lo], [hi-pct]],
                fmt="none", color="#333333", capsize=4, linewidth=1.1, zorder=4)
    ax.text(xi, hi + 1.5, f"{pct:.1f}%\n(n={n:,})",
            ha="center", va="bottom", fontsize=8)


overall_full = (full["label_gecombineerd"]=="afwijkend").sum() / len(full) * 100
ax.axhline(overall_full, color="#333333", linestyle="--",
           linewidth=1.0, alpha=0.6, label=f"Overall {overall_full:.0f}%")

ax.set_xticks(x)
ax.set_xticklabels([g[0] for g in groups], fontsize=8.8)
ax.set_ylabel("% afwijkend (silver label)", fontsize=9.5)
ax.set_ylim(0, 115)
ax.set_title("Volledige dataset (n=6,042)\nLLM-consensus silver label", fontsize=10, fontweight="bold")
ax.legend(fontsize=8.5, frameon=False)
clean_ax(ax)

ax.annotate("Paper: 88.6% (ja)\n         69.5% (nee)",
            xy=(0.02, 0.97), xycoords="axes fraction",
            fontsize=7.5, va="top", color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="#cccccc"))

ax = axes[1]

groups2 = [
    ("Artrose in\nvraagstelling: Ja",  gold[gold["artrose_vraagstelling"]=="ja"],  "#009E73"),
    ("Artrose in\nvraagstelling: Nee", gold[gold["artrose_vraagstelling"]=="nee"], "#56B4E9"),
    ("Artrose in\nradiologie: Ja",     gold[gold["artrose_radiologie"]=="ja"],     "#E69F00"),
    ("Artrose in\nradiologie: Nee",    gold[gold["artrose_radiologie"]=="nee"],    "#D55E00"),
]

x = np.arange(len(groups2))
for xi, (label, sub, color) in enumerate(groups2):
    k = (sub["handmatig_label"] == "afwijkend").sum()
    n = len(sub)
    pct, lo, hi = wilson(k, n)
    ax.bar(xi, pct, width=0.55, color=color, alpha=0.88, zorder=3)
    ax.errorbar(xi, pct, yerr=[[pct-lo], [hi-pct]],
                fmt="none", color="#333333", capsize=4, linewidth=1.1, zorder=4)
    ax.text(xi, hi + 1.5, f"{pct:.1f}%\n(n={n})",
            ha="center", va="bottom", fontsize=8)

overall_gold = (gold["handmatig_label"]=="afwijkend").sum() / len(gold) * 100
ax.axhline(overall_gold, color="#333333", linestyle="--",
           linewidth=1.0, alpha=0.6, label=f"Overall {overall_gold:.0f}%")

ax.set_xticks(x)
ax.set_xticklabels([g[0] for g in groups2], fontsize=8.8)
ax.set_ylabel("% afwijkend (gold label)", fontsize=9.5)
ax.set_ylim(0, 115)
ax.set_title("Gold test set (n=383)\nHandmatig gold label", fontsize=10, fontweight="bold")
ax.legend(fontsize=8.5, frameon=False)
clean_ax(ax)


ct1 = pd.crosstab(gold["artrose_vraagstelling"], gold["handmatig_label"])
chi2_1, p1, _, _ = chi2_contingency(ct1)
ct2 = pd.crosstab(gold["artrose_radiologie"], gold["handmatig_label"])
chi2_2, p2, _, _ = chi2_contingency(ct2)
ax.annotate(f"Vraagstelling p={p1:.3f}\nRadiologie     p<0.001",
            xy=(0.02, 0.97), xycoords="axes fraction",
            fontsize=7.5, va="top", color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="#cccccc"))


fig.tight_layout(rect=[0, 0.02, 1, 1])
out = pathlib.Path(__file__).parent / "figures" / "fig_vraagstelling_vs_label"
out.parent.mkdir(exist_ok=True)
fig.savefig(str(out) + ".pdf", bbox_inches="tight")
fig.savefig(str(out) + ".png", dpi=180, bbox_inches="tight")
print(f"Saved: {out}.pdf  and  {out}.png")
