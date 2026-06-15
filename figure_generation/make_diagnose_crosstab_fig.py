import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import pathlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import final_dataset

df = pd.read_excel(final_dataset)
gold = df[df["handmatig_label"].isin(["afwijkend", "niet-afwijkend"])].copy()

gold["artrose"]  = gold["artrose_radiologie"].map({"ja": "artrose", "nee": "geen artrose"})
gold["trauma"]   = gold["label_trauma"].map({
    "trauma": "trauma", "niet_trauma": "geen trauma",
    "oud_trauma": "oud trauma", "onbekend": None
})
gold["fractuur"] = gold["label_fractuur"].map({
    "ja": "fractuur", "nee": "geen fractuur", "onbekend": None
})


fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
fig.subplots_adjust(wspace=0.40)

def clean_ax(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.5, zorder=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

ax = axes[0]
sub = gold.dropna(subset=["artrose", "trauma"])
ct  = pd.crosstab(sub["artrose"], sub["trauma"])
chi2_v, p_v, _, _ = chi2_contingency(ct)

cats   = ["artrose", "geen artrose"]
order  = ["geen trauma", "oud trauma", "trauma"]
pal    = {"geen trauma": "#56B4E9", "oud trauma": "#E69F00", "trauma": "#D55E00"}

x = np.arange(len(cats))
bottoms = np.zeros(len(cats))
for sp in order:
    vals = [
        (sub[sub["artrose"] == c]["trauma"] == sp).sum() /
        (sub["artrose"] == c).sum() * 100
        for c in cats
    ]
    ax.bar(x, vals, bottom=bottoms, width=0.55,
           color=pal[sp], alpha=0.88, label=sp, zorder=3)
    bottoms += np.array(vals)

ns = [(sub["artrose"] == c).sum() for c in cats]
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cats, ns)], fontsize=9)
ax.set_ylabel("Percentage (%)", fontsize=9.5)
ax.set_ylim(0, 115)
ax.set_title("Artrose vs. Trauma\n(p<0.001, ***)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, frameon=False, loc="upper right")
clean_ax(ax)

ax = axes[1]
sub = gold.dropna(subset=["artrose", "fractuur"])
ct  = pd.crosstab(sub["artrose"], sub["fractuur"])
chi2_v2, p_v2, _, _ = chi2_contingency(ct)

order2 = ["geen fractuur", "fractuur"]
pal2   = {"geen fractuur": "#56B4E9", "fractuur": "#D55E00"}

x = np.arange(len(cats))
bottoms = np.zeros(len(cats))
for sp in order2:
    vals = [
        (sub[sub["artrose"] == c]["fractuur"] == sp).sum() /
        (sub["artrose"] == c).sum() * 100
        for c in cats
    ]
    ax.bar(x, vals, bottom=bottoms, width=0.55,
           color=pal2[sp], alpha=0.88, label=sp, zorder=3)
    bottoms += np.array(vals)

ns = [(sub["artrose"] == c).sum() for c in cats]
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cats, ns)], fontsize=9)
ax.set_ylabel("Percentage (%)", fontsize=9.5)
ax.set_ylim(0, 115)
ax.set_title("Artrose vs. Fractuur\n(p<0.001, ***)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, frameon=False, loc="upper right")
clean_ax(ax)

ax = axes[2]
sub3 = gold.dropna(subset=["artrose", "fractuur", "trauma"])
sub3 = sub3[sub3["trauma"].isin(["trauma", "geen trauma"])].copy()

def diag_group(row):
    a = row["artrose"] == "artrose"
    f = row["fractuur"] == "fractuur"
    t = row["trauma"] == "trauma"
    if a and not f and not t:
        return "Artrose\n(geen frac/trauma)"
    elif t and f:
        return "Trauma +\nfractuur"
    elif a and f:
        return "Artrose +\nfractuur"
    elif not a and not f and not t:
        return "Geen artrose/\nfrac/trauma"
    else:
        return "Overig"

sub3["groep"] = sub3.apply(diag_group, axis=1)

groups_order = [
    "Artrose\n(geen frac/trauma)",
    "Artrose +\nfractuur",
    "Trauma +\nfractuur",
    "Geen artrose/\nfrac/trauma",
    "Overig",
]
groups_order = [g for g in groups_order if g in sub3["groep"].unique()]

x = np.arange(len(groups_order))
pct_afw = []
ns3 = []
for g in groups_order:
    s = sub3[sub3["groep"] == g]
    pct_afw.append((s["handmatig_label"] == "afwijkend").mean() * 100)
    ns3.append(len(s))

overall = (gold["handmatig_label"] == "afwijkend").mean() * 100

ax.bar(x, pct_afw, width=0.55, color="#009E73", alpha=0.88, zorder=3)
ax.axhline(y=overall, color="#333333", linestyle="--",
           linewidth=1.1, alpha=0.65, label=f"Overall {overall:.0f}%")

for xi, v, n in zip(x, pct_afw, ns3):
    ax.text(xi, v + 1.8, f"{v:.0f}%\n(n={n})",
            ha="center", va="bottom", fontsize=8.2)

ax.set_xticks(x)
ax.set_xticklabels(groups_order, fontsize=8.5)
ax.set_ylabel("% afwijkend", fontsize=9.5)
ax.set_ylim(0, 118)
ax.set_title("% afwijkend per diagnosegroep", fontsize=10, fontweight="bold")
ax.legend(fontsize=8.5, frameon=False)
clean_ax(ax)


fig.tight_layout(rect=[0, 0.02, 1, 1])
out = pathlib.Path(__file__).parent / "figures" / "fig_diagnose_crosstab"
out.parent.mkdir(exist_ok=True)
fig.savefig(str(out) + ".pdf", bbox_inches="tight")
fig.savefig(str(out) + ".png", dpi=180, bbox_inches="tight")
print(f"Saved: {out}.pdf  and  {out}.png")
