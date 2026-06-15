import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pathlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import final_dataset

df   = pd.read_excel(final_dataset)
gold = df[df["handmatig_label"].isin(["afwijkend", "niet-afwijkend"])].copy()

gold["artrose"]  = gold["artrose_radiologie"].map({"ja": "Artrose", "nee": "Geen artrose"})
gold["trauma"]   = gold["label_trauma"].map({
    "trauma":      "Trauma",
    "oud_trauma":  "Oud trauma",
    "niet_trauma": "Geen trauma",
    "onbekend":    "Onbekend",
})
gold["fractuur"] = gold["label_fractuur"].map({
    "ja":       "Fractuur",
    "nee":      "Geen fractuur",
    "onbekend": "Onbekend",
})


artrose_vals = ["Artrose", "Geen artrose"]
trauma_vals  = ["Geen trauma", "Oud trauma", "Trauma", "Onbekend"]
frac_vals    = ["Fractuur", "Geen fractuur", "Onbekend"]

sub = gold.dropna(subset=["artrose", "trauma", "fractuur"])


rows_meta = [(a, t) for a in artrose_vals for t in trauma_vals
             if len(sub[(sub["artrose"]==a) & (sub["trauma"]==t)]) > 0]


n_mat   = np.zeros((len(rows_meta), len(frac_vals)), dtype=int)
pct_mat = np.full((len(rows_meta), len(frac_vals)), np.nan)

for ri, (a, t) in enumerate(rows_meta):
    for ci, f in enumerate(frac_vals):
        cell = sub[(sub["artrose"]==a) & (sub["trauma"]==t) & (sub["fractuur"]==f)]
        n = len(cell)
        n_mat[ri, ci] = n
        if n > 0:
            pct_mat[ri, ci] = (cell["handmatig_label"] == "afwijkend").mean() * 100


row_total_n   = n_mat.sum(axis=1)
row_total_pct = np.array([
    (sub[(sub["artrose"]==a) & (sub["trauma"]==t)]["handmatig_label"] == "afwijkend").mean() * 100
    if row_total_n[ri] > 0 else np.nan
    for ri, (a, t) in enumerate(rows_meta)
])


nrows = len(rows_meta)
ncols = len(frac_vals) + 1  

fig, ax = plt.subplots(figsize=(10, 0.55 * nrows + 2.5))
ax.axis("off")

col_labels = frac_vals + ["Totaal"]
row_labels  = [f"{a}\n× {t}" for a, t in rows_meta]


cmap = plt.cm.get_cmap("YlOrRd")


cell_text   = []
cell_colors = []

for ri, (a, t) in enumerate(rows_meta):
    row_txt = []
    row_col = []
    for ci, f in enumerate(frac_vals):
        n = n_mat[ri, ci]
        if n == 0:
            row_txt.append("—")
            row_col.append("#f5f5f5")
        else:
            pct = pct_mat[ri, ci]
            row_txt.append(f"n={n}\n{pct:.0f}% afw.")
            row_col.append(mcolors.to_hex(cmap(pct / 100)))
    
    nt = row_total_n[ri]
    pt = row_total_pct[ri]
    row_txt.append(f"n={nt}\n{pt:.0f}% afw.")
    row_col.append(mcolors.to_hex(cmap(pt / 100)))
    cell_text.append(row_txt)
    cell_colors.append(row_col)


sep_after = sum(1 for a, _ in rows_meta if a == "Artrose") - 1

table = ax.table(
    cellText=cell_text,
    rowLabels=row_labels,
    colLabels=col_labels,
    cellColours=cell_colors,
    cellLoc="center",
    rowLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1.0, 2.2)


for ci in range(ncols):
    cell = table[0, ci]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")


for ri, (a, _) in enumerate(rows_meta):
    cell = table[ri+1, -1]   
    if a == "Artrose":
        cell.set_facecolor("#d5e8d4")
    else:
        cell.set_facecolor("#dae8fc")


for ci in range(-1, ncols):
    try:
        table[sep_after + 1, ci].visible_edges = "TBL" if ci == -1 else "TB"
    except Exception:
        pass


sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                    fraction=0.015, pad=0.01, shrink=0.6)
cbar.set_label("% afwijkend", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_title(
    "3-weg kruistabel: Artrose × Trauma × Fractuur  (gold test set, n=383)\n"
    "Celwaarden: aantal gevallen + % afwijkend gold label",
    fontsize=10.5, fontweight="bold", pad=14,
)

fig.tight_layout()
out = pathlib.Path(__file__).parent / "figures" / "fig_3way_crosstab"
out.parent.mkdir(exist_ok=True)
fig.savefig(str(out) + ".pdf", bbox_inches="tight")
fig.savefig(str(out) + ".png", dpi=180, bbox_inches="tight")
print(f"Saved: {out}.pdf  and  {out}.png")
