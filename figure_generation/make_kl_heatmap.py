import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pathlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import final_dataset

df   = pd.read_excel(final_dataset)
gold = df[df["handmatig_label"].notna()].copy()   


sub = gold.dropna(subset=["artrose_graad_consensus", "handmatig_artrose_graad"])
sub = sub[sub["artrose_graad_consensus"] != "onbekend"].copy()
sub["artrose_graad_consensus"]  = sub["artrose_graad_consensus"].astype(int)
sub["handmatig_artrose_graad"]  = sub["handmatig_artrose_graad"].astype(int)

grades = [0, 1, 2, 3, 4]
y_true = sub["handmatig_artrose_graad"]
y_pred = sub["artrose_graad_consensus"]
n      = len(sub)

cm = confusion_matrix(y_true, y_pred, labels=grades)


cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)


fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)


for i in range(len(grades)):
    for j in range(len(grades)):
        count = cm[i, j]
        pct   = cm_pct[i, j]
        color = "white" if pct > 0.55 else "black"
        ax.text(j, i, f"{count}\n({pct*100:.0f}%)",
                ha="center", va="center", fontsize=9, color=color)

ax.set_xticks(range(len(grades)))
ax.set_yticks(range(len(grades)))
ax.set_xticklabels([f"KL {g}" for g in grades])
ax.set_yticklabels([f"KL {g}" for g in grades])
ax.invert_yaxis()
ax.set_xlabel("LLM consensus grade", fontsize=10)
ax.set_ylabel("Gold standard grade", fontsize=10)
ax.set_title("Kellgren–Lawrence grading: LLM consensus vs gold", fontsize=11)
plt.tight_layout()

out = pathlib.Path(__file__).parent / "figures" / "fig_kl_heatmap"
out.parent.mkdir(exist_ok=True)
fig.savefig(str(out) + ".pdf", bbox_inches="tight")
fig.savefig(str(out) + ".png", dpi=180, bbox_inches="tight")
print(f"Saved: {out}.pdf  and  {out}.png  (n={n})")
