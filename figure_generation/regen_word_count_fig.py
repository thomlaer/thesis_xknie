import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import data_root

long = pd.read_csv(data_root / "results" / "results_figures" / "word_count_distribution_long.csv")

bins = ["0 words (empty)", "1-4 words", "5-9 words", "10-19 words", "20+ words"]
plot_data = long.pivot(index="label", columns="word_group", values="percentage").fillna(0)
plot_data = plot_data[[b for b in bins if b in plot_data.columns]]


row_order = [
    "Radiology findings",
    "Radiology conclusion",
    "GP referral question",
    "GP clinical information",
]
plot_data = plot_data.reindex([r for r in row_order if r in plot_data.index])

wc_colors = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]

ax = plot_data.plot(kind="barh", stacked=True, figsize=(8, 4), width=0.75,
                    color=wc_colors[:len(plot_data.columns)])
ax.set_xlabel("Percentage of records")
ax.set_ylabel("")
ax.set_xlim(0, 100)
ax.legend(title="Word count", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()

out = data_root / "results" / "results_figures"
out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "figure_word_count_distribution_new.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved.")
