import re
import pathlib
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import final_dataset


stopwords = {
    "de", "het", "een", "en", "van", "in", "is", "op", "te", "dat",
    "die", "der", "den", "voor", "met", "aan", "niet", "ook", "er",
    "om", "als", "maar", "of", "bij", "uit", "door", "ze", "zich",
    "tot", "no", "na", "wat", "worden", "zijn", "heeft", "worden",
    "worden", "heeft", "hebben", "had", "was", "worden", "zijn",
    "rechts", "links", "re", "li", "dd", "ivm", "nav", "obv",
    "graag", "even", "nog", "reeds", "al", "wel", "geen", "dit",
    "deze", "het", "kan", "wil", "zien", "doen", "gaan", "komen",
    "wordt", "werd", "waren", "heeft", "hebben", "want", "dus",
    "nu", "zo", "dan", "meer", "zeer", "veel", "hoe", "waar",
    "naar", "over", "onder", "boven", "tussen", "zonder", "samen",
    "reeds", "echter", "mede", "verder", "patient", "patiënt",
    "man", "vrouw", "jaar", "oud", "bekend", "zie", "nb",
    
    "stadspoli", "poli", "poliklinisch", "mumc", "azm",
}





nl_en = {
    
    "artrose":        "osteoarthritis",
    "gonartrose":     "gonarthrosis",
    "afwijkingen":    "abnormalities",
    "fractuur":       "fracture",
    "pijn":           "pain",
    "ossale":         "osseous",
    "pijnklachten":   "pain complaints",
    "pathologie":     "pathology",
    "toename":        "progression",
    "knieklachten":   "knee complaints",
    "patella":        "patella",
    "mediale":        "medial",
    "anders":         "other",
    "trauma":         "trauma",
    "progressie":     "progression",
    "knie":           "knee",
    "klachten":       "complaints",
    "laterale":       "lateral",
    "meniscus":       "meniscus",
    "letsel":         "injury",
    "degeneratieve":  "degenerative",
    "degeneratief":   "degenerative",
    
    "rechterknie":    "right knee",
    "linkerknie":     "left knee",
    "knieën":         "knees",
    "zwelling":       "swelling",
    "gevallen":       "fall(s)",
    "geleden":        "ago",
    "since":          "since",
    "sinds":          "since",
    "last":           "discomfort",
    "gonarthrosis":   "gonarthrosis",
    "klacht":         "complaint",
    "rechter":        "right",
    "linker":         "left",
    "andere":         "other",
    "anders":         "other",
    "weken":          "weeks",
    "maanden":        "months",
    "dagen":          "days",
    "hydrops":        "joint effusion",
    "zwellin":        "swelling",
}


def tokenize(text: str):
    tokens = re.findall(r"[a-zäëïöüáéíóúàèìòùâêîôû]+", str(text).lower())
    return [t for t in tokens if t not in stopwords and len(t) > 2]


def translate(term: str) -> str:
    return nl_en.get(term, term)





df = pd.read_excel(final_dataset)

fields = {
    "Clinical information": "klinische_gegevens_huisarts",
    "Clinical question": "vraagstelling_huisarts",
}

top_n = 15
color = "#4878CF"




fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(wspace=0.55)

for ax, (panel_title, col) in zip(axes, fields.items()):
    texts = df[col].dropna().astype(str)
    n_records = len(texts)
    word_counts = [len(tokenize(t)) for t in texts]
    mean_words = np.mean(word_counts) if word_counts else 0

    counter = Counter()
    for t in texts:
        counter.update(set(tokenize(t)))  # document frequency: each word once per record

    translated: Counter = Counter()
    for term, freq in counter.items():
        translated[translate(term)] += freq

    top = translated.most_common(top_n)
    terms_en, freqs = zip(*top)

    y = np.arange(top_n)
    ax.barh(y, freqs, color=color, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(terms_en, fontsize=9.5)
    ax.invert_yaxis()

    for yi, v in zip(y, freqs):
        ax.text(v + max(freqs)*0.01, yi, f"{v:,}",
                va="center", ha="left", fontsize=8.5)

    ax.set_xlabel("Number of records containing term", fontsize=10)
    ax.set_title(panel_title, fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, max(freqs) * 1.18)
    ax.text(0.98, -0.10,
            f"n = {n_records:,} records",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="dimgray")

    
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.5)

fig.tight_layout()

out = pathlib.Path(__file__).parent / "figures" / "figuur_topwoorden"
out.parent.mkdir(exist_ok=True)
fig.savefig(str(out) + ".pdf", bbox_inches="tight")
fig.savefig(str(out) + ".png", dpi=180, bbox_inches="tight")
print(f"Saved: {out}.pdf  and  {out}.png")
