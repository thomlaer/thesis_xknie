import math
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix

from config import final_dataset, llm_labels, data_root, gold_final, models_dir

input_definitief = final_dataset
input_llm        = llm_labels
output_txt       = data_root / "resultaten.txt"
n_bootstrap      = 1000
random_seed      = 2026
few_shot_example_case_ids = {11, 374, 437, 455}

gold_rename = {
    "definitief_handmatig_label": "handmatig_label",
    "definitief_artrose_aanwezig": "handmatig_artrose_aanwezig",
    "definitief_artrose_graad": "handmatig_artrose_graad",
    "definitief_nhg": "handmatig_nhg",
    "herkomst": "gold_herkomst",
}

gold_columns = [
    "case_id", "handmatig_label", "handmatig_artrose_aanwezig",
    "handmatig_artrose_graad", "handmatig_nhg",
    "label_martijn", "label_heleen",
    "label_artrose_martijn", "label_artrose_heleen",
    "artrose_graad_martijn", "artrose_graad_heleen",
    "nhg_martijn", "nhg_heleen", "gold_herkomst",
]


def pct(n, totaal):
    if totaal == 0:
        return "  n/a"
    return f"{100*n/totaal:.1f}%"


def bootstrap_f1(y_true, y_pred, pos_label="afwijkend", n=n_bootstrap, seed=random_seed):
    rng = np.random.default_rng(seed)
    scores = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        yt = [y_true[i] for i in s]
        yp = [y_pred[i] for i in s]
        uniek = set(yt)
        if len(uniek) < 2:
            continue
        scores.append(f1_score(yt, yp, average="macro", zero_division=0))
    if not scores:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def evalueer(y_true, y_pred, pos="afwijkend", neg="niet-afwijkend"):
    paren = [(t, p) for t, p in zip(y_true, y_pred) if t in (pos, neg) and p in (pos, neg)]
    if not paren:
        return None
    yt = [p[0] for p in paren]
    yp = [p[1] for p in paren]
    n = len(yt)
    acc = sum(a == b for a, b in zip(yt, yp)) / n
    kappa = cohen_kappa_score(yt, yp) if len(set(yt)) > 1 else float("nan")
    f1 = f1_score(yt, yp, average="macro", zero_division=0)
    lo, hi = bootstrap_f1(yt, yp, pos_label=pos)
    cm = confusion_matrix(yt, yp, labels=[pos, neg])
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return dict(n=n, acc=acc, f1=f1, ci_lo=lo, ci_hi=hi, kappa=kappa,
                sens=sens, spec=spec, prec=prec,
                tp=int(tp), fn=int(fn), tn=int(tn), fp=int(fp))


def merge_gold_with_primary(gold_bron, definitief):
    """Use final reviewer labels as gold and primary dataset columns for LLM labels."""
    gold_base = gold_bron.rename(columns=gold_rename)
    gold_base = gold_base[[c for c in gold_columns if c in gold_base.columns]].copy()
    primary_cols = [c for c in definitief.columns if c == "case_id" or c not in gold_base.columns]
    return gold_base.merge(definitief[primary_cols], on="case_id", how="left", validate="one_to_one")


def subset_for_variant(df, variant_name):
    if variant_name == "few-shot" and "case_id" in df.columns:
        return df[~df["case_id"].isin(few_shot_example_case_ids)].copy()
    return df


def fmt_eval(r):
    if r is None:
        return "no data"
    return (f"n={r['n']}, acc={r['acc']:.3f}, f1={r['f1']:.3f} "
            f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}], "
            f"kappa={r['kappa']:.3f}, "
            f"sens={r['sens']:.3f}, spec={r['spec']:.3f}")


def crosstab_pct(df, rij, kolom, rij_labels=None, kolom_labels=None):
    ct = pd.crosstab(df[rij], df[kolom])
    if rij_labels:
        ct = ct.reindex(rij_labels, fill_value=0)
    if kolom_labels:
        ct = ct.reindex(columns=kolom_labels, fill_value=0)
    rij_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    return ct, rij_pct


def write_table(lines, df_ct, df_pct, titel=""):
    if titel:
        lines.append(f"\n{titel}")
    lines.append(df_ct.to_string())
    lines.append("  row %:")
    lines.append(df_pct.round(1).to_string())


def main():
    lines = []

    def s(text=""):
        lines.append(text)

    gold_bron = gold_final

    definitief   = pd.read_excel(input_definitief)
    llm_volledig = pd.read_excel(input_llm)
    gold_bron    = pd.read_excel(gold_bron)

    primair_n     = len(definitief)
    uniek_primair = definitief["case_id"].nunique()

    gold = merge_gold_with_primary(gold_bron, definitief)
    n_gold = len(gold)

    s("llm evaluation results knee radiography")
    s(f"analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    s(f"llm labels: llm_labels.xlsx ({len(llm_volledig)} rows total)")
    s(f"final dataset: x_knie_definitief.xlsx")
    s(f"gold standard: {n_gold} records (manually labelled, consensus)")
    s()
    s("1. dataset overview")

    s()
    s(f"full llm dataset: {len(llm_volledig)} rows, {llm_volledig['case_id'].nunique()} unique cases")
    s(f"primary dataset (after mumc exclusions): {primair_n} rows, {uniek_primair} unique cases")
    s()
    s("laterality distribution (full llm dataset):")
    for z, c in llm_volledig["zijde"].value_counts().items():
        s(f"  {z:12s}  {c:5d}  ({pct(c, len(llm_volledig))})")

    s()
    s("is_split distribution:")
    for v, c in llm_volledig["is_split"].value_counts().items():
        s(f"  {str(v):8s}  {c:5d}  ({pct(c, len(llm_volledig))})")

    s()
    s("label_protesis (full dataset):")
    for v, c in llm_volledig["label_protesis"].value_counts().items():
        s(f"  {v:12s}  {c:5d}  ({pct(c, len(llm_volledig))})")

    s()
    s("2. exclusion analysis")

    excl_cols = [c for c in llm_volledig.columns if c.startswith("excl_")]
    if excl_cols:
        s()
        s(f"  criterion                              rows     %")
        for col in excl_cols:
            n = int(llm_volledig[col].sum())
            s(f"  {col:38s}  {n:5d}  ({pct(n, len(llm_volledig))})")

    s()
    s(f"  primary dataset (not excluded): {primair_n} rows, {uniek_primair} unique cases")

    s()
    s("2b. abnormal rate per exclusion group (llm consensus label)")
    s()
    s("  criterion                              n_excl  n_afwijkend  %_afwijkend")
    if excl_cols and "label_gecombineerd_consensus" in llm_volledig.columns:
        for col in excl_cols:
            mask = llm_volledig[col].astype(bool)
            n_excl_g = int(mask.sum())
            if n_excl_g == 0:
                continue
            n_afw = int((llm_volledig[mask]["label_gecombineerd_consensus"] == "afwijkend").sum())
            s(f"  {col:38s}  {n_excl_g:5d}   {n_afw:5d}  ({pct(n_afw, n_excl_g)})")

    s()
    s("2c. abnormal rate per exclusion group - by section and variant (findings / conclusion / combined)")
    s()
    s("  ZS = zero-shot, FS = few-shot, CS = consensus")
    s("  valid = afwijkend/niet-afwijkend")
    s()
    s("  group                          n   fnd-ZS fnd-FS fnd-CS   con-ZS con-FS con-CS   cmb-ZS cmb-FS cmb-CS")
    section_variant_cols = [
        ("fnd-ZS", "label_bevindingen"),
        ("fnd-FS", "label_bevindingen_fewshot"),
        ("fnd-CS", "label_bevindingen_consensus"),
        ("con-ZS", "label_conclusie"),
        ("con-FS", "label_conclusie_fewshot"),
        ("con-CS", "label_conclusie_consensus"),
        ("cmb-ZS", "label_gecombineerd"),
        ("cmb-FS", "label_gecombineerd_fewshot"),
        ("cmb-CS", "label_gecombineerd_consensus"),
    ]
    if excl_cols:
        for col in excl_cols:
            mask = llm_volledig[col].astype(bool)
            n_group = int(mask.sum())
            if n_group == 0:
                continue
            row = []
            for _, label_col in section_variant_cols:
                if label_col not in llm_volledig.columns:
                    row.append("   n/a")
                    continue
                labels = llm_volledig.loc[mask, label_col]
                valid = labels.isin(["afwijkend", "niet-afwijkend"])
                n_valid = int(valid.sum())
                n_afw = int((labels[valid] == "afwijkend").sum())
                row.append(pct(n_afw, n_valid))
            s(f"  {col:28s}  {n_group:5d}   " +
              " ".join(f"{v:>6s}" for v in row[0:3]) + "   " +
              " ".join(f"{v:>6s}" for v in row[3:6]) + "   " +
              " ".join(f"{v:>6s}" for v in row[6:9]))

    s()
    s("3. gold label distributions (consensus, n=%d)" % n_gold)

    s()
    s("radiological outcome:")
    for v, c in gold["handmatig_label"].value_counts().items():
        s(f"  {v:20s}  {c:4d}  ({pct(c, n_gold)})")

    if "handmatig_artrose_aanwezig" in gold.columns:
        s()
        s("arthrosis present:")
        for v, c in gold["handmatig_artrose_aanwezig"].value_counts().items():
            s(f"  {str(v):20s}  {c:4d}  ({pct(c, n_gold)})")

    if "handmatig_nhg" in gold.columns:
        s()
        s("nhg conformity:")
        for v, c in gold["handmatig_nhg"].value_counts().items():
            s(f"  {str(v):20s}  {c:4d}  ({pct(c, n_gold)})")

    if "handmatig_artrose_graad" in gold.columns:
        s()
        s("kl grade distribution:")
        for v, c in gold["handmatig_artrose_graad"].value_counts().sort_index().items():
            s(f"  grade {v}:  {c:4d}  ({pct(c, n_gold)})")

    if "gold_herkomst" in gold.columns:
        s()
        s("origin of gold records:")
        for v, c in gold["gold_herkomst"].value_counts().items():
            s(f"  {v:25s}  {c:4d}  ({pct(c, n_gold)})")

    if all(c in gold.columns for c in ["label_martijn", "label_heleen"]):
        s()
        s("3a. inter-rater agreement (martijn vs heleen)")
        s()
        sub = gold.dropna(subset=["label_martijn", "label_heleen"])
        sub = sub[sub["label_martijn"].isin(["afwijkend","niet-afwijkend"]) &
                  sub["label_heleen"].isin(["afwijkend","niet-afwijkend"])]
        if len(sub) > 0:
            n_sub = len(sub)
            ov = (sub["label_martijn"] == sub["label_heleen"]).mean()
            k = cohen_kappa_score(sub["label_martijn"], sub["label_heleen"])
            s(f"  outcome (n={n_sub}): agreement={ov:.1%}, kappa={k:.3f}")
            n_m_afw = (sub["label_martijn"] == "afwijkend").sum()
            n_h_afw = (sub["label_heleen"]  == "afwijkend").sum()
            s(f"    martijn afwijkend: {n_m_afw} ({pct(n_m_afw, n_sub)}),  "
              f"heleen afwijkend: {n_h_afw} ({pct(n_h_afw, n_sub)})")
            disagree = sub[sub["label_martijn"] != sub["label_heleen"]]
            s(f"    disagreements: {len(disagree)}")

    if all(c in gold.columns for c in ["nhg_martijn", "nhg_heleen"]):
        sub_nhg = gold.dropna(subset=["nhg_martijn","nhg_heleen"])
        sub_nhg = sub_nhg[sub_nhg["nhg_martijn"].isin(["ja","nee"]) &
                          sub_nhg["nhg_heleen"].isin(["ja","nee"])]
        if len(sub_nhg) > 0:
            ov_nhg = (sub_nhg["nhg_martijn"] == sub_nhg["nhg_heleen"]).mean()
            k_nhg  = cohen_kappa_score(sub_nhg["nhg_martijn"], sub_nhg["nhg_heleen"])
            s(f"  nhg (n={len(sub_nhg)}): agreement={ov_nhg:.1%}, kappa={k_nhg:.3f}")

    if all(c in gold.columns for c in ["label_artrose_martijn", "label_artrose_heleen"]):
        sub_a = gold.dropna(subset=["label_artrose_martijn","label_artrose_heleen"])
        sub_a = sub_a[sub_a["label_artrose_martijn"].isin(["ja","nee"]) &
                      sub_a["label_artrose_heleen"].isin(["ja","nee"])]
        if len(sub_a) > 0:
            ov_a = (sub_a["label_artrose_martijn"] == sub_a["label_artrose_heleen"]).mean()
            k_a  = cohen_kappa_score(sub_a["label_artrose_martijn"], sub_a["label_artrose_heleen"])
            s(f"  arthrosis present (n={len(sub_a)}): agreement={ov_a:.1%}, kappa={k_a:.3f}")

    if all(c in gold.columns for c in ["artrose_graad_martijn", "artrose_graad_heleen"]):
        sub_kl = gold.dropna(subset=["artrose_graad_martijn","artrose_graad_heleen"])
        sub_kl = sub_kl[
            pd.to_numeric(sub_kl["artrose_graad_martijn"], errors="coerce").notna() &
            pd.to_numeric(sub_kl["artrose_graad_heleen"],  errors="coerce").notna()
        ]
        if "handmatig_artrose_aanwezig" in sub_kl.columns:
            sub_kl = sub_kl[sub_kl["handmatig_artrose_aanwezig"] == "ja"]
        if "case_id" in sub_kl.columns:
            sub_kl = sub_kl[~sub_kl["case_id"].isin([1, 2])]
        if len(sub_kl) > 0:
            yt_kl = pd.to_numeric(sub_kl["artrose_graad_martijn"]).astype(int)
            yp_kl = pd.to_numeric(sub_kl["artrose_graad_heleen"]).astype(int)
            ov_kl = (yt_kl == yp_kl).mean()
            k_kl_uw = cohen_kappa_score(yt_kl, yp_kl, weights=None)
            k_kl_qw = cohen_kappa_score(yt_kl, yp_kl, weights="quadratic")
            s(f"  kl grade (n={len(sub_kl)}, consensus artrose=ja, excl cases 1-2): "
              f"exact agreement={ov_kl:.1%}, kappa unweighted={k_kl_uw:.3f}, "
              f"kappa quadratic={k_kl_qw:.3f}")

    s()
    s("4. llm evaluation vs gold standard (combined report)")

    s()
    s("  variant       sectie           n     acc     f1        95%-ci    kappa   sens   spec")

    varianten = [
        ("zero-shot",  "label_gecombineerd"),
        ("few-shot",   "label_gecombineerd_fewshot"),
        ("consensus",  "label_gecombineerd_consensus"),
    ]

    for naam, col in varianten:
        if col not in gold.columns:
            continue
        eval_gold = subset_for_variant(gold, naam)
        r = evalueer(eval_gold["handmatig_label"].tolist(), eval_gold[col].tolist())
        if r:
            s(f"  {naam:13s}  gecombineerd   {r['n']:4d}  {r['acc']:.3f}  {r['f1']:.3f}  "
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  {r['kappa']:.3f}  {r['sens']:.3f}  {r['spec']:.3f}")

    s()
    s("4a. by section (findings / conclusion / combined) - all variants")
    s("  few-shot excludes example cases")
    s()
    s("  variant       sectie           n     acc     f1        95%-ci    kappa   sens   spec")

    for naam, suf in [("zero-shot",""), ("few-shot","_fewshot"), ("consensus","_consensus")]:
        eval_gold = subset_for_variant(gold, naam)
        for sectie in ["bevindingen", "conclusie", "gecombineerd"]:
            col = f"label_{sectie}{suf}"
            if col not in eval_gold.columns:
                continue
            r = evalueer(eval_gold["handmatig_label"].tolist(), eval_gold[col].tolist())
            if r:
                s(f"  {naam:13s}  {sectie:15s}  {r['n']:4d}  {r['acc']:.3f}  {r['f1']:.3f}  "
                  f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  {r['kappa']:.3f}  {r['sens']:.3f}  {r['spec']:.3f}")

    s()
    s("4b. confusion matrices (combined)")

    for naam, col in varianten:
        if col not in gold.columns:
            continue
        eval_gold = subset_for_variant(gold, naam)
        sub = eval_gold[eval_gold["handmatig_label"].isin(["afwijkend","niet-afwijkend"]) &
                        eval_gold[col].isin(["afwijkend","niet-afwijkend"])]
        if len(sub) == 0:
            continue
        cm = pd.crosstab(sub["handmatig_label"], sub[col],
                         rownames=["gold"], colnames=[naam])
        s()
        s(f"  {naam} (n={len(sub)}):")
        s(cm.to_string())

    s()
    s("4c. per-label correct rates (combined section, gold standard)")
    s()
    s("  variant         afwijkend (n_true)  correct         niet-afwijkend (n_true)  correct")
    for naam, col in varianten:
        if col not in gold.columns:
            continue
        eval_gold = subset_for_variant(gold, naam)
        r = evalueer(eval_gold["handmatig_label"].tolist(), eval_gold[col].tolist())
        if r:
            n_afw_tot = r['tp'] + r['fn']
            n_niet_tot = r['tn'] + r['fp']
            s(f"  {naam:13s}    n={n_afw_tot:3d}  {r['tp']:3d}/{n_afw_tot} ({pct(r['tp'], n_afw_tot)})    "
              f"n={n_niet_tot:3d}  {r['tn']:3d}/{n_niet_tot} ({pct(r['tn'], n_niet_tot)})")

    s()
    s("4d. per-label correct rates - by section and variant (all)")
    s()
    s("  variant       sectie           afwijkend correct       niet-afwijkend correct")
    for naam, suf in [("zero-shot",""), ("few-shot","_fewshot"), ("consensus","_consensus")]:
        eval_gold = subset_for_variant(gold, naam)
        for sectie in ["bevindingen", "conclusie", "gecombineerd"]:
            col = f"label_{sectie}{suf}"
            if col not in eval_gold.columns:
                continue
            r = evalueer(eval_gold["handmatig_label"].tolist(), eval_gold[col].tolist())
            if r:
                n_afw_tot = r['tp'] + r['fn']
                n_niet_tot = r['tn'] + r['fp']
                s(f"  {naam:13s}  {sectie:15s}  {r['tp']:3d}/{n_afw_tot} ({pct(r['tp'], n_afw_tot)})    "
                  f"{r['tn']:3d}/{n_niet_tot} ({pct(r['tn'], n_niet_tot)})")

    s()
    s("5. nhg conformity analysis")

    if "handmatig_nhg" in gold.columns:
        s()
        s("5a. llm nhg vs manual nhg - all variants")
        s()
        s("  variant       n     acc     f1        95%-ci    kappa   sens   spec")
        nhg_varianten = [
            ("zero-shot",  "label_nhg"),
            ("few-shot",   "label_nhg_fewshot"),
            ("consensus",  "label_nhg_consensus"),
        ]
        for naam, col in nhg_varianten:
            if col not in gold.columns:
                continue
            sub = gold.dropna(subset=["handmatig_nhg", col])
            sub = sub[sub["handmatig_nhg"].isin(["ja","nee"]) & sub[col].isin(["ja","nee"])]
            if len(sub) == 0:
                continue
            r = evalueer(sub["handmatig_nhg"].tolist(), sub[col].tolist(), pos="ja", neg="nee")
            if r:
                s(f"  {naam:13s}  {r['n']:4d}  {r['acc']:.3f}  {r['f1']:.3f}  "
                  f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  {r['kappa']:.3f}  {r['sens']:.3f}  {r['spec']:.3f}")

        for naam, col in nhg_varianten:
            if col not in gold.columns:
                continue
            sub = gold.dropna(subset=["handmatig_nhg", col])
            sub = sub[sub["handmatig_nhg"].isin(["ja","nee"]) & sub[col].isin(["ja","nee"])]
            if len(sub) == 0:
                continue
            cm_nhg = pd.crosstab(sub["handmatig_nhg"], sub[col],
                                 rownames=["manual"], colnames=[naam])
            s()
            s(f"  {naam} (n={len(sub)}):")
            s(cm_nhg.to_string())

    if all(c in gold.columns for c in ["label_nhg_consensus", "handmatig_label"]):
        sub_ng = gold.dropna(subset=["label_nhg_consensus", "handmatig_label"])
        sub_ng = sub_ng[sub_ng["label_nhg_consensus"].isin(["ja", "nee"]) &
                        sub_ng["handmatig_label"].isin(["afwijkend", "niet-afwijkend"])]
        if len(sub_ng) > 0:
            s()
            s("5a (supplementary). nhg (llm consensus) x outcome (manual gold, n=%d)" % len(sub_ng))
            ct_ng = pd.crosstab(sub_ng["label_nhg_consensus"], sub_ng["handmatig_label"],
                                rownames=["nhg_consensus"], colnames=["gold_outcome"])
            ct_ng = ct_ng.reindex(["ja", "nee"], fill_value=0)
            ct_ng = ct_ng.reindex(columns=["afwijkend", "niet-afwijkend"], fill_value=0)
            ct_ng_pct = ct_ng.div(ct_ng.sum(axis=1), axis=0) * 100
            write_table(lines, ct_ng, ct_ng_pct)

    s()
    s("5b. nhg distribution primary dataset")
    s()
    s("  label        zero-shot       consensus")
    for v in ["ja", "nee", "onbekend"]:
        vals = []
        for col in ["label_nhg", "label_nhg_consensus"]:
            if col in definitief.columns:
                c = (definitief[col] == v).sum()
                vals.append(f"{c:5d} ({pct(c, primair_n)})")
            else:
                vals.append("  n/a       ")
        s(f"  {v:12s}  {'  '.join(vals)}")

    for naam, nhg_col in [("zero-shot","label_nhg"), ("consensus","label_nhg_consensus")]:
        if all(c in definitief.columns for c in [nhg_col, "label_trauma"]):
            s()
            s(f"5c. nhg x trauma ({naam}, primary dataset)")
            ct, pct_ct = crosstab_pct(definitief, nhg_col, "label_trauma",
                                      rij_labels=["ja","nee","onbekend"],
                                      kolom_labels=["trauma","oud_trauma","niet_trauma","onbekend"])
            write_table(lines, ct, pct_ct)

    for naam, nhg_col in [("zero-shot","label_nhg"), ("consensus","label_nhg_consensus")]:
        if all(c in definitief.columns for c in [nhg_col, "label_fractuur"]):
            s()
            s(f"5d. nhg x fracture suspicion ({naam}, primary dataset)")
            ct, pct_ct = crosstab_pct(definitief, nhg_col, "label_fractuur",
                                      rij_labels=["ja","nee","onbekend"],
                                      kolom_labels=["ja","nee","onbekend"])
            write_table(lines, ct, pct_ct)

    if all(c in gold.columns for c in ["handmatig_nhg", "handmatig_label"]):
        s()
        s("5e. nhg x outcome (manual, gold n=%d)" % n_gold)
        ct, pct_ct = crosstab_pct(gold, "handmatig_nhg", "handmatig_label",
                                  rij_labels=["ja","nee","onbekend"],
                                  kolom_labels=["afwijkend","niet-afwijkend"])
        write_table(lines, ct, pct_ct)

    if all(c in definitief.columns for c in ["label_nhg_consensus","label_gecombineerd_consensus"]):
        s()
        s("5f. nhg (llm consensus) x outcome (llm consensus), primary dataset")
        ct, pct_ct = crosstab_pct(definitief, "label_nhg_consensus", "label_gecombineerd_consensus",
                                  rij_labels=["ja","nee","onbekend"],
                                  kolom_labels=["afwijkend","niet-afwijkend","onbekend"])
        write_table(lines, ct, pct_ct)

    for naam, nhg_col in [("zero-shot","label_nhg"), ("consensus","label_nhg_consensus")]:
        if all(c in definitief.columns for c in [nhg_col, "artrose_radiologie"]):
            s()
            s(f"5g. nhg x arthrosis (regex radiology) ({naam}, primary dataset)")
            ct, pct_ct = crosstab_pct(definitief, nhg_col, "artrose_radiologie",
                                      rij_labels=["ja","nee","onbekend"],
                                      kolom_labels=["ja","nee"])
            write_table(lines, ct, pct_ct)

    s()
    s("6. arthrosis analysis")

    if all(c in gold.columns for c in ["artrose_radiologie", "handmatig_artrose_aanwezig"]):
        s()
        s("6a. arthrosis detection regex vs manual (n=%d)" % n_gold)
        sub_a = gold.dropna(subset=["artrose_radiologie","handmatig_artrose_aanwezig"])
        sub_a = sub_a[sub_a["artrose_radiologie"].isin(["ja","nee"]) &
                      sub_a["handmatig_artrose_aanwezig"].isin(["ja","nee"])]
        if len(sub_a) > 0:
            r = evalueer(sub_a["handmatig_artrose_aanwezig"].tolist(),
                         sub_a["artrose_radiologie"].tolist(), pos="ja", neg="nee")
            if r:
                s(f"  n={r['n']}, acc={r['acc']:.3f}, f1={r['f1']:.3f} "
                  f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}], kappa={r['kappa']:.3f}, "
                  f"sens={r['sens']:.3f}, spec={r['spec']:.3f}")
            cm_a = pd.crosstab(sub_a["handmatig_artrose_aanwezig"], sub_a["artrose_radiologie"],
                               rownames=["manual"], colnames=["regex"])
            s(cm_a.to_string())

    if all(c in definitief.columns for c in ["artrose_vraagstelling","artrose_radiologie"]):
        s()
        s("6b. arthrosis in referral question x arthrosis in radiology (primary dataset)")
        ct, pct_ct = crosstab_pct(definitief, "artrose_vraagstelling", "artrose_radiologie",
                                  rij_labels=["ja","nee"], kolom_labels=["ja","nee"])
        write_table(lines, ct, pct_ct)

    s()
    s("6c. kl grade evaluation vs gold - all variants")
    kl_varianten = [
        ("zero-shot",  "artrose_graad"),
        ("few-shot",   "artrose_graad_fewshot"),
        ("consensus",  "artrose_graad_consensus"),
    ]
    for naam, col in kl_varianten:
        if col not in gold.columns or "handmatig_artrose_graad" not in gold.columns:
            continue
        sub_kl = gold.dropna(subset=["handmatig_artrose_graad", col])
        sub_kl = sub_kl[
            pd.to_numeric(sub_kl["handmatig_artrose_graad"], errors="coerce").notna() &
            pd.to_numeric(sub_kl[col], errors="coerce").notna()
        ]
        if len(sub_kl) == 0:
            continue
        yt = pd.to_numeric(sub_kl["handmatig_artrose_graad"]).astype(int)
        yp = pd.to_numeric(sub_kl[col]).astype(int)
        exact   = (yt == yp).mean()
        within1 = (abs(yt - yp) <= 1).mean()
        kappa_uw = cohen_kappa_score(yt, yp, weights=None)
        kappa_qw = cohen_kappa_score(yt, yp, weights="quadratic")
        s()
        s(f"  {naam} (n={len(sub_kl)}): exact agreement={exact:.1%}, within-1={within1:.1%}, "
          f"kappa unweighted={kappa_uw:.3f}, kappa quadratic={kappa_qw:.3f}")
        ct_kl = pd.crosstab(yt.rename("gold"), yp.rename(naam))
        s(ct_kl.to_string())

    if all(c in gold.columns for c in ["handmatig_artrose_aanwezig","handmatig_label"]):
        s()
        s("6d. arthrosis present x outcome (manual, gold)")
        ct, pct_ct = crosstab_pct(gold, "handmatig_artrose_aanwezig", "handmatig_label",
                                  rij_labels=["ja","nee"],
                                  kolom_labels=["afwijkend","niet-afwijkend"])
        write_table(lines, ct, pct_ct)

    if all(c in gold.columns for c in ["handmatig_artrose_aanwezig","handmatig_nhg"]):
        s()
        s("6e. arthrosis x nhg conformity (manual, gold)")
        ct, pct_ct = crosstab_pct(gold, "handmatig_artrose_aanwezig", "handmatig_nhg",
                                  rij_labels=["ja","nee"],
                                  kolom_labels=["ja","nee","onbekend"])
        write_table(lines, ct, pct_ct)

    if all(c in definitief.columns for c in ["artrose_radiologie","label_gecombineerd_consensus"]):
        s()
        s("6f. arthrosis radiology x consensus outcome (primary dataset)")
        ct, pct_ct = crosstab_pct(definitief, "artrose_radiologie", "label_gecombineerd_consensus",
                                  rij_labels=["ja","nee"],
                                  kolom_labels=["afwijkend","niet-afwijkend","onbekend"])
        write_table(lines, ct, pct_ct)

    s()
    s("6g. artrose_radiologie distribution primary dataset")
    for v in ["ja","nee"]:
        c = (definitief["artrose_radiologie"] == v).sum() if "artrose_radiologie" in definitief.columns else 0
        s(f"  {v}: {c} ({pct(c, primair_n)})")

    s()
    s("7. cross-tables (primary dataset) - consensus as primary variant, zero-shot for comparison")

    def dubbel_crosstab(rij_col, kolom_col, rij_labels, kolom_labels, titel_a, titel_b, col_a, col_b):
        for naam, col in [(titel_a, col_a), (titel_b, col_b)]:
            if not all(c in definitief.columns for c in [rij_col, col]):
                continue
            ct, pct_ct = crosstab_pct(definitief, rij_col, col,
                                      rij_labels=rij_labels,
                                      kolom_labels=kolom_labels)
            write_table(lines, ct, pct_ct, titel=f"  {naam}:")

    s()
    s("7a. trauma x outcome (llm combined)")
    dubbel_crosstab(
        "label_trauma", None,
        ["trauma","oud_trauma","niet_trauma","onbekend"],
        ["afwijkend","niet-afwijkend","onbekend"],
        "consensus", "zero-shot",
        "label_gecombineerd_consensus", "label_gecombineerd"
    )

    s()
    s("7b. arthrosis in referral question x outcome (llm combined)")
    dubbel_crosstab(
        "artrose_vraagstelling", None,
        ["ja","nee"],
        ["afwijkend","niet-afwijkend","onbekend"],
        "consensus", "zero-shot",
        "label_gecombineerd_consensus", "label_gecombineerd"
    )

    if all(c in definitief.columns for c in ["artrose_vraagstelling", "label_gecombineerd_consensus"]) and\
       all(c in gold.columns for c in ["handmatig_artrose_aanwezig", "handmatig_label"]):
        sub_aq = definitief[definitief["artrose_vraagstelling"] == "ja"]
        n_aq_afw = int((sub_aq["label_gecombineerd_consensus"] == "afwijkend").sum())
        gold_art = gold[gold["handmatig_artrose_aanwezig"].isin(["ja"]) &
                        gold["handmatig_label"].isin(["afwijkend", "niet-afwijkend"])]
        n_ga_afw = int((gold_art["handmatig_label"] == "afwijkend").sum())
        s()
        s("  comparison")
        s(f"  arthrosis in referral question (consensus, primary, n={len(sub_aq)}) "
          f"-> {pct(n_aq_afw, len(sub_aq))} abnormal (llm label)")
        s(f"  arthrosis present in radiology report (manual gold, n={len(gold_art)}) "
          f"-> {pct(n_ga_afw, len(gold_art))} abnormal (manual label, sec 6d)")

    s()
    s("7c. fracture suspicion x outcome (llm combined)")
    dubbel_crosstab(
        "label_fractuur", None,
        ["ja","nee","onbekend"],
        ["afwijkend","niet-afwijkend","onbekend"],
        "consensus", "zero-shot",
        "label_gecombineerd_consensus", "label_gecombineerd"
    )

    s()
    s("7d. trauma x nhg conformity")
    dubbel_crosstab(
        "label_trauma", None,
        ["trauma","oud_trauma","niet_trauma","onbekend"],
        ["ja","nee","onbekend"],
        "consensus nhg", "zero-shot nhg",
        "label_nhg_consensus", "label_nhg"
    )

    s()
    s("7e. fracture suspicion x nhg conformity")
    dubbel_crosstab(
        "label_fractuur", None,
        ["ja","nee","onbekend"],
        ["ja","nee","onbekend"],
        "consensus nhg", "zero-shot nhg",
        "label_nhg_consensus", "label_nhg"
    )

    if all(c in definitief.columns for c in ["label_trauma","artrose_vraagstelling"]):
        s()
        s("7f. trauma x arthrosis in referral question (primary dataset)")
        ct, pct_ct = crosstab_pct(definitief, "label_trauma", "artrose_vraagstelling",
                                  rij_labels=["trauma","oud_trauma","niet_trauma","onbekend"],
                                  kolom_labels=["ja","nee"])
        write_table(lines, ct, pct_ct)

    if all(c in definitief.columns for c in ["label_trauma","artrose_radiologie"]):
        s()
        s("7g. trauma x arthrosis radiology (primary dataset)")
        ct, pct_ct = crosstab_pct(definitief, "label_trauma", "artrose_radiologie",
                                  rij_labels=["trauma","oud_trauma","niet_trauma","onbekend"],
                                  kolom_labels=["ja","nee"])
        write_table(lines, ct, pct_ct)

    s()
    s("7h. arthrosis in referral question x nhg conformity")
    dubbel_crosstab(
        "artrose_vraagstelling", None,
        ["ja","nee"],
        ["ja","nee","onbekend"],
        "consensus nhg", "zero-shot nhg",
        "label_nhg_consensus", "label_nhg"
    )

    s()
    s("8. variant agreement (combined, primary dataset)")

    paren = [
        ("zero-shot", "few-shot",  "label_gecombineerd",          "label_gecombineerd_fewshot"),
        ("zero-shot", "consensus", "label_gecombineerd",          "label_gecombineerd_consensus"),
        ("few-shot",  "consensus", "label_gecombineerd_fewshot",  "label_gecombineerd_consensus"),
    ]
    s()
    s("  variant a       variant b       n       agree   agree%   kappa")
    for a_naam, b_naam, col_a, col_b in paren:
        if col_a not in definitief.columns or col_b not in definitief.columns:
            continue
        sub = definitief.dropna(subset=[col_a, col_b])
        eens = (sub[col_a] == sub[col_b]).sum()
        n_sub = len(sub)
        try:
            k = cohen_kappa_score(sub[col_a], sub[col_b])
        except Exception:
            k = float("nan")
        s(f"  {a_naam:14s}  {b_naam:14s}  {n_sub:6d}  {eens:6d}  {pct(eens, n_sub)}  {k:.3f}")

    nhg_paren = [
        ("zero-shot nhg", "few-shot nhg",  "label_nhg",          "label_nhg_fewshot"),
        ("zero-shot nhg", "consensus nhg", "label_nhg",          "label_nhg_consensus"),
        ("few-shot nhg",  "consensus nhg", "label_nhg_fewshot",  "label_nhg_consensus"),
    ]
    s()
    s("  nhg variant agreement:")
    for a_naam, b_naam, col_a, col_b in nhg_paren:
        if col_a not in definitief.columns or col_b not in definitief.columns:
            continue
        sub = definitief.dropna(subset=[col_a, col_b])
        eens = (sub[col_a] == sub[col_b]).sum()
        n_sub = len(sub)
        try:
            k = cohen_kappa_score(sub[col_a], sub[col_b])
        except Exception:
            k = float("nan")
        s(f"  {a_naam:20s}  {b_naam:20s}  {n_sub:6d}  {eens:6d}  {pct(eens, n_sub)}  {k:.3f}")


    s()
    s("9. llm label distributions primary dataset")

    s()
    s("  label                   zero-shot       few-shot       consensus")
    for sectie in ["bevindingen", "conclusie", "gecombineerd"]:
        s()
        s(f"  {sectie}:")
        for label in ["afwijkend", "niet-afwijkend", "onbekend"]:
            vals = []
            for suf in ["", "_fewshot", "_consensus"]:
                col = f"label_{sectie}{suf}"
                if col in definitief.columns:
                    c = (definitief[col] == label).sum()
                    vals.append(f"{c:5d} ({pct(c, primair_n)})")
                else:
                    vals.append("  n/a         ")
            s(f"  {label:18s}  {'  '.join(vals)}")

    s()
    s("  artrose_graad distribution (primary dataset):")
    s()
    s("  grade   zero-shot       few-shot       consensus")
    for g in [0, 1, 2, 3, 4, "onbekend"]:
        vals = []
        for col in ["artrose_graad", "artrose_graad_fewshot", "artrose_graad_consensus"]:
            if col in definitief.columns:
                c = (definitief[col].astype(str) == str(g)).sum()
                vals.append(f"{c:5d} ({pct(c, primair_n)})")
            else:
                vals.append("  n/a         ")
        s(f"  {str(g):8s}  {'  '.join(vals)}")

    s()
    s("  nhg distribution (primary dataset):")
    s()
    s("  label        zero-shot       few-shot       consensus")
    for v in ["ja", "nee", "onbekend"]:
        vals = []
        for col in ["label_nhg", "label_nhg_fewshot", "label_nhg_consensus"]:
            if col in definitief.columns:
                c = (definitief[col] == v).sum()
                vals.append(f"{c:5d} ({pct(c, primair_n)})")
            else:
                vals.append("  n/a         ")
        s(f"  {v:12s}  {'  '.join(vals)}")

    s()
    s("10. llm consistency (findings vs conclusion, primary dataset)")

    for naam, suf in [("zero-shot",""), ("consensus","_consensus")]:
        col_b = f"label_bevindingen{suf}"
        col_c = f"label_conclusie{suf}"
        if not all(c in definitief.columns for c in [col_b, col_c]):
            continue
        sub_con = definitief.dropna(subset=[col_b, col_c])
        eens = (sub_con[col_b] == sub_con[col_c]).sum()
        n_con = len(sub_con)
        try:
            k_con = cohen_kappa_score(sub_con[col_b], sub_con[col_c])
        except Exception:
            k_con = float("nan")
        s()
        s(f"  {naam}: n={n_con},  agree={eens} ({pct(eens, n_con)}),  kappa={k_con:.3f}")

    s()
    s("  most common disagreements findings vs conclusion (zero-shot, primary dataset):")
    col_b, col_c = "label_bevindingen", "label_conclusie"
    if all(c in definitief.columns for c in [col_b, col_c]):
        sub_con = definitief.dropna(subset=[col_b, col_c])
        pairs_dis = sub_con[sub_con[col_b] != sub_con[col_c]]
        for (bev, concl), cnt in (pairs_dis
                .groupby([col_b, col_c])
                .size()
                .sort_values(ascending=False)
                .head(6)
                .items()):
            s(f"    {bev} -> {concl}: {cnt}")

    s()
    s("11. confidence analysis (zero-shot combined, primary dataset)")

    conf_cols = ["conf_afwijkend_gecombineerd",
                 "conf_niet_afwijkend_gecombineerd",
                 "conf_onbekend_gecombineerd"]
    avail_conf = [c for c in conf_cols if c in definitief.columns]
    if avail_conf:
        conf_max = definitief[avail_conf].max(axis=1)
        conf_vals = conf_max.dropna()
        s()
        s(f"  n={len(conf_vals)}")
        s(f"  mean: {conf_vals.mean():.3f},  median: {conf_vals.median():.3f},  min: {conf_vals.min():.3f}")
        for drempel in [0.95, 0.90, 0.80, 0.70]:
            boven = (conf_vals >= drempel).sum()
            s(f"  >= {drempel:.0%}: {boven} ({pct(boven, len(conf_vals))})")

        if "label_gecombineerd" in gold.columns:
            gold_conf = gold[
                gold["handmatig_label"].isin(["afwijkend","niet-afwijkend"]) &
                gold["label_gecombineerd"].isin(["afwijkend","niet-afwijkend"])
            ].copy()
            avail2 = [c for c in avail_conf if c in gold_conf.columns]
            if avail2:
                gold_conf["correct"] = (gold_conf["handmatig_label"] == gold_conf["label_gecombineerd"])
                gold_conf["conf_max"] = gold_conf[avail2].max(axis=1)
                c_correct = gold_conf[gold_conf["correct"]]["conf_max"].mean()
                c_fout    = gold_conf[~gold_conf["correct"]]["conf_max"].mean()
                s()
                s(f"  mean confidence correct:   {c_correct:.3f}")
                s(f"  mean confidence incorrect: {c_fout:.3f}")

                s()
                s("  overconfidence analysis - errors per confidence bin (zero-shot combined, gold n=%d):" % len(gold_conf))
                s()
                s("  bin         n_total    n_correct   n_error  %_correct   %_error")
                bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.001)]
                bin_labels = ["0.00-0.50", "0.50-0.70", "0.70-0.80", "0.80-0.90", "0.90-0.95", "0.95-1.00"]
                for (lo, hi), label in zip(bins, bin_labels):
                    in_bin = gold_conf[(gold_conf["conf_max"] >= lo) & (gold_conf["conf_max"] < hi)]
                    n_tot = len(in_bin)
                    if n_tot == 0:
                        continue
                    n_cor = in_bin["correct"].sum()
                    n_fou = n_tot - n_cor
                    s(f"  {label}   {n_tot:6d}     {n_cor:6d}   {n_fou:6d}   {pct(n_cor, n_tot):>8s}   {pct(n_fou, n_tot):>8s}")

                try:
                    from scipy.stats import pearsonr
                    r_val, p_val = pearsonr(gold_conf["conf_max"].fillna(0), gold_conf["correct"].astype(float))
                    s()
                    s(f"  pearson r (conf_max ~ correct): r={r_val:.3f}, p={p_val:.4f}")
                except ImportError:
                    pass

    s()
    s("12. summary statistics")

    s()
    s("llm combined vs gold (n_gold=%d):" % n_gold)
    s()
    s("  variant       n     acc     f1        95%-ci    kappa   sens   spec   tp   fn   fp   tn")
    for naam, col in varianten:
        if col not in gold.columns:
            continue
        eval_gold = subset_for_variant(gold, naam)
        r = evalueer(eval_gold["handmatig_label"].tolist(), eval_gold[col].tolist())
        if r:
            s(f"  {naam:13s}  {r['n']:4d}  {r['acc']:.3f}  {r['f1']:.3f}  "
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  {r['kappa']:.3f}  "
              f"{r['sens']:.3f}  {r['spec']:.3f}  {r['tp']:3d}  {r['fn']:3d}  {r['fp']:3d}  {r['tn']:3d}")

    s()
    s("by section (all variants):")
    s()
    s("  variant       sectie           n     acc     f1        95%-ci    kappa   sens   spec")
    for naam, suf in [("zero-shot",""), ("few-shot","_fewshot"), ("consensus","_consensus")]:
        eval_gold = subset_for_variant(gold, naam)
        for sectie in ["bevindingen", "conclusie", "gecombineerd"]:
            col = f"label_{sectie}{suf}"
            if col not in eval_gold.columns:
                continue
            r = evalueer(eval_gold["handmatig_label"].tolist(), eval_gold[col].tolist())
            if r:
                s(f"  {naam:13s}  {sectie:15s}  {r['n']:4d}  {r['acc']:.3f}  {r['f1']:.3f}  "
                  f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  {r['kappa']:.3f}  {r['sens']:.3f}  {r['spec']:.3f}")

    s()
    s("nhg llm vs manual:")
    s()
    s("  variant       n     acc     f1        95%-ci    kappa   sens   spec")
    for naam, col in [("zero-shot","label_nhg"), ("few-shot","label_nhg_fewshot"), ("consensus","label_nhg_consensus")]:
        if col not in gold.columns or "handmatig_nhg" not in gold.columns:
            continue
        sub = gold.dropna(subset=["handmatig_nhg", col])
        sub = sub[sub["handmatig_nhg"].isin(["ja","nee"]) & sub[col].isin(["ja","nee"])]
        if len(sub) == 0:
            continue
        r = evalueer(sub["handmatig_nhg"].tolist(), sub[col].tolist(), pos="ja", neg="nee")
        if r:
            s(f"  {naam:13s}  {r['n']:4d}  {r['acc']:.3f}  {r['f1']:.3f}  "
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  {r['kappa']:.3f}  {r['sens']:.3f}  {r['spec']:.3f}")

    s()
    s("kl grade vs manual:")
    for naam, col in kl_varianten:
        if col not in gold.columns or "handmatig_artrose_graad" not in gold.columns:
            continue
        sub_kl = gold.dropna(subset=["handmatig_artrose_graad", col])
        sub_kl = sub_kl[
            pd.to_numeric(sub_kl["handmatig_artrose_graad"], errors="coerce").notna() &
            pd.to_numeric(sub_kl[col], errors="coerce").notna()
        ]
        if len(sub_kl) == 0:
            continue
        yt = pd.to_numeric(sub_kl["handmatig_artrose_graad"]).astype(int)
        yp = pd.to_numeric(sub_kl[col]).astype(int)
        s(f"  {naam}: n={len(sub_kl)}, exact={(yt==yp).mean():.1%}, "
          f"within-1={(abs(yt-yp)<=1).mean():.1%}, "
          f"kappa_uw={cohen_kappa_score(yt, yp, weights=None):.3f}, "
          f"kappa_qw={cohen_kappa_score(yt, yp, weights='quadratic'):.3f}")

    s()
    s("dataset:")
    s(f"  full llm dataset: {len(llm_volledig)} rows, {llm_volledig['case_id'].nunique()} unique cases")
    s(f"  primary dataset (after mumc exclusions): {primair_n} rows, {uniek_primair} unique cases")
    s(f"  gold standard: {n_gold} records")

    import json as _json

    models_dir_path = models_dir

    s()
    s("13. prediction model results (gold test, n=370)")
    s()
    s("13a. classical models (tfidf-based, gold test)")
    s()
    s("  model                        n     acc     f1-macro  afwijkend_%correct  niet-afwijkend_%correct")

    classical = [
        ("majority baseline",    "majority_baseline_minwords0_gold_test_metrics.json"),
        ("tfidf log.regression", "tfidf_logreg_minwords0_gold_test_metrics.json"),
        ("tfidf MLP",            "tfidf_mlp_minwords0_gold_test_metrics.json"),
        ("tfidf XGBoost",        "tfidf_xgboost_minwords0_gold_test_metrics.json"),
    ]

    for naam, fname in classical:
        p = models_dir_path / fname
        if not p.exists():
            continue
        try:
            m = _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        n_m   = m.get("n", "?")
        acc_v = m.get("accuracy", {}).get("value")
        f1_v  = m.get("macro_f1", {}).get("value")
        sens_v = m.get("sensitivity_afwijkend", {}).get("value")
        spec_v = m.get("specificity_non_abnormal", {}).get("value")
        acc_s  = f"{acc_v:.3f}" if acc_v is not None else "n/a"
        f1_s   = f"{f1_v:.3f}" if f1_v is not None else "n/a"
        sens_s = f"{100*sens_v:.1f}%" if sens_v is not None else "n/a"
        spec_s = f"{100*spec_v:.1f}%" if spec_v is not None else "n/a"
        s(f"  {naam:28s}  {n_m:4}  {acc_s}   {f1_s}     {sens_s:<14s}  {spec_s}")

    
    s()
    s("13b. bert models (MedRoBERTa.nl fine-tuned, gold test)")
    s()
    s("  scenario                                   n     acc     f1-macro  afwijkend_%correct  niet-afwijkend_%correct")

    bert_scenarios = [
        ("A - GP only (silver train)",          models_dir_path / "bert" / "gp_only"),
        ("C - GP + radiology combined",         models_dir_path / "bert" / "gp_plus_radiology"),
    ]

    for naam, bert_dir in bert_scenarios:
        metrics_p = bert_dir / "metrics.json"
        preds_p   = bert_dir / "gold_predictions.csv"
        if not metrics_p.exists():
            s(f"  {naam:42s}  (not yet available)")
            continue
        try:
            m = _json.loads(metrics_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        n_m   = m.get("n_gold", "?")
        acc_v = m.get("accuracy")
        f1_v  = m.get("macro_f1")
        acc_s = f"{acc_v:.3f}" if acc_v is not None else "n/a"
        f1_s  = f"{f1_v:.3f}" if f1_v is not None else "n/a"

        
        
        sens_s, spec_s = "n/a", "n/a"
        if m.get("sensitivity_afwijkend") is not None:
            sens_s = f"{100*m['sensitivity_afwijkend']:.1f}%"
        if m.get("specificity_niet_afwijkend") is not None:
            spec_s = f"{100*m['specificity_niet_afwijkend']:.1f}%"
        if (sens_s == "n/a" or spec_s == "n/a") and preds_p.exists():
            try:
                preds = pd.read_csv(preds_p, encoding="utf-8-sig")
                if "y_true" in preds.columns and "y_pred" in preds.columns:
                    afw_m  = preds["y_true"] == "afwijkend"
                    niet_m = preds["y_true"] == "niet-afwijkend"
                    if afw_m.sum() > 0:
                        n_tp = int((preds.loc[afw_m, "y_pred"] == "afwijkend").sum())
                        sens_s = f"{n_tp}/{int(afw_m.sum())} ({pct(n_tp, int(afw_m.sum()))})"
                    if niet_m.sum() > 0:
                        n_tn = int((preds.loc[niet_m, "y_pred"] == "niet-afwijkend").sum())
                        spec_s = f"{n_tn}/{int(niet_m.sum())} ({pct(n_tn, int(niet_m.sum()))})"
            except Exception:
                pass

        s(f"  {naam:42s}  {n_m:4}  {acc_s}   {f1_s}     {sens_s:<22s}  {spec_s}")

    output = "\n".join(lines)
    Path(output_txt).write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
