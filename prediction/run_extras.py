import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import models_dir


labels = ["afwijkend", "niet-afwijkend"]


def load_gold_preds(prefix):
    path = models_dir / f"{prefix}_gold_test_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def metrics_from_preds(y_true, y_pred):
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)

    n_afw = int((y_true == "afwijkend").sum())
    n_niet = int((y_true == "niet-afwijkend").sum())
    tp = int(((y_true == "afwijkend") & (y_pred == "afwijkend")).sum())
    tn = int(((y_true == "niet-afwijkend") & (y_pred == "niet-afwijkend")).sum())

    return {
        "n": int(len(y_true)),
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "sens": tp / n_afw if n_afw else float("nan"),
        "spec": tn / n_niet if n_niet else float("nan"),
        "tp": tp,
        "tn": tn,
        "n_afw": n_afw,
        "n_niet": n_niet,
    }


def mcnemar(y_true, pred_a, pred_b):
    correct_a = np.array(pred_a) == np.array(y_true)
    correct_b = np.array(pred_b) == np.array(y_true)

    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if b + c == 0:
        return b, c, float("nan"), float("nan"), ""

    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(stat, df=1)
    marker = "*" if p_value < 0.05 else ("~" if p_value < 0.10 else "")
    return b, c, stat, p_value, marker


def section_top_features(n=20):
    print()
    print("1. top tf-idf features")

    path = models_dir / "tfidf_logreg_minwords0_top_words.csv"
    if not path.exists():
        print("  file not found")
        return

    df = pd.read_csv(path, encoding="utf-8-sig")
    for class_name in labels:
        sub = df[df["class"] == class_name].nlargest(n, "coefficient")
        print()
        print(f"  {class_name}")
        for _, row in sub.iterrows():
            feature = row["feature"].replace("tfidf__", "").replace("manual__", "manual:")
            print(f"    {feature:<40} {row['coefficient']:.3f}")


def section_mcnemar(all_preds):
    print()
    print("2. McNemar test")
    print("  b = model A correct, model B wrong")
    print("  c = model A wrong, model B correct")

    names = list(all_preds)
    if len(names) < 2:
        print("  not enough model predictions")
        return

    case_ids = set(all_preds[names[0]]["case_id"].tolist())
    for df in all_preds.values():
        case_ids &= set(df["case_id"].tolist())
    case_ids = sorted(case_ids)
    print(f"  common test cases: {len(case_ids)}")

    aligned = {
        name: df[df["case_id"].isin(case_ids)].set_index("case_id")
        for name, df in all_preds.items()
    }
    y_true = aligned[names[0]]["y_true"].loc[case_ids].tolist()

    print(f"  {'model A':<32} {'model B':<32} {'b':>4} {'c':>4} {'chi2':>7} {'p':>8}")
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            pred_a = aligned[name_a]["y_pred"].loc[case_ids].tolist()
            pred_b = aligned[name_b]["y_pred"].loc[case_ids].tolist()
            b, c, stat, p_value, marker = mcnemar(y_true, pred_a, pred_b)
            stat_s = "n/a" if np.isnan(stat) else f"{stat:.2f}"
            p_s = "n/a" if np.isnan(p_value) else f"{p_value:.4f}"
            print(f"  {name_a:<32} {name_b:<32} {b:>4} {c:>4} {stat_s:>7} {p_s:>8} {marker}")


def section_wordcount_subgroups(all_preds):
    print()
    print("3. word-count subgroups")

    base_df = next((df for df in all_preds.values() if "word_count" in df.columns), None)
    if base_df is None:
        print("  word_count column not found")
        return

    groups = {
        "short (<10)": set(base_df[base_df["word_count"] < 10]["case_id"].tolist()),
        "long (>=10)": set(base_df[base_df["word_count"] >= 10]["case_id"].tolist()),
    }

    for group_name, ids in groups.items():
        print()
        print(f"  {group_name}: {len(ids)} cases")
        print(f"  {'model':<32} {'n':>4} {'acc':>6} {'f1':>6} {'afw':>12} {'niet-afw':>12}")
        for name, df in all_preds.items():
            sub = df[df["case_id"].isin(ids)]
            if sub.empty:
                continue
            metrics = metrics_from_preds(sub["y_true"], sub["y_pred"])
            sens = f"{metrics['tp']}/{metrics['n_afw']} ({100 * metrics['sens']:.1f}%)"
            spec = f"{metrics['tn']}/{metrics['n_niet']} ({100 * metrics['spec']:.1f}%)"
            print(f"  {name:<32} {metrics['n']:>4} {metrics['acc']:>6.3f} {metrics['f1']:>6.3f} {sens:>12} {spec:>12}")


def section_minwords10():
    print()
    print("4. min_words=10 comparison")

    models = {
        "majority baseline": ("majority_baseline_minwords0", "majority_baseline_minwords10"),
        "tfidf logreg": ("tfidf_logreg_minwords0", "tfidf_logreg_minwords10"),
        "tfidf MLP": ("tfidf_mlp_minwords0", "tfidf_mlp_minwords10"),
        "tfidf XGBoost": ("tfidf_xgboost_minwords0", "tfidf_xgboost_minwords10"),
    }

    if not any((models_dir / f"{prefix10}_gold_test_predictions.csv").exists() for _, prefix10 in models.values()):
        print("  min_words=10 files not found")
        return

    print(f"  {'model':<24} {'filter':>6} {'n':>4} {'acc':>6} {'f1':>6} {'kappa':>7} {'afw':>12} {'niet-afw':>12}")
    for model_name, (prefix0, prefix10) in models.items():
        for filter_name, prefix in [("mw=0", prefix0), ("mw=10", prefix10)]:
            df = load_gold_preds(prefix)
            if df is None:
                continue
            metrics = metrics_from_preds(df["y_true"], df["y_pred"])
            sens = f"{metrics['tp']}/{metrics['n_afw']} ({100 * metrics['sens']:.1f}%)"
            spec = f"{metrics['tn']}/{metrics['n_niet']} ({100 * metrics['spec']:.1f}%)"
            print(
                f"  {model_name:<24} {filter_name:>6} {metrics['n']:>4} "
                f"{metrics['acc']:>6.3f} {metrics['f1']:>6.3f} {metrics['kappa']:>7.3f} "
                f"{sens:>12} {spec:>12}"
            )


def load_all_predictions():
    classical = {
        "majority_baseline": "majority_baseline_minwords0",
        "tfidf_logreg": "tfidf_logreg_minwords0",
        "tfidf_mlp": "tfidf_mlp_minwords0",
        "tfidf_xgboost": "tfidf_xgboost_minwords0",
    }
    bert = {
        "bert_gp_only": models_dir / "bert" / "gp_only" / "gold_predictions.csv",
        "bert_gp_plus_radiology": models_dir / "bert" / "gp_plus_radiology" / "gold_predictions.csv",
    }

    all_preds = {}
    for name, prefix in classical.items():
        df = load_gold_preds(prefix)
        if df is not None:
            all_preds[name] = df

    for name, path in bert.items():
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8-sig")
            if "case_id" not in df.columns:
                df = df.reset_index().rename(columns={"index": "case_id"})
            all_preds[name] = df

    return all_preds


def main():
    print("extra prediction checks")
    all_preds = load_all_predictions()
    section_top_features()
    section_mcnemar(all_preds)
    section_wordcount_subgroups(all_preds)
    section_minwords10()


if __name__ == "__main__":
    main()
