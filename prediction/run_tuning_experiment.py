import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import bootstrap_n, models_dir, random_state
from feature_engineering import build_feature_extractor
from model_utils import load_silver_validation_gold_frames


out_dir = models_dir / "test_tuning"

labels = ["afwijkend", "niet-afwijkend"]


def pct(n, tot):
    return f"{100 * n / tot:.1f}%" if tot else "n/a"


def full_metrics(name, y_true, y_pred, y_score=None, threshold=None):
    y_true = list(y_true)
    y_pred = list(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred) if len(set(y_true)) > 1 else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp, fn, fp, tn = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    n_afw, n_niet = tp + fn, tn + fp
    sens = tp / n_afw if n_afw else float("nan")
    spec = tn / n_niet if n_niet else float("nan")

    auc = float("nan")
    if y_score is not None and len(set(y_true)) == 2:
        try:
            auc = float(roc_auc_score((np.array(y_true) == "afwijkend").astype(int), y_score))
        except Exception:
            pass

    rng = np.random.default_rng(random_state)
    f1s = []
    yt, yp = np.array(y_true), np.array(y_pred)
    for _ in range(bootstrap_n):
        idx = rng.integers(0, len(yt), len(yt))
        f1s.append(f1_score(yt[idx], yp[idx], labels=labels, average="macro", zero_division=0))
    f1_lo, f1_hi = float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    result = {
        "model": name,
        "threshold": threshold,
        "n_test": len(y_true),
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "kappa": round(kappa, 4) if not np.isnan(kappa) else None,
        "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
        "sensitivity_afwijkend": round(sens, 4) if not np.isnan(sens) else None,
        "specificity_niet_afwijkend": round(spec, 4) if not np.isnan(spec) else None,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "macro_f1_ci_95": [round(f1_lo, 4), round(f1_hi, 4)],
    }
    return result, report


def print_result(r, report=None):
    thr = f"  threshold={r['threshold']:.2f}" if r["threshold"] is not None else ""
    tp, fn, fp, tn = r["tp"], r["fn"], r["fp"], r["tn"]
    n_afw, n_niet = tp + fn, tn + fp

    print()
    print(f"{r['model']}{thr}")
    print(
        f"  n={r['n_test']}, acc={r['accuracy']:.4f}, "
        f"macro_f1={r['macro_f1']:.4f} "
        f"[{r['macro_f1_ci_95'][0]:.3f}-{r['macro_f1_ci_95'][1]:.3f}]"
    )
    print(f"  kappa={r['kappa']}, auc={r['roc_auc']}")
    print(f"  afwijkend      : {tp}/{n_afw} ({pct(tp, n_afw)}) correct")
    print(f"  niet-afwijkend : {tn}/{n_niet} ({pct(tn, n_niet)}) correct")
    if report:
        print(report)


def tune_threshold(val_probs, val_true, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        preds = ["afwijkend" if p >= thr else "niet-afwijkend" for p in val_probs]
        score = f1_score(val_true, preds, labels=labels, average="macro", zero_division=0)
        if score > best_f1:
            best_f1, best_thr = score, float(thr)
    return best_thr, best_f1


def apply_threshold(gold_probs, gold_true, threshold, model_name, y_score=None):
    gold_pred = ["afwijkend" if p >= threshold else "niet-afwijkend" for p in gold_probs]
    return full_metrics(model_name, gold_true, gold_pred, y_score=y_score, threshold=threshold)


def run_threshold_tuning():
    print()
    print("threshold tuning")

    files = {
        "logreg": "tfidf_logreg_minwords0",
        "mlp": "tfidf_mlp_minwords0",
        "xgboost": "tfidf_xgboost_minwords0",
    }

    all_results = []
    for key, stem in files.items():
        val_f = models_dir / f"{stem}_validation_predictions.csv"
        gold_f = models_dir / f"{stem}_gold_test_predictions.csv"
        if not val_f.exists() or not gold_f.exists():
            print(f"  {key}: prediction files not found, skipping")
            continue

        val = pd.read_csv(val_f, encoding="utf-8-sig")
        gold = pd.read_csv(gold_f, encoding="utf-8-sig")

        val = val[val["y_true"].isin(labels)].copy()
        gold = gold[gold["y_true"].isin(labels)].copy()

        val_probs = val["prob_afwijkend"].values
        gold_probs = gold["prob_afwijkend"].values

        r_orig, rep_orig = full_metrics(
            f"tfidf {key} (original, thr=0.50)",
            gold["y_true"],
            gold["y_pred"].values,
            y_score=gold_probs,
            threshold=0.50,
        )
        print_result(r_orig, rep_orig)
        all_results.append(r_orig)

        best_thr, best_val_f1 = tune_threshold(val_probs, val["y_true"].values)
        old_val_f1 = f1_score(
            val["y_true"],
            val["y_pred"],
            labels=labels,
            average="macro",
            zero_division=0,
        )
        print(
            f"  optimal validation threshold: {best_thr:.2f} "
            f"(val macro-F1 {old_val_f1:.4f} -> {best_val_f1:.4f})"
        )

        r_tuned, rep_tuned = apply_threshold(
            gold_probs,
            gold["y_true"].values,
            best_thr,
            f"tfidf {key} (threshold-tuned, thr={best_thr:.2f})",
            y_score=gold_probs,
        )
        print_result(r_tuned, rep_tuned)
        all_results.append(r_tuned)

    return all_results


def load_data():
    from config import label_column, llm_labels_csv, manual_test_column, text_columns

    train_f, val_f, gold_f = load_silver_validation_gold_frames(
        llm_labels_csv,
        label_column=label_column,
        manual_test_column=manual_test_column,
        text_columns=text_columns,
        labels=labels,
    )

    x_tr = train_f["tekst_model"].astype(str).values
    y_tr = train_f["label_model"].astype(str).values
    x_va = val_f["tekst_model"].astype(str).values
    y_va = val_f["label_model"].astype(str).values
    x_go = gold_f["tekst_model"].astype(str).values
    y_go = gold_f["label_model"].astype(str).values

    print(f"  train: {len(y_tr)} | val: {len(y_va)} | gold: {len(y_go)}")
    counts = pd.Series(y_tr).value_counts()
    print(f"  train dist: {dict(counts)}")
    return x_tr, y_tr, x_va, y_va, x_go, y_go


def retrain_logreg(x_tr, y_tr, x_va, y_va, x_go, y_go):
    print()
    print("logreg balanced")

    best_f1, best_C, best_model = -1, 1.0, None
    for C in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        pipe = Pipeline(
            [
                ("features", build_feature_extractor(max_features=10000)),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        class_weight="balanced",
                        max_iter=1000,
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(x_tr, y_tr)

        val_pred = pipe.predict(x_va)
        score = f1_score(y_va, val_pred, labels=labels, average="macro", zero_division=0)
        print(f"    C={C:<6} val_f1={score:.4f}")
        if score > best_f1:
            best_f1, best_C, best_model = score, C, pipe

    print(f"  best C={best_C}, val_f1={best_f1:.4f}")
    gold_pred = best_model.predict(x_go)
    gold_proba = best_model.predict_proba(x_go)
    afw_idx = list(best_model.classes_).index("afwijkend")
    gold_score = gold_proba[:, afw_idx]

    r, rep = full_metrics(
        "logreg balanced (retrained)",
        y_go,
        gold_pred,
        y_score=gold_score,
        threshold=0.50,
    )
    print_result(r, rep)

    val_proba = best_model.predict_proba(x_va)[:, afw_idx]
    best_thr, _ = tune_threshold(val_proba, y_va)
    r_thr, rep_thr = apply_threshold(
        gold_score,
        y_go,
        best_thr,
        f"logreg balanced + threshold (thr={best_thr:.2f})",
        y_score=gold_score,
    )
    print_result(r_thr, rep_thr)
    return [r, r_thr], best_model


def retrain_mlp(x_tr, y_tr, x_va, y_va, x_go, y_go):
    print()
    print("mlp balanced")

    sw_tr = compute_sample_weight("balanced", y_tr)
    best_f1, best_params, best_model = -1, {}, None
    grid = list(
        ParameterGrid(
            {
                "features__tfidf__max_features": [10000, 20000],
                "clf__hidden_layer_sizes": [(128,), (256, 128), (128, 64)],
                "clf__alpha": [0.0001, 0.001],
            }
        )
    )

    for params in grid:
        pipe = Pipeline(
            [
                (
                    "features",
                    build_feature_extractor(
                        max_features=params["features__tfidf__max_features"]
                    ),
                ),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=params["clf__hidden_layer_sizes"],
                        alpha=params["clf__alpha"],
                        activation="relu",
                        batch_size=64,
                        learning_rate_init=0.001,
                        max_iter=500,
                        early_stopping=False,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(x_tr, y_tr, clf__sample_weight=sw_tr)

        val_pred = pipe.predict(x_va)
        score = f1_score(y_va, val_pred, labels=labels, average="macro", zero_division=0)
        p_str = (
            f"mf={params['features__tfidf__max_features']} "
            f"hl={params['clf__hidden_layer_sizes']} "
            f"a={params['clf__alpha']}"
        )
        print(f"    {p_str:<55} val_f1={score:.4f}")
        if score > best_f1:
            best_f1, best_params, best_model = score, params, pipe

    print(f"  best params={best_params}, val_f1={best_f1:.4f}")
    gold_pred = best_model.predict(x_go)
    gold_proba = best_model.predict_proba(x_go)
    afw_idx = list(best_model.classes_).index("afwijkend")
    gold_score = gold_proba[:, afw_idx]

    r, rep = full_metrics(
        "mlp sample_weight_balanced (retrained)",
        y_go,
        gold_pred,
        y_score=gold_score,
        threshold=0.50,
    )
    print_result(r, rep)

    val_proba = best_model.predict_proba(x_va)[:, afw_idx]
    best_thr, _ = tune_threshold(val_proba, y_va)
    r_thr, rep_thr = apply_threshold(
        gold_score,
        y_go,
        best_thr,
        f"mlp balanced + threshold (thr={best_thr:.2f})",
        y_score=gold_score,
    )
    print_result(r_thr, rep_thr)
    return [r, r_thr], best_model


def retrain_xgboost(x_tr, y_tr, x_va, y_va, x_go, y_go):
    print()
    print("xgboost scale_pos_weight search")

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  xgboost not installed, skipping")
        return [], None

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(labels)
    y_tr_enc = le.transform(y_tr)

    n_afw = int((y_tr == "afwijkend").sum())
    n_niet = int((y_tr == "niet-afwijkend").sum())
    spw_auto = n_afw / n_niet if n_niet > 0 else 1.0
    print(f"  train: {n_afw} afwijkend, {n_niet} niet-afwijkend, spw_auto={spw_auto:.2f}")

    best_f1, best_spw, best_model, best_feat = -1, 1.0, None, None
    grid = list(
        ParameterGrid(
            {
                "max_features": [10000, 30000],
                "scale_pos_weight": [
                    1.0,
                    spw_auto,
                    spw_auto * 0.5,
                    spw_auto * 1.5,
                    spw_auto * 2.0,
                ],
            }
        )
    )

    for params in grid:
        spw = round(params["scale_pos_weight"], 2)
        feat = build_feature_extractor(max_features=params["max_features"])
        x_tr_f = feat.fit_transform(x_tr, y_tr)
        x_va_f = feat.transform(x_va)

        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2.0,
            reg_lambda=3.0,
            scale_pos_weight=spw,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        )
        xgb.fit(x_tr_f, y_tr_enc)

        val_pred_enc = xgb.predict(x_va_f)
        val_pred = le.inverse_transform(val_pred_enc)
        score = f1_score(y_va, val_pred, labels=labels, average="macro", zero_division=0)
        print(f"    mf={params['max_features']} spw={spw:<6} val_f1={score:.4f}")
        if score > best_f1:
            best_f1, best_spw = score, spw
            best_model, best_feat = xgb, feat

    print(f"  best scale_pos_weight={best_spw}, val_f1={best_f1:.4f}")
    x_go_f = best_feat.transform(x_go)
    gold_pred_enc = best_model.predict(x_go_f)
    gold_pred = le.inverse_transform(gold_pred_enc)
    gold_proba = best_model.predict_proba(x_go_f)
    afw_idx = list(le.classes_).index("afwijkend")
    gold_score = gold_proba[:, afw_idx]

    r, rep = full_metrics(
        f"xgboost spw={best_spw} (retrained)",
        y_go,
        gold_pred,
        y_score=gold_score,
        threshold=0.50,
    )
    print_result(r, rep)

    x_va_f2 = best_feat.transform(x_va)
    val_proba = best_model.predict_proba(x_va_f2)[:, afw_idx]
    best_thr, _ = tune_threshold(val_proba, y_va)
    r_thr, rep_thr = apply_threshold(
        gold_score,
        y_go,
        best_thr,
        f"xgboost spw={best_spw} + threshold (thr={best_thr:.2f})",
        y_score=gold_score,
    )
    print_result(r_thr, rep_thr)
    return [r, r_thr], best_model


def print_summary(all_results):
    print()
    print("summary")
    print(
        f"  {'model':<52s} {'n':>4s} {'acc':>6s} {'f1':>6s} "
        f"{'f1-ci':>13s} {'kappa':>6s} {'afwijk.%':>9s} {'niet-afw.%':>10s}"
    )

    for r in all_results:
        tp, fn, fp, tn = r["tp"], r["fn"], r["fp"], r["tn"]
        n_afw, n_niet = tp + fn, tn + fp
        sens_s = pct(tp, n_afw)
        spec_s = pct(tn, n_niet)
        ci_s = f"[{r['macro_f1_ci_95'][0]:.3f},{r['macro_f1_ci_95'][1]:.3f}]"
        k_s = f"{r['kappa']:.3f}" if r["kappa"] is not None else "n/a"
        print(
            f"  {r['model']:<52s} {r['n_test']:>4d} {r['accuracy']:>6.3f} "
            f"{r['macro_f1']:>6.3f} {ci_s:>13s} {k_s:>6s} "
            f"{sens_s:>9s} {spec_s:>10s}"
        )


def save_results(all_results):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tuning_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nresults saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold-only", action="store_true")
    parser.add_argument("--skip-threshold", action="store_true")
    args = parser.parse_args()

    print("threshold tuning")

    results_threshold = [] if args.skip_threshold else run_threshold_tuning()
    if args.threshold_only:
        print_summary(results_threshold)
        save_results(results_threshold)
        return

    print()
    print("retraining")
    x_tr, y_tr, x_va, y_va, x_go, y_go = load_data()

    results_lr, _ = retrain_logreg(x_tr, y_tr, x_va, y_va, x_go, y_go)
    results_mlp, _ = retrain_mlp(x_tr, y_tr, x_va, y_va, x_go, y_go)
    results_xgb, _ = retrain_xgboost(x_tr, y_tr, x_va, y_va, x_go, y_go)

    all_results = results_threshold + results_lr + results_mlp + results_xgb
    print_summary(all_results)
    save_results(all_results)


if __name__ == "__main__":
    main()
