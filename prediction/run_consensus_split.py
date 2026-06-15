import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import final_dataset, models_dir, model_bert, random_state
from feature_engineering import build_feature_extractor

data_path = final_dataset
out_dir = models_dir / "test_met_consensus"
bert_model = model_bert

label_col = "label_gecombineerd_consensus"
text_cols = ["klinische_gegevens_huisarts", "vraagstelling_huisarts"]
labels = ["afwijkend", "niet-afwijkend"]
bootstrap_n = 1000


def has_text(v):
    return v is not None and not pd.isna(v) and str(v).strip() not in ("", "nan", "None")


def combine_text(row):
    return " ".join(str(row[c]).strip() for c in text_cols if c in row and has_text(row[c])).strip()


def normalize_label(v):
    if not has_text(v):
        return None
    v = str(v).strip().lower().replace("_", "-")
    return {"niet-afwijking": "niet-afwijkend", "normaal": "niet-afwijkend"}.get(v, v)


def pct(n, tot):
    return f"{100*n/tot:.1f}%" if tot else "n/a"


def bootstrap_metrics(y_true, y_pred, n=bootstrap_n, seed=2026):
    rng = np.random.default_rng(seed)
    accs, f1s, senss, specs = [], [], [], []
    yt = np.array(y_true)
    yp = np.array(y_pred)
    for _ in range(n):
        idx = rng.integers(0, len(yt), len(yt))
        t, p = yt[idx], yp[idx]
        accs.append(accuracy_score(t, p))
        f1s.append(f1_score(t, p, labels=labels, average="macro", zero_division=0))
        afw_m = t == "afwijkend"
        niet_m = t == "niet-afwijkend"
        senss.append(float((p[afw_m]  == "afwijkend").mean())      if afw_m.any()  else np.nan)
        specs.append(float((p[niet_m] == "niet-afwijkend").mean())  if niet_m.any() else np.nan)
    ci = lambda a: (float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5)))
    return {
        "accuracy_ci":    ci(accs),
        "macro_f1_ci":    ci(f1s),
        "sens_afwijkend_ci": ci(senss),
        "spec_niet_ci":   ci(specs),
    }


def evaluate(name, y_true, y_pred, y_score=None):
    """Compute and print all metrics; return dict."""
    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred) if len(set(y_true)) > 1 else float("nan")
    cm    = confusion_matrix(y_true, y_pred, labels=labels)
    tp, fn, fp, tn = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    n_afw  = tp + fn
    n_niet = tn + fp
    sens = tp / n_afw  if n_afw  > 0 else float("nan")
    spec = tn / n_niet if n_niet > 0 else float("nan")

    auc = float("nan")
    if y_score is not None and len(set(y_true)) == 2:
        try:
            auc = float(roc_auc_score((np.array(y_true) == "afwijkend").astype(int), y_score))
        except Exception:
            pass

    ci = bootstrap_metrics(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    print()
    print(name)
    print(f"  n_test={len(y_true)}, acc={acc:.4f}, macro_f1={f1:.4f}, kappa={kappa:.4f}, auc={auc:.4f}")
    print(f"  afwijkend correct    : {tp}/{n_afw} ({pct(tp, n_afw)})   "
          f"[95%-ci {ci['sens_afwijkend_ci'][0]:.3f}–{ci['sens_afwijkend_ci'][1]:.3f}]")
    print(f"  niet-afwijkend correct: {tn}/{n_niet} ({pct(tn, n_niet)})   "
          f"[95%-ci {ci['spec_niet_ci'][0]:.3f}–{ci['spec_niet_ci'][1]:.3f}]")
    print(f"  macro_f1 95%-ci : [{ci['macro_f1_ci'][0]:.3f}, {ci['macro_f1_ci'][1]:.3f}]")
    print(report)

    return {
        "model": name,
        "n_test": int(len(y_true)),
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "kappa": round(kappa, 4),
        "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
        "sensitivity_afwijkend": round(sens, 4) if not np.isnan(sens) else None,
        "specificity_niet_afwijkend": round(spec, 4) if not np.isnan(spec) else None,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy_ci_95":    [round(ci["accuracy_ci"][0], 4),    round(ci["accuracy_ci"][1], 4)],
        "macro_f1_ci_95":    [round(ci["macro_f1_ci"][0], 4),    round(ci["macro_f1_ci"][1], 4)],
        "sens_afwijkend_ci": [round(ci["sens_afwijkend_ci"][0], 4), round(ci["sens_afwijkend_ci"][1], 4)],
        "spec_niet_ci":      [round(ci["spec_niet_ci"][0], 4),   round(ci["spec_niet_ci"][1], 4)],
    }



def load_data(random_state):
    print("loading data...")
    df = pd.read_excel(data_path)
    df["_text"]  = df.apply(combine_text, axis=1)
    df["_label"] = df[label_col].map(normalize_label)

    
    mask = df["_label"].isin(labels) & df["_text"].map(has_text)
    df = df[mask].copy()
    print(f"  usable rows (binary consensus label + text): {len(df)}")

    X = df["_text"].astype(str).values
    y = df["_label"].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f"  train: {len(y_train)} (afwijkend: {(y_train=='afwijkend').sum()}, "
          f"niet-afwijkend: {(y_train=='niet-afwijkend').sum()})")
    print(f"  test : {len(y_test)}  (afwijkend: {(y_test=='afwijkend').sum()}, "
          f"niet-afwijkend: {(y_test=='niet-afwijkend').sum()})")
    return X_train, X_test, y_train, y_test



def run_classical(X_train, X_test, y_train, y_test, random_state):
    results = []

    feat = build_feature_extractor()

    
    print("\ntraining majority baseline...")
    maj = DummyClassifier(strategy="most_frequent", random_state=random_state)
    maj.fit(X_train, y_train)
    y_pred_maj = maj.predict(X_test)
    results.append(evaluate("majority baseline", y_test, y_pred_maj))

    
    print("\nfitting feature extractor...")
    X_train_f = feat.fit_transform(X_train, y_train)
    X_test_f  = feat.transform(X_test)

    
    print("training tfidf logistic regression...")
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                            random_state=random_state, solver="lbfgs")
    lr.fit(X_train_f, y_train)
    y_pred_lr = lr.predict(X_test_f)
    y_score_lr = lr.predict_proba(X_test_f)
    afw_idx = list(lr.classes_).index("afwijkend") if "afwijkend" in lr.classes_ else 0
    results.append(evaluate("tfidf log.regression", y_test, y_pred_lr,
                             y_score=y_score_lr[:, afw_idx]))

    
    print("training tfidf MLP...")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                        random_state=random_state, early_stopping=False)
    mlp.fit(X_train_f, y_train)
    y_pred_mlp = mlp.predict(X_test_f)
    y_score_mlp = mlp.predict_proba(X_test_f)
    afw_idx_mlp = list(mlp.classes_).index("afwijkend") if "afwijkend" in mlp.classes_ else 0
    results.append(evaluate("tfidf MLP", y_test, y_pred_mlp,
                             y_score=y_score_mlp[:, afw_idx_mlp]))

    
    try:
        from xgboost import XGBClassifier
        print("training tfidf XGBoost...")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_enc = le.fit_transform(y_train)
        xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                             use_label_encoder=False, eval_metric="logloss",
                             random_state=random_state, verbosity=0)
        xgb.fit(X_train_f, y_enc)
        y_pred_enc = xgb.predict(X_test_f)
        y_pred_xgb = le.inverse_transform(y_pred_enc)
        y_score_xgb = xgb.predict_proba(X_test_f)
        afw_idx_xgb = list(le.classes_).index("afwijkend") if "afwijkend" in le.classes_ else 0
        results.append(evaluate("tfidf XGBoost", y_test, y_pred_xgb,
                                 y_score=y_score_xgb[:, afw_idx_xgb]))
    except ImportError:
        print("xgboost not installed, skipping")

    return results



def run_bert(X_train, X_test, y_train, y_test, random_state,
             epochs=3, batch_size=8, max_length=256):
    import torch
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, Trainer, TrainingArguments)

    print()
    print("BERT fine-tuning on 80/20 consensus split")

    out_bert = out_dir / "bert_consensus"
    out_bert.mkdir(parents=True, exist_ok=True)

    enc = LabelEncoder()
    enc.fit(labels)
    y_train_enc = enc.transform(y_train)
    y_test_enc  = enc.transform(y_test)

    tokenizer = AutoTokenizer.from_pretrained(bert_model)

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(list(texts), truncation=True,
                                 padding="max_length", max_length=max_length,
                                 return_tensors="pt")
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.enc.items()} | {"labels": self.labels[idx]}

    
    from sklearn.model_selection import train_test_split as tts
    X_tr, X_val, y_tr, y_val = tts(
        X_train, y_train_enc, test_size=0.10,
        random_state=random_state, stratify=y_train_enc
    )
    train_ds = TextDataset(X_tr,    y_tr)
    val_ds   = TextDataset(X_val,   y_val)
    test_ds  = TextDataset(X_test,  y_test_enc)

    model = AutoModelForSequenceClassification.from_pretrained(
        bert_model, num_labels=len(labels)
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        }

    training_kwargs = dict(
        output_dir=str(out_bert / "checkpoints"),
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=random_state,
        report_to=[],
        save_only_model=True,
        save_total_limit=1,
    )
    try:
        args = TrainingArguments(eval_strategy="epoch", **training_kwargs)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="epoch", **training_kwargs)

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    pred_output = trainer.predict(test_ds)
    y_pred_enc  = np.argmax(pred_output.predictions, axis=1)
    y_pred = enc.inverse_transform(y_pred_enc)

    
    logits = pred_output.predictions
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    afw_idx = list(enc.classes_).index("afwijkend") if "afwijkend" in enc.classes_ else 0

    result = evaluate("bert MedRoBERTa.nl (consensus 80/20)",
                      y_test, y_pred, y_score=probs[:, afw_idx])

    trainer.save_model(str(out_bert / "best_model"))
    tokenizer.save_pretrained(str(out_bert / "best_model"))
    return result



def main():
    parser = argparse.ArgumentParser(
        description="Internal validation: train+test on LLM consensus labels (80/20 split)"
    )
    parser.add_argument("--bert",   action="store_true", help="also fine-tune MedRoBERTa.nl")
    parser.add_argument("--seed",   type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    print("consensus split experiment (80/20)")
    print("  label  : label_gecombineerd_consensus")
    print("  text   : GP referral text (huisarts kolommen)")
    print(f"  seed   : {args.seed}")

    X_train, X_test, y_train, y_test = load_data(args.seed)

    results = run_classical(X_train, X_test, y_train, y_test, args.seed)

    if args.bert:
        bert_result = run_bert(X_train, X_test, y_train, y_test, args.seed,
                               epochs=args.epochs, batch_size=args.batch_size)
        results.append(bert_result)

    
    summary_path = out_dir / "results_consensus_split.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nresults saved: {summary_path}")

    
    print()
    print("summary - consensus 80/20 split")
    print(f"  {'model':<30s}  {'n':>4s}  {'acc':>6s}  {'f1-macro':>8s}  "
          f"{'kappa':>6s}  {'afwijkend_correct':>18s}  {'niet-afw_correct':>18s}")
    for r in results:
        sens = r.get("sensitivity_afwijkend")
        spec = r.get("specificity_niet_afwijkend")
        n_afw  = r["tp"] + r["fn"]
        n_niet = r["tn"] + r["fp"]
        tp, tn = r["tp"], r["tn"]
        sens_s = f"{tp}/{n_afw} ({100*sens:.1f}%)" if sens is not None else "n/a"
        spec_s = f"{tn}/{n_niet} ({100*spec:.1f}%)" if spec is not None else "n/a"
        print(f"  {r['model']:<30s}  {r['n_test']:>4d}  {r['accuracy']:>6.3f}  "
              f"{r['macro_f1']:>8.3f}  {r['kappa']:>6.3f}  {sens_s:>18s}  {spec_s:>18s}")


if __name__ == "__main__":
    main()
