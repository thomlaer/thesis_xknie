import argparse
import json
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import final_dataset, models_dir, model_bert, random_state, label_llm, label_gold, outcome_labels

bootstrap_n = 1000

bert_models = {
    "gp_only": {
        "name": "gp_only",
        "columns": ["klinische_gegevens_huisarts", "vraagstelling_huisarts"],
        "description": "GP referral text only (silver train only)",
    },
    "gp_plus_radiology": {
        "name": "gp_plus_radiology",
        "columns": ["klinische_gegevens_huisarts", "vraagstelling_huisarts", "bevindingen", "conclusie"],
        "description": "GP text + radiology combined",
    },
}


def has_text(v):
    return v is not None and not pd.isna(v) and str(v).strip() not in ("", "nan", "None")


def combine_columns(row, columns):
    return " ".join(str(row[c]).strip() for c in columns if c in row and has_text(row[c])).strip()


def normalize_label(v):
    if not has_text(v):
        return None
    v = str(v).strip().lower().replace("_", "-")
    return {"niet-afwijking": "niet-afwijkend", "normaal": "niet-afwijkend"}.get(v, v)


def bootstrap_ci(y_true, y_pred, n=bootstrap_n):
    rng = np.random.default_rng(random_state)
    accs, f1s = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, yp = y_true[idx], y_pred[idx]
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, labels=outcome_labels, average="macro", zero_division=0))
    return {
        "accuracy_ci": (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))),
        "macro_f1_ci": (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))),
    }


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()} | {"labels": self.labels[idx]}


def run_model(model_key, epochs=3, batch_size=8, max_length=256):
    cfg = bert_models[model_key]
    out_dir = models_dir / "bert" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(final_dataset)
    df["_text"]   = df.apply(lambda r: combine_columns(r, cfg["columns"]), axis=1)
    df["_silver"] = df[label_llm].map(normalize_label) if label_llm in df.columns else None
    df["_gold"]   = df[label_gold].map(normalize_label) if label_gold in df.columns else None

    gold_mask   = df["_gold"].isin(outcome_labels) & df["_text"].map(has_text)
    silver_mask = (~gold_mask) & df["_silver"].isin(outcome_labels) & df["_text"].map(has_text)

    gold_df   = df[gold_mask].copy()
    silver_df = df[silver_mask].copy()

    train_pool = silver_df.copy()
    train_pool["_label_for_train"] = train_pool["_silver"]

    train_df, val_df = train_test_split(
        train_pool, test_size=0.15, random_state=random_state, stratify=train_pool["_label_for_train"],
    )

    print(f"model {model_key} ({cfg['description']})")
    print(f"  train: {len(train_df)}, val: {len(val_df)}, gold test: {len(gold_df)}")

    enc = LabelEncoder()
    enc.fit(outcome_labels)

    y_train = enc.transform(train_df["_label_for_train"].tolist())
    y_val   = enc.transform(val_df["_label_for_train"].tolist())
    y_gold  = enc.transform(gold_df["_gold"].tolist())

    tokenizer = AutoTokenizer.from_pretrained(model_bert)
    train_ds  = TextDataset(train_df["_text"], y_train, tokenizer, max_length)
    val_ds    = TextDataset(val_df["_text"],   y_val,   tokenizer, max_length)
    gold_ds   = TextDataset(gold_df["_text"],  y_gold,  tokenizer, max_length)

    model = AutoModelForSequenceClassification.from_pretrained(model_bert, num_labels=len(outcome_labels))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        }

    training_kwargs = dict(
        output_dir=str(out_dir / "checkpoints"),
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

    pred_output  = trainer.predict(gold_ds)
    y_pred_enc   = np.argmax(pred_output.predictions, axis=1)
    y_pred       = enc.inverse_transform(y_pred_enc)
    y_true       = enc.inverse_transform(y_gold)

    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, labels=outcome_labels, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    probs = softmax(pred_output.predictions)
    afw_idx = list(enc.classes_).index("afwijkend") if "afwijkend" in enc.classes_ else 0
    auc   = float(roc_auc_score((np.array(y_true) == "afwijkend").astype(int), probs[:, afw_idx]))
    ci    = bootstrap_ci(np.array(y_true), np.array(y_pred))
    report = classification_report(y_true, y_pred, labels=outcome_labels, zero_division=0)

    
    sens_afw  = float(recall_score(y_true, y_pred, labels=["afwijkend"],       average="macro", zero_division=0))
    spec_niet = float(recall_score(y_true, y_pred, labels=["niet-afwijkend"],  average="macro", zero_division=0))

    metrics = {
        "model": model_key,
        "description": cfg["description"],
        "n_gold": int(len(y_true)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "kappa": round(kappa, 4),
        "roc_auc": round(auc, 4),
        "sensitivity_afwijkend": round(sens_afw, 4),
        "specificity_niet_afwijkend": round(spec_niet, 4),
        "accuracy_ci_95": [round(ci["accuracy_ci"][0], 4), round(ci["accuracy_ci"][1], 4)],
        "macro_f1_ci_95": [round(ci["macro_f1_ci"][0], 4), round(ci["macro_f1_ci"][1], 4)],
    }

    print(f"  accuracy: {acc:.4f}  f1: {f1:.4f}  kappa: {kappa:.4f}  auc: {auc:.4f}")
    print(report)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    result_df = gold_df[["case_id"]].copy() if "case_id" in gold_df.columns else pd.DataFrame()
    result_df["y_true"] = y_true
    result_df["y_pred"] = y_pred
    result_df["prob_afwijkend"] = probs[:, afw_idx]
    result_df.to_csv(out_dir / "gold_predictions.csv", index=False, encoding="utf-8-sig")

    trainer.save_model(str(out_dir / "best_model"))
    tokenizer.save_pretrained(str(out_dir / "best_model"))

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(bert_models), choices=list(bert_models))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    all_results = {}
    for model_key in args.models:
        all_results[model_key] = run_model(model_key, args.epochs, args.batch_size, args.max_length)

    summary_path = models_dir / "bert" / "model_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print("\nmodel summary:")
    print(f"{'model':<20} {'description':<35} {'acc':>6} {'f1':>6} {'kappa':>6} {'auc':>6}")
    for s, r in all_results.items():
        print(f"{s:<20} {r['description']:<35} {r['accuracy']:>6.3f} {r['macro_f1']:>6.3f} {r['kappa']:>6.3f} {r['roc_auc']:>6.3f}")


if __name__ == "__main__":
    main()
