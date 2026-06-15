import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


text_columns = config.text_columns
label_column = config.label_column
manual_test_column = config.manual_test_column
labels = config.labels
random_state = config.random_state
validation_size = getattr(config, "validation_size", 0.15)
bootstrap_n = getattr(config, "bootstrap_n", 1000)


def read_table(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def has_text(value):
    return value is not None and not pd.isna(value) and str(value).strip() not in ("", "nan", "None", "<NA>")


def combine_text(row, text_columns=text_columns):
    parts = []
    for col in text_columns:
        if col in row and has_text(row[col]):
            parts.append(str(row[col]).strip())
    return " ".join(parts).strip()


def count_words(text):
    if not has_text(text):
        return 0
    return len(re.findall(r"\w+", str(text).lower()))


def normalize_label(value):
    if not has_text(value):
        return None
    value = str(value).strip().lower()
    value = value.replace("_", "-")
    value = re.sub(r"\s+", "-", value)
    replacements = {
        "niet-afwijking": "niet-afwijkend",
        "niet-afwijkend.": "niet-afwijkend",
        "afwijkend.": "afwijkend",
        "onbekend.": "onbekend",
        "yes": "ja",
        "no": "nee",
        "unknown": "onbekend",
        "normaal": "niet-afwijkend",
        "normal": "niet-afwijkend",
        "a": "afwijkend",
        "n": "niet-afwijkend",
    }
    return replacements.get(value, value)


def make_model_frame(df, label_column=label_column, text_columns=text_columns, labels=labels, min_words=0):
    if "tekst" in df.columns and "label" in df.columns and label_column not in df.columns:
        work = df.rename(columns={"tekst": "tekst_model", "label": "label_model"}).copy()
    else:
        if label_column not in df.columns:
            raise ValueError(f"Missing column: {label_column}")
        work = df.copy()
        work["tekst_model"] = work.apply(lambda r: combine_text(r, text_columns), axis=1)
        work["label_model"] = work[label_column]

    work["label_model"] = work["label_model"].map(normalize_label)
    work["word_count"] = work["tekst_model"].map(count_words)
    work = work[work["tekst_model"].map(has_text) & work["label_model"].map(has_text)].copy()
    work = work[work["label_model"].isin(labels)].copy()
    if min_words and min_words > 0:
        work = work[work["word_count"] >= min_words].copy()
    return work


def load_training_data(path, label_column=label_column, text_columns=text_columns, labels=labels, min_words=0):
    df = read_table(path)
    work = make_model_frame(df, label_column=label_column, text_columns=text_columns, labels=labels, min_words=min_words)
    if len(work) == 0:
        raise ValueError("No usable training rows found.")
    return work["tekst_model"].astype(str), work["label_model"].astype(str)


def split_data(x, y, test_size=0.2):
    from sklearn.model_selection import train_test_split

    counts = y.value_counts()
    stratify = y if counts.min() >= 2 and len(counts) > 1 else None
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=stratify)


def split_frame(frame, test_size=validation_size):
    from sklearn.model_selection import train_test_split

    counts = frame["label_model"].value_counts()
    stratify = frame["label_model"] if counts.min() >= 2 and len(counts) > 1 else None
    train_frame, val_frame = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_frame.copy(), val_frame.copy()


def make_cv(y, max_splits=3):
    from sklearn.model_selection import StratifiedKFold

    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return StratifiedKFold(n_splits=min(max_splits, int(counts.min())), shuffle=True, random_state=random_state)


def combine_sets(x_first, x_second, y_first, y_second):
    x = pd.concat([pd.Series(x_first), pd.Series(x_second)], ignore_index=True).astype(str)
    y = pd.concat([pd.Series(y_first), pd.Series(y_second)], ignore_index=True).astype(str)
    return x, y


def combine_frames(first, second):
    return pd.concat([first, second], ignore_index=True).copy()


def load_silver_validation_gold_frames(
    train_path,
    label_column=label_column,
    test_path=None,
    test_label_column=None,
    manual_test_column=manual_test_column,
    text_columns=text_columns,
    labels=labels,
    validation_size=validation_size,
    min_words=0,
):
    if test_path:
        silver_source = read_table(train_path)
        gold_source = read_table(test_path)
        silver_frame = make_model_frame(
            silver_source,
            label_column=label_column,
            text_columns=text_columns,
            labels=labels,
            min_words=min_words,
        )
        gold_frame = make_model_frame(
            gold_source,
            label_column=test_label_column or manual_test_column,
            text_columns=text_columns,
            labels=labels,
            min_words=min_words,
        )
    else:
        df = read_table(train_path)
        if not manual_test_column or manual_test_column not in df.columns:
            raise ValueError(f"{manual_test_column} not found")

        manual_labels = df[manual_test_column].map(normalize_label)
        manual_mask = manual_labels.isin(labels)
        if not manual_mask.any():
            raise ValueError(f"{manual_test_column} not found")

        silver_frame = make_model_frame(
            df[~manual_mask].copy(),
            label_column=label_column,
            text_columns=text_columns,
            labels=labels,
            min_words=min_words,
        )
        gold_frame = make_model_frame(
            df[manual_mask].copy(),
            label_column=manual_test_column,
            text_columns=text_columns,
            labels=labels,
            min_words=min_words,
        )

    if len(silver_frame) == 0:
        raise ValueError("No usable silver-labelled training rows found.")
    if len(gold_frame) == 0:
        raise ValueError("No usable gold test rows found.")
    if len(silver_frame) < 2:
        raise ValueError("Not enough silver-labelled rows for training.")

    train_frame, val_frame = split_frame(silver_frame, test_size=validation_size)
    print(
        f"silver/gold split: {len(train_frame)} train, {len(val_frame)} val, {len(gold_frame)} gold test",
        flush=True,
    )
    return train_frame, val_frame, gold_frame


def load_silver_validation_gold_data(
    train_path,
    label_column=label_column,
    test_path=None,
    test_label_column=None,
    manual_test_column=manual_test_column,
    text_columns=text_columns,
    labels=labels,
    validation_size=validation_size,
    min_words=0,
):
    train_frame, val_frame, gold_frame = load_silver_validation_gold_frames(
        train_path,
        label_column=label_column,
        test_path=test_path,
        test_label_column=test_label_column,
        manual_test_column=manual_test_column,
        text_columns=text_columns,
        labels=labels,
        validation_size=validation_size,
        min_words=min_words,
    )
    return (
        train_frame["tekst_model"].astype(str),
        val_frame["tekst_model"].astype(str),
        gold_frame["tekst_model"].astype(str),
        train_frame["label_model"].astype(str),
        val_frame["label_model"].astype(str),
        gold_frame["label_model"].astype(str),
    )


def load_train_test_data(
    train_path,
    label_column=label_column,
    test_path=None,
    test_label_column=None,
    manual_test_column=manual_test_column,
    text_columns=text_columns,
    labels=labels,
    test_size=validation_size,
    min_words=0,
):
    x_train, x_val, x_gold, y_train, y_val, y_gold = load_silver_validation_gold_data(
        train_path,
        label_column=label_column,
        test_path=test_path,
        test_label_column=test_label_column,
        manual_test_column=manual_test_column,
        text_columns=text_columns,
        labels=labels,
        validation_size=test_size,
        min_words=min_words,
    )
    x_silver, y_silver = combine_sets(x_train, x_val, y_train, y_val)
    print(f"silver: {len(y_silver)} ({label_column}), gold test: {len(y_gold)} ({manual_test_column})", flush=True)
    return x_silver, x_gold, y_silver, y_gold


def metric_value(y_true, y_pred, metric, labels):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_true = pd.Series(y_true).reset_index(drop=True).astype(str)
    y_pred = pd.Series(y_pred).reset_index(drop=True).astype(str)
    if len(y_true) == 0:
        return math.nan

    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    if metric == "sensitivity_afwijkend":
        if "afwijkend" not in set(y_true):
            return math.nan
        return float(recall_score(y_true, y_pred, labels=["afwijkend"], average="macro", zero_division=0))
    if metric == "precision_afwijkend":
        if "afwijkend" not in set(y_pred) and "afwijkend" not in set(y_true):
            return math.nan
        return float(precision_score(y_true, y_pred, labels=["afwijkend"], average="macro", zero_division=0))
    if metric == "specificity_non_abnormal":
        mask = y_true == "niet-afwijkend"
        if not mask.any():
            return math.nan
        return float((y_pred[mask] != "afwijkend").mean())
    raise ValueError(f"Unknown metric: {metric}")


def confidence_interval(values):
    values = np.asarray([value for value in values if not pd.isna(value)], dtype=float)
    if len(values) == 0:
        return {"lower": None, "upper": None}
    return {
        "lower": float(np.percentile(values, 2.5)),
        "upper": float(np.percentile(values, 97.5)),
    }


def bootstrap_metrics(y_true, y_pred, labels, n_bootstrap=bootstrap_n):
    y_true = pd.Series(y_true).reset_index(drop=True).astype(str)
    y_pred = pd.Series(y_pred).reset_index(drop=True).astype(str)
    metrics = ["accuracy", "macro_f1", "sensitivity_afwijkend", "specificity_non_abnormal", "precision_afwijkend"]
    result = {}
    for metric in metrics:
        point = metric_value(y_true, y_pred, metric, labels)
        result[metric] = {"value": None if pd.isna(point) else float(point)}

    if len(y_true) < 2 or n_bootstrap <= 0:
        for metric in metrics:
            result[metric]["ci_95"] = {"lower": None, "upper": None}
        return result

    rng = np.random.default_rng(random_state)
    boot_values = {metric: [] for metric in metrics}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        sample_true = y_true.iloc[indices].reset_index(drop=True)
        sample_pred = y_pred.iloc[indices].reset_index(drop=True)
        for metric in metrics:
            boot_values[metric].append(metric_value(sample_true, sample_pred, metric, labels))

    for metric in metrics:
        result[metric]["ci_95"] = confidence_interval(boot_values[metric])
    return result


def save_label_distribution(y, output_dir, name, labels=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = pd.Series(y).astype(str).value_counts(dropna=False).rename_axis("label").reset_index(name="count")
    counts["percentage"] = counts["count"] / counts["count"].sum()
    if labels:
        base = pd.DataFrame({"label": labels})
        counts = base.merge(counts, on="label", how="left").fillna({"count": 0, "percentage": 0})
    counts.to_csv(output_dir / f"{name}_label_distribution.csv", index=False, encoding="utf-8-sig")
    return counts


def save_split_distributions(splits, output_dir, name):
    rows = []
    for split_name, y in splits.items():
        distribution = pd.Series(y).astype(str).value_counts(dropna=False)
        total = int(distribution.sum())
        for label, count in distribution.items():
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "count": int(count),
                    "percentage": float(count / total) if total else 0.0,
                }
            )
    result = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / f"{name}_split_label_distribution.csv", index=False, encoding="utf-8-sig")
    return result


def save_metrics(y_true, y_pred, output_dir, name, labels=None, bootstrap_n=bootstrap_n, y_score=None):
    from sklearn.metrics import (
        average_precision_score,
        classification_report,
        confusion_matrix,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = labels or sorted(set(y_true) | set(y_pred))

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    matrix = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    metrics = bootstrap_metrics(y_true, y_pred, labels, n_bootstrap=bootstrap_n)
    metrics["labels"] = [str(label) for label in labels]
    metrics["n"] = int(len(y_true))

    if y_score is not None and len(labels) == 2 and "afwijkend" in labels:
        y_true_binary = pd.Series(y_true).astype(str).eq("afwijkend").astype(int)
        scores = np.asarray(y_score, dtype=float)
        if len(set(y_true_binary)) == 2:
            roc_auc = float(roc_auc_score(y_true_binary, scores))
            pr_auc = float(average_precision_score(y_true_binary, scores))
            metrics["roc_auc"] = {"value": roc_auc, "ci_95": {"lower": None, "upper": None}}
            metrics["pr_auc"] = {"value": pr_auc, "ci_95": {"lower": None, "upper": None}}

            fpr, tpr, thresholds = roc_curve(y_true_binary, scores)
            pd.DataFrame(
                {
                    "false_positive_rate": fpr,
                    "true_positive_rate": tpr,
                    "threshold": thresholds,
                }
            ).to_csv(output_dir / f"{name}_roc_curve.csv", index=False, encoding="utf-8-sig")

            precision, recall, thresholds = precision_recall_curve(y_true_binary, scores)
            threshold_values = list(thresholds) + [None]
            pd.DataFrame(
                {
                    "precision": precision,
                    "recall": recall,
                    "threshold": threshold_values,
                }
            ).to_csv(output_dir / f"{name}_precision_recall_curve.csv", index=False, encoding="utf-8-sig")

    (output_dir / f"{name}_classification_report.txt").write_text(report, encoding="utf-8")
    matrix.to_csv(output_dir / f"{name}_confusion_matrix.csv", encoding="utf-8-sig")
    (output_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "metric": metric,
                "value": values.get("value"),
                "ci_95_lower": values.get("ci_95", {}).get("lower"),
                "ci_95_upper": values.get("ci_95", {}).get("upper"),
            }
            for metric, values in metrics.items()
            if isinstance(values, dict)
        ]
    ).to_csv(output_dir / f"{name}_metrics_ci.csv", index=False, encoding="utf-8-sig")

    print(report)
    print(matrix)
    print(metrics)
    return metrics


def probability_for_label(probabilities, classes, label):
    if probabilities is None or classes is None or label not in list(classes):
        return None
    class_index = list(classes).index(label)
    return np.asarray(probabilities)[:, class_index]


def predict_probabilities(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)
    return None


def save_prediction_output(frame, y_pred, output_dir, name, classes=None, probabilities=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame()
    for column in ["case_id", "klinische_gegevens_huisarts", "vraagstelling_huisarts", "word_count"]:
        if column in frame.columns:
            result[column] = frame[column].values

    result["y_true"] = frame["label_model"].astype(str).values
    result["y_pred"] = pd.Series(y_pred).astype(str).values

    if probabilities is not None and classes is not None:
        for label in ["afwijkend", "niet-afwijkend"]:
            probs = probability_for_label(probabilities, classes, label)
            if probs is not None:
                result[f"prob_{label}"] = probs

    result["correct"] = result["y_true"] == result["y_pred"]
    result.to_csv(output_dir / f"{name}_predictions.csv", index=False, encoding="utf-8-sig")
    return result


def save_model_split_outputs(model, train_frame, val_frame, gold_frame, output_dir, name, labels):
    for split_name, frame, bootstrap_n in [
        ("train", train_frame, 0),
        ("validation", val_frame, 0),
        ("gold_test", gold_frame, bootstrap_n),
    ]:
        x = frame["tekst_model"].astype(str)
        y = frame["label_model"].astype(str)
        pred = model.predict(x)
        probabilities = predict_probabilities(model, x)
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            classes = getattr(model.named_steps.get("clf"), "classes_", None)

        y_score = probability_for_label(probabilities, classes, "afwijkend")
        save_metrics(
            y,
            pred,
            output_dir,
            f"{name}_{split_name}",
            labels,
            bootstrap_n=bootstrap_n,
            y_score=y_score,
        )
        save_prediction_output(frame, pred, output_dir, f"{name}_{split_name}", classes, probabilities)

    save_prediction_output(
        gold_frame,
        model.predict(gold_frame["tekst_model"].astype(str)),
        output_dir,
        name,
        getattr(model.named_steps.get("clf"), "classes_", None) if hasattr(model, "named_steps") else getattr(model, "classes_", None),
        predict_probabilities(model, gold_frame["tekst_model"].astype(str)),
    )


def save_json(data, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
