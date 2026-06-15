import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import llm_labels_csv, output, label_column, manual_test_column, text_columns, labels, random_state
from feature_engineering import build_feature_extractor
from model_utils import (
    load_silver_validation_gold_frames,
    probability_for_label,
    save_json,
    save_metrics,
    save_prediction_output,
    save_split_distributions,
)

outcome_labels = [label for label in labels if label != "onbekend"]

n_estimators = 2000
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
reg_alpha = 2.0
reg_lambda = 5.0
early_stop = 100


def grid_for_mode(mode):
    if mode == "none":
        return [
            {
                "max_features": 30000,
                "ngram_range": (1, 2),
                "min_df": 2,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
            }
        ]
    if mode == "quick":
        return list(
            ParameterGrid(
                {
                    "max_features": [10000, 30000],
                    "ngram_range": [(1, 1), (1, 2)],
                    "min_df": [2],
                    "max_depth": [3, 5],
                    "learning_rate": [0.03, 0.05],
                    "subsample": [0.8],
                    "colsample_bytree": [0.8],
                    "reg_alpha": [1.0, 2.0],
                    "reg_lambda": [3.0, 5.0],
                }
            )
        )
    return list(
        ParameterGrid(
            {
                "max_features": [10000, 30000, 50000],
                "ngram_range": [(1, 1), (1, 2)],
                "min_df": [1, 2, 3],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.02, 0.05, 0.08],
                "subsample": [0.7, 0.85],
                "colsample_bytree": [0.7, 0.85],
                "reg_alpha": [0.0, 1.0, 2.0],
                "reg_lambda": [1.0, 3.0, 5.0],
            }
        )
    )


def build_model(xgb, params, class_count, device):
    settings = {
        "n_estimators": n_estimators,
        "max_depth": params["max_depth"],
        "learning_rate": params["learning_rate"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "reg_alpha": params["reg_alpha"],
        "reg_lambda": params["reg_lambda"],
        "tree_method": "hist",
        "device": device,
        "random_state": random_state,
        "n_jobs": -1,
        "early_stopping_rounds": early_stop,
    }
    if class_count == 2:
        settings["objective"] = "binary:logistic"
        settings["eval_metric"] = "logloss"
    else:
        settings["objective"] = "multi:softprob"
        settings["num_class"] = class_count
        settings["eval_metric"] = "mlogloss"
    return xgb.XGBClassifier(**settings)


def predict_labels(model, x_matrix, label_encoder):
    raw = np.asarray(model.predict(x_matrix))
    if raw.ndim == 2:
        raw = raw.argmax(axis=1)
    raw = raw.astype(int)
    return label_encoder.inverse_transform(raw)


def save_xgboost_split_outputs(model, features, label_encoder, train_frame, val_frame, gold_frame, output_dir, name):
    classes = list(label_encoder.classes_)
    for split_name, frame, bootstrap_n in [
        ("train", train_frame, 0),
        ("validation", val_frame, 0),
        ("gold_test", gold_frame, 1000),
    ]:
        x_vec = features.transform(frame["tekst_model"].astype(str))
        pred = predict_labels(model, x_vec, label_encoder)
        probabilities = model.predict_proba(x_vec) if hasattr(model, "predict_proba") else None
        y_score = probability_for_label(probabilities, classes, "afwijkend")
        save_metrics(
            frame["label_model"].astype(str),
            pred,
            output_dir,
            f"{name}_{split_name}",
            classes,
            bootstrap_n=bootstrap_n,
            y_score=y_score,
        )
        save_prediction_output(frame, pred, output_dir, f"{name}_{split_name}", classes, probabilities)

    x_gold_vec = features.transform(gold_frame["tekst_model"].astype(str))
    gold_pred = predict_labels(model, x_gold_vec, label_encoder)
    gold_prob = model.predict_proba(x_gold_vec) if hasattr(model, "predict_proba") else None
    save_prediction_output(gold_frame, gold_pred, output_dir, name, classes, gold_prob)


def train(
    input_path=llm_labels_csv,
    output_dir=output / "models",
    label_column=label_column,
    device="cpu",
    test_input=None,
    test_label_column=None,
    manual_test_column=manual_test_column,
    grid="quick",
    use_manual_features=True,
    min_words=0,
):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise SystemExit("Install xgboost first: pip install xgboost") from exc

    train_frame, val_frame, gold_frame = load_silver_validation_gold_frames(
        input_path,
        label_column=label_column,
        test_path=test_input,
        test_label_column=test_label_column,
        manual_test_column=manual_test_column,
        text_columns=text_columns,
        labels=outcome_labels,
        min_words=min_words,
    )
    x_train = train_frame["tekst_model"].astype(str)
    y_train = train_frame["label_model"].astype(str)
    x_val = val_frame["tekst_model"].astype(str)
    y_val = val_frame["label_model"].astype(str)
    suffix = f"_minwords{min_words}"
    model_name = f"tfidf_xgboost{suffix}"

    save_split_distributions(
        {"silver_train": y_train, "silver_validation": y_val, "gold_test": gold_frame["label_model"]},
        output_dir,
        model_name,
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(outcome_labels)
    y_train_enc = label_encoder.transform(y_train)
    y_val_enc = label_encoder.transform(y_val)

    best = None
    grid_results = []
    for run_number, params in enumerate(grid_for_mode(grid), start=1):
        print(f"XGBoost grid run {run_number}: {params}", flush=True)
        features = build_feature_extractor(
            max_features=params["max_features"],
            ngram_range=params["ngram_range"],
            min_df=params["min_df"],
            use_manual_features=use_manual_features,
        )
        x_train_vec = features.fit_transform(x_train)
        x_val_vec = features.transform(x_val)

        model = build_model(xgb, params, len(label_encoder.classes_), device)
        model.fit(
            x_train_vec,
            y_train_enc,
            eval_set=[(x_train_vec, y_train_enc), (x_val_vec, y_val_enc)],
            verbose=100,
        )
        val_pred = predict_labels(model, x_val_vec, label_encoder)
        score = f1_score(y_val, val_pred, labels=list(label_encoder.classes_), average="macro", zero_division=0)
        grid_results.append({"params": params, "validation_f1_macro": score})
        print(f"validation f1_macro: {score:.4f}", flush=True)

        if best is None or score > best["score"]:
            best = {"score": score, "params": params, "features": features, "model": model}

    x_val_vec = best["features"].transform(x_val)
    val_pred = predict_labels(best["model"], x_val_vec, label_encoder)
    save_metrics(y_val, val_pred, output_dir, f"{model_name}_validation_model_selection", list(label_encoder.classes_), bootstrap_n=0)
    save_xgboost_split_outputs(best["model"], best["features"], label_encoder, train_frame, val_frame, gold_frame, output_dir, model_name)

    save_json(
        {
            "best_params": best["params"],
            "best_validation_f1_macro": best["score"],
            "grid_results": grid_results,
        },
        output_dir,
        f"{model_name}_grid",
    )
    dump(
        {
            "model": best["model"],
            "features": best["features"],
            "label_encoder": label_encoder,
        },
        Path(output_dir) / f"{model_name}.joblib",
    )
    if min_words == 0:
        dump(
            {
                "model": best["model"],
                "features": best["features"],
                "label_encoder": label_encoder,
            },
            Path(output_dir) / "tfidf_xgboost.joblib",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(llm_labels_csv))
    parser.add_argument("--output", default=str(output / "models"))
    parser.add_argument("--label-column", default=label_column)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--test-input", default=None)
    parser.add_argument("--test-label-column", default=None)
    parser.add_argument("--manual-test-column", default=manual_test_column)
    parser.add_argument("--grid", default="quick", choices=["none", "quick", "full"])
    parser.add_argument("--min-words", type=int, default=0)
    parser.add_argument("--no-manual-features", action="store_true")
    args = parser.parse_args()
    train(
        args.input,
        Path(args.output),
        args.label_column,
        args.device,
        args.test_input,
        args.test_label_column,
        args.manual_test_column,
        args.grid,
        not args.no_manual_features,
        args.min_words,
    )
