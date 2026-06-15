import argparse
import sys
from pathlib import Path

from joblib import dump
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import llm_labels_csv, output, label_column, manual_test_column, text_columns, labels, random_state
from feature_engineering import build_feature_extractor
from model_utils import (
    load_silver_validation_gold_frames,
    save_json,
    save_model_split_outputs,
    save_split_distributions,
)


outcome_labels = [label for label in labels if label != "onbekend"]


def grid_for_mode(mode, use_manual_features=True):
    prefix = "features__tfidf__" if use_manual_features else "features__"
    if mode == "none":
        return None
    if mode == "quick":
        return {
            f"{prefix}max_features": [10000, 20000],
            f"{prefix}ngram_range": [(1, 1), (1, 2)],
            "clf__hidden_layer_sizes": [(64,), (128,)],
            "clf__alpha": [0.0001, 0.0005],
        }
    return {
        f"{prefix}max_features": [10000, 20000, 30000],
        f"{prefix}ngram_range": [(1, 1), (1, 2)],
        f"{prefix}min_df": [1, 2],
        "clf__hidden_layer_sizes": [(64,), (128,), (128, 64)],
        "clf__alpha": [0.0001, 0.0005, 0.001],
        "clf__learning_rate_init": [0.0005, 0.001],
    }


def choose_with_validation(model, param_grid, x_train, y_train, x_val, y_val, labels, output_dir, grid_name):
    if not param_grid:
        model.fit(x_train, y_train)
        return model, None

    best = None
    results = []
    for run_number, params in enumerate(ParameterGrid(param_grid), start=1):
        candidate = clone(model)
        candidate.set_params(**params)
        candidate.fit(x_train, y_train)
        prediction = candidate.predict(x_val)
        score = f1_score(y_val, prediction, labels=labels, average="macro", zero_division=0)
        results.append({"run": run_number, "params": params, "validation_f1_macro": float(score)})
        print(f"MLP grid run {run_number}: validation f1_macro {score:.4f}", flush=True)
        if best is None or score > best["score"]:
            best = {"score": score, "params": params, "model": candidate}

    save_json({"best_params": best["params"], "best_validation_f1_macro": best["score"], "grid_results": results}, output_dir, grid_name)
    return best["model"], best["params"]


def train(
    input_path=llm_labels_csv,
    output_dir=output / "models",
    label_column=label_column,
    test_input=None,
    test_label_column=None,
    manual_test_column=manual_test_column,
    grid="quick",
    use_manual_features=True,
    min_words=0,
):
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
    model_name = f"tfidf_mlp{suffix}"

    save_split_distributions(
        {"silver_train": y_train, "silver_validation": y_val, "gold_test": gold_frame["label_model"]},
        output_dir,
        model_name,
    )

    model = Pipeline(
        [
            ("features", build_feature_extractor(max_features=20000, use_manual_features=use_manual_features)),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(128,),
                    activation="relu",
                    alpha=0.0005,
                    batch_size=64,
                    learning_rate_init=0.001,
                    early_stopping=False,
                    max_iter=60,
                    random_state=random_state,
                ),
            ),
        ]
    )

    param_grid = grid_for_mode(grid, use_manual_features)
    if param_grid:
        validation_model, best_params = choose_with_validation(
            model,
            param_grid,
            x_train,
            y_train,
            x_val,
            y_val,
            outcome_labels,
            output_dir,
            f"{model_name}_grid",
        )
        model = validation_model
    else:
        model.set_params(clf__early_stopping=False)
        model.fit(x_train, y_train)

    save_model_split_outputs(model, train_frame, val_frame, gold_frame, output_dir, model_name, outcome_labels)
    dump(model, Path(output_dir) / f"{model_name}.joblib")
    if min_words == 0:
        dump(model, Path(output_dir) / "tfidf_mlp.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(llm_labels_csv))
    parser.add_argument("--output", default=str(output / "models"))
    parser.add_argument("--label-column", default=label_column)
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
        args.test_input,
        args.test_label_column,
        args.manual_test_column,
        args.grid,
        not args.no_manual_features,
        args.min_words,
    )
