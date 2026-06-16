import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import llm_labels_csv, output


outcome_labels = ["afwijkend", "niet-afwijkend"]
yes_no_labels = ["ja", "nee"]
grade_labels = ["0", "1", "2", "3", "4"]
confidence_fields = ["bevindingen", "conclusie", "gecombineerd"]

text_columns = [
    ("klinische_gegevens_huisarts", "GP clinical information"),
    ("vraagstelling_huisarts", "GP referral question"),
    ("bevindingen", "Radiology findings"),
    ("conclusie", "Radiology conclusion"),
]

outcome_columns = [
    ("label_bevindingen", "Findings"),
    ("label_conclusie", "Conclusion"),
    ("label_gecombineerd", "Combined report"),
    ("handmatig_label", "Expert gold standard"),
]

nhg_columns = [
    ("label_nhg", "LLM NHG"),
    ("handmatig_nhg", "Expert NHG"),
]


def has_text(value):
    return value is not None and not pd.isna(value) and str(value).strip() not in ("", "nan", "None", "<NA>")


def read_table(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def count_words(value):
    if not has_text(value):
        return 0
    return len(re.findall(r"\w+", str(value)))


def normalize_outcome(value):
    if not has_text(value):
        return None
    value = str(value).strip().lower().replace("_", "-")
    value = re.sub(r"\s+", "-", value)
    if value in ("a", "afwijkend"):
        return "afwijkend"
    if value in ("n", "normaal", "normal", "niet-afwijkend"):
        return "niet-afwijkend"
    return None


def normalize_yes_no(value):
    if not has_text(value):
        return None
    value = str(value).strip().lower().replace(".", "")
    if value in ("j", "ja", "yes", "y", "1", "true"):
        return "ja"
    if value in ("n", "nee", "no", "0", "false"):
        return "nee"
    return None


def percentage(count, total):
    return float(count / total * 100) if total else 0.0


def write_table(df, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    df.to_excel(output_dir / f"{name}.xlsx", index=False)


def chosen_confidence(row, field):
    direct_column = f"conf_gekozen_{field}"
    if direct_column in row and has_text(row[direct_column]):
        return pd.to_numeric(row[direct_column], errors="coerce")

    label = normalize_outcome(row.get(f"label_{field}"))
    if label == "afwijkend":
        return pd.to_numeric(row.get(f"conf_afwijkend_{field}"), errors="coerce")
    if label == "niet-afwijkend":
        return pd.to_numeric(row.get(f"conf_niet_afwijkend_{field}"), errors="coerce")
    if has_text(row.get(f"conf_onbekend_{field}")):
        return pd.to_numeric(row.get(f"conf_onbekend_{field}"), errors="coerce")
    return np.nan


def confidence_category(value):
    if pd.isna(value):
        return "missing"
    if value >= 0.9:
        return ">=0.9"
    if value >= 0.6:
        return "0.6-0.9"
    return "<0.6"


def make_confidence_distribution(df, output_dir):
    rows = []
    categories = [">=0.9", "0.6-0.9", "<0.6", "missing"]
    for field in confidence_fields:
        label_column = f"label_{field}"
        if label_column not in df.columns:
            continue

        confidences = df.apply(lambda row: chosen_confidence(row, field), axis=1)
        category_counts = confidences.map(confidence_category).value_counts(dropna=False)
        total = int(len(confidences))

        for category in categories:
            count = int(category_counts.get(category, 0))
            rows.append(
                {
                    "field": field,
                    "confidence_group": category,
                    "n": count,
                    "percentage": percentage(count, total),
                }
            )

    result = pd.DataFrame(rows)
    write_table(result, output_dir, "confidence_distribution")

    if len(result) > 0:
        plot_data = result.pivot(index="field", columns="confidence_group", values="percentage").fillna(0)
        plot_data = plot_data[[c for c in categories if c in plot_data.columns]]
        ax = plot_data.plot(kind="bar", stacked=True, figsize=(7, 4), width=0.75)
        ax.set_xlabel("")
        ax.set_ylabel("Percentage of records")
        ax.set_ylim(0, 100)
        ax.legend(title="Chosen confidence", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "figure_confidence_distribution.pdf")
        plt.close()
    return result


def word_bin(count):
    if count == 0:
        return "0 words (empty)"
    if count <= 4:
        return "1-4 words"
    if count <= 9:
        return "5-9 words"
    if count <= 19:
        return "10-19 words"
    return "20+ words"


def make_word_count_distribution(df, output_dir):
    bins = ["0 words (empty)", "1-4 words", "5-9 words", "10-19 words", "20+ words"]
    rows = []
    long_rows = []

    for column, label in text_columns:
        if column not in df.columns:
            continue
        counts = df[column].map(count_words)
        total = int(len(counts))
        bin_counts = counts.map(word_bin).value_counts()
        row = {
            "column": column,
            "label": label,
            "mean": round(float(counts.mean()), 1),
            "median": round(float(counts.median()), 1),
            "n": total,
        }
        for bin_name in bins:
            count = int(bin_counts.get(bin_name, 0))
            row[f"{bin_name}_n"] = count
            row[f"{bin_name}_percentage"] = percentage(count, total)
            long_rows.append(
                {
                    "column": column,
                    "label": label,
                    "word_group": bin_name,
                    "n": count,
                    "percentage": percentage(count, total),
                }
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    long = pd.DataFrame(long_rows)
    write_table(summary, output_dir, "word_count_distribution")
    write_table(long, output_dir, "word_count_distribution_long")

    if len(long) > 0:
        plot_data = long.pivot(index="label", columns="word_group", values="percentage").fillna(0)
        plot_data = plot_data[[b for b in bins if b in plot_data.columns]]
        
        wc_colors = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]
        ax = plot_data.plot(kind="barh", stacked=True, figsize=(8, 4), width=0.75,
                            color=wc_colors[:len(plot_data.columns)])
        ax.set_xlabel("Percentage of records")
        ax.set_ylabel("")
        ax.set_xlim(0, 100)
        ax.legend(title="Word count", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "figure_word_count_distribution.pdf")
        plt.savefig(Path(output_dir) / "figure_word_count_distribution.png", dpi=180)
        plt.close()
    return summary


def label_distribution(df, columns, labels, normalizer, output_dir, name, figure_name):
    rows = []
    for column, display_name in columns:
        if column not in df.columns:
            continue
        values = df[column].map(normalizer)
        total = int(values.isin(labels).sum())
        counts = values.value_counts()
        for label in labels:
            count = int(counts.get(label, 0))
            rows.append(
                {
                    "column": column,
                    "label_source": display_name,
                    "label": label,
                    "n": count,
                    "percentage": percentage(count, total),
                }
            )

    result = pd.DataFrame(rows)
    write_table(result, output_dir, name)

    if len(result) > 0:
        plot_data = result.pivot(index="label_source", columns="label", values="percentage").fillna(0)
        plot_data = plot_data[[l for l in labels if l in plot_data.columns]]
        ax = plot_data.plot(kind="bar", stacked=True, figsize=(7, 4), width=0.75)
        ax.set_xlabel("")
        ax.set_ylabel("Percentage")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / figure_name)
        plt.close()
    return result


def make_confusion_matrix(df, predicted_column, true_column, labels, normalizer):
    if predicted_column not in df.columns or true_column not in df.columns:
        return None
    work = pd.DataFrame(
        {
            "predicted": df[predicted_column].map(normalizer),
            "true": df[true_column].map(normalizer),
        }
    )
    work = work[work["predicted"].isin(labels) & work["true"].isin(labels)].copy()
    if len(work) == 0:
        return None
    return pd.crosstab(work["true"], work["predicted"]).reindex(index=labels, columns=labels, fill_value=0)


def _plot_confusion_matrix_on_ax(ax, matrix, title):
    image = ax.imshow(matrix.values, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Expert label")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix.iloc[i, j])
            color = "white" if value > matrix.values.max() / 2 else "black"
            ax.text(j, i, value, ha="center", va="center", color=color)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def make_confusion_matrix_grid(df, output_dir):
    comparisons = [
        ("label_gecombineerd", "handmatig_label", outcome_labels, normalize_outcome, "Combined vs expert"),
        ("label_bevindingen", "handmatig_label", outcome_labels, normalize_outcome, "Findings vs expert"),
        ("label_conclusie", "handmatig_label", outcome_labels, normalize_outcome, "Conclusion vs expert"),
        ("label_nhg", "handmatig_nhg", yes_no_labels, normalize_yes_no, "NHG vs expert"),
    ]

    matrices = []
    for predicted, true, labels, normalizer, title in comparisons:
        matrix = make_confusion_matrix(df, predicted, true, labels, normalizer)
        if matrix is None:
            continue
        matrix.to_csv(Path(output_dir) / f"confusion_matrix_{predicted}_vs_{true}.csv", encoding="utf-8-sig")
        matrices.append((title, matrix))

    if len(matrices) == 0:
        return []

    cols = 2
    rows = int(np.ceil(len(matrices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8, 3.8 * rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, (title, matrix) in zip(axes, matrices):
        _plot_confusion_matrix_on_ax(ax, matrix, title)

    for ax in axes[len(matrices):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "figure_confusion_matrices_grid.pdf")
    plt.close()
    return matrices


def make_support_confusion_matrix_figures(llm_eval_dir, output_dir):
    llm_eval_dir = Path(llm_eval_dir)
    output_dir = Path(output_dir)

    support_cases = [
        ("support_artrose_radiologie_confusion_matrix.csv", yes_no_labels, "Osteoarthritis detection\n(LLM vs expert)"),
        ("support_artrose_graad_confusion_matrix.csv", grade_labels, "Kellgren-Lawrence grade\n(LLM vs expert)"),
    ]

    available = [(f, labels, title) for f, labels, title in support_cases if (llm_eval_dir / f).exists()]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.5))
    if len(available) == 1:
        axes = [axes]

    for ax, (filename, labels, title) in zip(axes, available):
        matrix = pd.read_csv(llm_eval_dir / filename, index_col=0)
        matrix = matrix.reindex(index=labels, columns=labels, fill_value=0)
        _plot_confusion_matrix_on_ax(ax, matrix, title)

    plt.tight_layout()
    plt.savefig(output_dir / "figure_support_confusion_matrices.pdf")
    plt.close()


def clean_roc_label(path):
    name = Path(path).stem.replace("_gold_test_roc_curve", "").replace("_gold_test_precision_recall_curve", "")
    name = name.replace("_minwords0", "").replace("_minwords10", " (>=10 words)").replace("_", " ")
    replacements = {
        "tfidf logreg": "TF-IDF logistic regression",
        "tfidf mlp": "TF-IDF MLP",
        "tfidf xgboost": "TF-IDF XGBoost",
        "transformer": "MedRoBERTa.nl",
        "multitask outcome": "Multitask outcome",
        "majority baseline": "Majority baseline",
    }
    return replacements.get(name.strip(), name.strip())


def metric_value_from_file(path, metric):
    metrics_path = Path(str(path).replace("_roc_curve.csv", "_metrics_ci.csv").replace("_precision_recall_curve.csv", "_metrics_ci.csv"))
    if not metrics_path.exists():
        return None
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return None
    row = df[df["metric"] == metric]
    if len(row) == 0:
        return None
    return pd.to_numeric(row.iloc[0]["value"], errors="coerce")


def make_combined_roc_plot(models_dir, output_dir, min_words=0):
    models_dir = Path(models_dir)
    paths = sorted(models_dir.rglob(f"*_minwords{min_words}_gold_test_roc_curve.csv"))
    if len(paths) == 0:
        (Path(output_dir) / f"figure_roc_curves_combined_minwords{min_words}_missing.txt").write_text(
            "No gold-test ROC curve files were found. Run prediction models first.\n",
            encoding="utf-8",
        )
        return []

    plt.figure(figsize=(6, 5))
    rows = []
    for path in paths:
        curve = pd.read_csv(path)
        if "false_positive_rate" not in curve.columns or "true_positive_rate" not in curve.columns:
            continue
        auc_value = metric_value_from_file(path, "roc_auc")
        label = clean_roc_label(path)
        if auc_value is not None and not pd.isna(auc_value):
            label = f"{label} (AUC {float(auc_value):.2f})"
        plt.plot(curve["false_positive_rate"], curve["true_positive_rate"], label=label)
        rows.append({"model": clean_roc_label(path), "roc_auc": auc_value, "file": str(path)})

    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"figure_roc_curves_combined_minwords{min_words}.pdf")
    plt.close()

    result = pd.DataFrame(rows)
    if len(result) > 0:
        write_table(result, output_dir, f"roc_curve_sources_minwords{min_words}")
    return rows


def make_combined_pr_plot(models_dir, output_dir, min_words=0):
    models_dir = Path(models_dir)
    paths = sorted(models_dir.rglob(f"*_minwords{min_words}_gold_test_precision_recall_curve.csv"))
    if len(paths) == 0:
        (Path(output_dir) / f"figure_pr_curves_combined_minwords{min_words}_missing.txt").write_text(
            "No gold-test PR curve files were found. Run prediction models first.\n",
            encoding="utf-8",
        )
        return []

    plt.figure(figsize=(6, 5))
    rows = []
    for path in paths:
        curve = pd.read_csv(path)
        if "precision" not in curve.columns or "recall" not in curve.columns:
            continue
        pr_auc = metric_value_from_file(path, "pr_auc")
        label = clean_roc_label(path)
        if pr_auc is not None and not pd.isna(pr_auc):
            label = f"{label} (AP {float(pr_auc):.2f})"
        plt.plot(curve["recall"], curve["precision"], label=label)
        rows.append({"model": clean_roc_label(path), "pr_auc": pr_auc, "file": str(path)})

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"figure_pr_curves_combined_minwords{min_words}.pdf")
    plt.close()

    result = pd.DataFrame(rows)
    if len(result) > 0:
        write_table(result, output_dir, f"pr_curve_sources_minwords{min_words}")
    return rows


def make_rq2_comparison_chart(llm_eval_dir, output_dir):
    llm_eval_dir = Path(llm_eval_dir)
    summary_path = llm_eval_dir / "summary_metrics.csv"
    if not summary_path.exists():
        return

    summary = pd.read_csv(summary_path)
    rq2 = summary[summary["comparison"].isin(["rq2_label_bevindingen", "rq2_label_conclusie"])].copy()
    if len(rq2) == 0:
        return

    metrics = [
        ("macro_f1", "Macro-F1"),
        ("sensitivity_afwijkend", "Sensitivity (abnormal)"),
        ("specificity_non_abnormal", "Specificity (non-abnormal)"),
        ("precision_afwijkend", "Precision (abnormal)"),
    ]

    label_map = {
        "rq2_label_bevindingen": "Findings",
        "rq2_label_conclusie": "Conclusion",
    }
    rq2["source"] = rq2["comparison"].map(label_map)

    metric_keys = [m for m, _ in metrics]
    metric_labels = [lbl for _, lbl in metrics]

    x = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (_, row) in enumerate(rq2.iterrows()):
        values = [pd.to_numeric(row.get(m), errors="coerce") for m in metric_keys]
        ci_lower = [pd.to_numeric(row.get(f"{m}_ci_95_lower"), errors="coerce") for m in metric_keys]
        ci_upper = [pd.to_numeric(row.get(f"{m}_ci_95_upper"), errors="coerce") for m in metric_keys]

        yerr_low = [
            float(v - l) if not pd.isna(v) and not pd.isna(l) else 0
            for v, l in zip(values, ci_lower)
        ]
        yerr_high = [
            float(u - v) if not pd.isna(v) and not pd.isna(u) else 0
            for v, u in zip(values, ci_upper)
        ]

        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            [v if not pd.isna(v) else 0 for v in values],
            width,
            label=row["source"],
            yerr=[yerr_low, yerr_high],
            capsize=4,
            error_kw={"linewidth": 1},
        )

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=15, ha="right")
    ax.legend(title="Input field")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "figure_rq2_bevindingen_vs_conclusie.pdf")
    plt.close()


def make_model_metrics_comparison(models_dir, output_dir, min_words=0):
    models_dir = Path(models_dir)
    paths = sorted(models_dir.rglob(f"*_minwords{min_words}_gold_test_metrics_ci.csv"))
    if len(paths) == 0:
        (Path(output_dir) / f"figure_model_metrics_minwords{min_words}_missing.txt").write_text(
            "No gold-test metrics_ci files were found. Run prediction models first.\n",
            encoding="utf-8",
        )
        return

    rows = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for metric in ["macro_f1", "accuracy"]:
            row = df[df["metric"] == metric]
            if len(row) == 0:
                continue
            rows.append(
                {
                    "model": clean_roc_label(path.name.replace("_metrics_ci.csv", "_roc_curve.csv")),
                    "metric": metric,
                    "value": pd.to_numeric(row.iloc[0]["value"], errors="coerce"),
                    "ci_lower": pd.to_numeric(row.iloc[0].get("ci_95_lower"), errors="coerce"),
                    "ci_upper": pd.to_numeric(row.iloc[0].get("ci_95_upper"), errors="coerce"),
                }
            )

    if not rows:
        return

    result = pd.DataFrame(rows)
    write_table(result, output_dir, f"model_metrics_comparison_minwords{min_words}")

    macro_f1 = result[result["metric"] == "macro_f1"].copy()
    if len(macro_f1) == 0:
        return

    macro_f1 = macro_f1.sort_values("value", ascending=True)
    xerr_low = [
        float(row["value"] - row["ci_lower"]) if not pd.isna(row["ci_lower"]) else 0
        for _, row in macro_f1.iterrows()
    ]
    xerr_high = [
        float(row["ci_upper"] - row["value"]) if not pd.isna(row["ci_upper"]) else 0
        for _, row in macro_f1.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(macro_f1) + 1)))
    y = np.arange(len(macro_f1))
    ax.barh(
        y,
        macro_f1["value"].fillna(0),
        xerr=[xerr_low, xerr_high],
        capsize=4,
        error_kw={"linewidth": 1},
        height=0.6,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(macro_f1["model"])
    ax.set_xlabel("Macro-F1 (gold test set)")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.5, linestyle="--", color="grey", linewidth=1, alpha=0.6)
    ax.grid(axis="x", alpha=0.3)
    title_suffix = "" if min_words == 0 else f" (>={min_words} words)"
    ax.set_title(f"Prediction model comparison{title_suffix}")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"figure_model_metrics_comparison_minwords{min_words}.pdf")
    plt.close()


def make_results_figures(
    input_path=llm_labels_csv,
    output_dir=output / "results_figures",
    models_dir=output / "models",
    llm_eval_dir=output / "llm_evaluation",
):
    df = read_table(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    make_confidence_distribution(df, output_dir)
    make_word_count_distribution(df, output_dir)
    label_distribution(df, outcome_columns, outcome_labels, normalize_outcome, output_dir, "outcome_label_distribution", "figure_outcome_distribution.pdf")
    label_distribution(df, nhg_columns, yes_no_labels, normalize_yes_no, output_dir, "nhg_label_distribution", "figure_nhg_distribution.pdf")
    make_confusion_matrix_grid(df, output_dir)
    make_support_confusion_matrix_figures(llm_eval_dir, output_dir)
    make_rq2_comparison_chart(llm_eval_dir, output_dir)
    make_combined_roc_plot(models_dir, output_dir, min_words=0)
    make_combined_roc_plot(models_dir, output_dir, min_words=10)
    make_combined_pr_plot(models_dir, output_dir, min_words=0)
    make_combined_pr_plot(models_dir, output_dir, min_words=10)
    make_model_metrics_comparison(models_dir, output_dir, min_words=0)
    make_model_metrics_comparison(models_dir, output_dir, min_words=10)

    print(f"Results figures saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(llm_labels_csv))
    parser.add_argument("--output", default=str(output / "results_figures"))
    parser.add_argument("--models-dir", default=str(output / "models"))
    parser.add_argument("--llm-eval-dir", default=str(output / "llm_evaluation"))
    args = parser.parse_args()
    make_results_figures(args.input, args.output, args.models_dir, args.llm_eval_dir)
