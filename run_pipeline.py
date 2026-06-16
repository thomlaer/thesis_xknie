import argparse
import subprocess
import sys
from pathlib import Path

STEPS = {
    "clean":    ["data_cleaning.py"],
    "label":    ["llm_labeling.py"],
    "dataset":  ["build_dataset.py"],
    "baseline": ["prediction/model_baseline_tfidf.py"],
    "mlp":      ["prediction/model_mlp_tfidf.py"],
    "xgboost":  ["prediction/model_xgboost_tfidf.py"],
    "bert":     ["prediction/train_bert.py"],
    "tuning":   ["prediction/run_tuning_experiment.py"],
    "figures": [
        "figure_generation/make_kl_heatmap.py",
        "figure_generation/make_topwoorden_fig.py",
        "figure_generation/make_results_figures.py",
        "figure_generation/make_section_comparison_fig.py",
        "figure_generation/make_3way_crosstab_fig.py",
        "figure_generation/make_diagnose_crosstab_fig.py",
        "figure_generation/make_vraagstelling_fig.py",
        "figure_generation/make_llm_overview_fig.py",
        "figure_generation/regen_word_count_fig.py",
    ],
}


def run_steps(selected_steps):
    root = Path(__file__).resolve().parent
    for step in selected_steps:
        if step not in STEPS:
            raise ValueError(f"unknown step: {step!r}, choose from {list(STEPS)}")
        for script in STEPS[step]:
            cmd = [sys.executable, str(root / script)]
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("steps", nargs="+", choices=list(STEPS))
    args = parser.parse_args()
    run_steps(args.steps)
