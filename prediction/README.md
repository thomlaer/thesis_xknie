# Prediction

Files for the prediction models.

| File | Use |
|------|---------|
| `feature_engineering.py` | TF-IDF and manual text features |
| `model_utils.py` | Loading, splits, metrics, and output |
| `model_baseline_tfidf.py` | Majority baseline and logistic regression |
| `model_mlp_tfidf.py` | MLP with TF-IDF features |
| `model_xgboost_tfidf.py` | XGBoost with TF-IDF features |
| `train_bert.py` | MedRoBERTa.nl: GP only and GP + radiology |
| `prediction_results_notebook.ipynb` | Prediction result tables |
| `run_tuning_experiment.py` | Threshold tuning experiment |

Run from the project root:

```bash
python run_pipeline.py baseline mlp xgboost bert
python run_pipeline.py tuning
```

Output is written to `DATA_ROOT/models`.

For the prediction result tables, open:

```text
prediction/prediction_results_notebook.ipynb
```

The notebook reads the saved output files from `DATA_ROOT/models`.
