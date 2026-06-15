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
| `run_tuning_experiment.py` | Threshold tuning |
| `run_extras.py` | McNemar and subgroup checks |
| `run_consensus_split.py` | 80/20 split on consensus labels |

Run from the project root:

```bash
python run_pipeline.py baseline mlp xgboost bert
```

Output is written to `DATA_ROOT/models`.
