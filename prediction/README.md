# Prediction code

Scripts for predicting the binary radiological outcome.

| File | Purpose |
|------|---------|
| `feature_engineering.py` | TF-IDF and manually defined referral-text features |
| `model_utils.py` | Shared data loading, silver/gold splitting, metrics, and output helpers |
| `model_baseline_tfidf.py` | Majority baseline and logistic regression with TF-IDF features |
| `model_mlp_tfidf.py` | MLP classifier with TF-IDF features |
| `model_xgboost_tfidf.py` | XGBoost classifier with TF-IDF features |
| `train_bert.py` | MedRoBERTa.nl models: GP only and GP + radiology text |
| `run_tuning_experiment.py` | Extra tuning/threshold experiments |
| `run_extras.py` | Extra checks such as McNemar and subgroup analyses |
| `run_consensus_split.py` | Internal 80/20 consensus-label split |

Run the final prediction models from the project root with:

```bash
python run_pipeline.py baseline mlp xgboost bert
```

Outputs are written to `DATA_ROOT/models`.
