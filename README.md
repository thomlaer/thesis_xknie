# Knee X-ray LLM/BERT thesis code

Code used for the thesis analyses on knee radiology reports.

Patient data are not included in this repository.

## Files

| Path | What it does |
|------|--------------|
| `config.py` | File names and settings |
| `data_cleaning.py` | Parses the raw PACS export |
| `llm_labeling.py` | Runs MedGemma labeling with Ollama |
| `build_dataset.py` | Combines LLM labels, exclusions, and gold labels |
| `analyse_results.py` | Makes `resultaten.txt` |
| `prediction/` | Prediction models |
| `figure_generation/` | Figure scripts |
| `run_pipeline.py` | Small script to run selected steps |

## Data

The data files contain patient information, so they are not on GitHub. To run
the code yourself, point `DATA_ROOT` to the folder where the secure data are
stored.

Example:

```powershell
$env:DATA_ROOT = "map_from_gpu"
```

Expected files in that folder:

| File | Used by | Notes |
|------|---------|-------|
| `x_knie_definitief.xlsx` | main analyses | Final primary dataset |
| `llm_labels.xlsx` | `analyse_results.py` | Full LLM dataset, including excluded rows |
| `gold_labels_eindresultaat_met_scores_niet_geexcludeerd.xlsx` | `build_dataset.py`, `analyse_results.py` | Final gold labels |
| `Thomas_van_Laer_MN.xlsx` | `data_cleaning.py` | Raw PACS export, needed when starting from the original export |
| `llm_labels_from_labeling.xlsx` | `build_dataset.py` | LLM output before merging with the gold labels |
| `few_shots.json` | `llm_labeling.py` | Few-shot examples, not included because these contain patient text |

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or 3.11 was used. CUDA is useful for the BERT model.

## Run

From the already prepared datasets:

```bash
python analyse_results.py
python run_pipeline.py baseline mlp xgboost
python run_pipeline.py bert
python run_pipeline.py figures
```

From the raw export:

```bash
python data_cleaning.py
python llm_labeling.py
python build_dataset.py
python analyse_results.py
```

## Prediction Models

`prediction/train_bert.py` trains two MedRoBERTa.nl models:

| Model | Input text |
|-------|------------|
| GP only | `klinische_gegevens_huisarts` + `vraagstelling_huisarts` |
| GP + radiology text | GP text + `bevindingen` + `conclusie` |

The BERT models train on silver LLM labels and are evaluated on the manual gold
labels. Gold-labelled records are left out of the silver training set.

## Notes

- Random seed: `2026`.
- `analyse_results.py` uses `x_knie_definitief.xlsx` plus the final gold label file.
- Binary metrics exclude labels outside `afwijkend` and `niet-afwijkend`.
- Few-shot evaluation excludes the cases that were used as few-shot examples.
