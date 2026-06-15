import os
from pathlib import Path

# The real local drive paths are not in GitHub, because of security reasons.
data_root = Path(os.environ.get("DATA_ROOT", "map_from_gpu"))

final_dataset  = data_root / "x_knie_definitief.xlsx"
llm_labels     = data_root / "llm_labels.xlsx"
llm_labels_csv = final_dataset
clean_csv      = data_root / "dataset_clean.csv"
clean_xlsx     = data_root / "dataset_clean.xlsx"
raw_input      = data_root / "Thomas_van_Laer_MN.xlsx"
llm_source     = data_root / "llm_labels_from_labeling.xlsx"
gold_mn        = data_root / "Thomas van Laer_MN.xlsx"
gold_final     = data_root / "gold_labels_eindresultaat_met_scores_niet_geexcludeerd.xlsx"
few_shots_path = data_root / "few_shots.json"
output         = data_root
models_dir     = data_root / "models"

model          = "alibayram/medgemma:27b"
ollama_url     = "http://localhost:11434/api/generate"
checkpoint     = 1000
motivatie      = True
sample_size    = 10420  # full LLM dataset without EPIC data

label_llm          = "label_gecombineerd_consensus"
label_gold         = "handmatig_label"
label_column       = label_llm
manual_test_column = label_gold
outcome_labels     = ["afwijkend", "niet-afwijkend"]
labels             = ["afwijkend", "niet-afwijkend", "onbekend"]
text_columns       = ["klinische_gegevens_huisarts", "vraagstelling_huisarts"]

model_bert      = "CLTL/MedRoBERTa.nl"
random_state    = 2026
validation_size = 0.15
bootstrap_n     = 1000
