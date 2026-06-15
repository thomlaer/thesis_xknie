# Figure generation

Scripts for generating the thesis figures.

| File | Purpose |
|------|---------|
| `make_kl_heatmap.py` | Kellgren-Lawrence confusion matrix figure |
| `make_topwoorden_fig.py` | Top-15 terms in GP referral text |
| `make_results_figures.py` | Confidence distributions, ROC/PR curves, and model comparison figures |
| `make_section_comparison_fig.py` | LLM section comparison figure |
| `make_3way_crosstab_fig.py` | 3-way cross-tab: arthrosis x trauma x fracture |
| `make_diagnose_crosstab_fig.py` | Diagnosis cross-tab figure |
| `make_vraagstelling_fig.py` | Referral question vs outcome figure |
| `make_llm_overview_fig.py` | LLM labeling pipeline overview figure |
| `regen_word_count_fig.py` | Re-render word count distribution |

Run all figure scripts from the project root with:

```bash
python run_pipeline.py figures
```
