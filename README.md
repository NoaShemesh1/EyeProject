# EyeBench: Predicting Reading Comprehension from Early and Late Eye-Movement Features

## Research Question

> How much predictive information is contained in early versus late eye-movement measures for reading comprehension prediction?

We investigate whether early (first-pass) vs. late (re-reading/regression) gaze features carry different predictive signal for binary reading-comprehension outcomes, using the **OneStop** eye-tracking dataset under the **EyeBench** evaluation protocol (10-fold CV, three generalization regimes: unseen_reader, unseen_text, unseen_both).

---

## Repository Structure

```
model.py               Experiments 1–3: LR / RF / XGB on feature-set comparison,
                       permutation importance, and ablation study
model_sequential.py    Experiment 4: BiGRU + position-aware approaches
                       (pos_split, surp_weighted, critical_span, combined A+B+C)
eda.py                 Core EDA — label distribution, feature correlations, EDA figures
eda_full.py            Extended EDA — full feature sweep, extra diagnostic plots
eda_graph.py           Second EDA pass (from teammate repo)
eda_graph.md           Write-up of eda_graph.py findings
create_trials.py       Builds data/trials_9718.csv and data/trials_9718_with_folds.csv
                       from the raw EyeBench OneStop data

data/
  trials_9718.csv              Processed dataset — 9,718 participant×question trials
  trials_9718_with_folds.csv   Same, with 10 EyeBench CV fold assignments added

results/
  exp1_feature_comparison.csv  Experiment 1 — early/late/combined/text_only/combined_ling
  exp2_permutation_importance.csv  Experiment 2 — RF permutation importance on combined
  exp3_ablation.csv            Experiment 3 — ablation (remove one feature group at a time)
  exp4_sequential.csv          Experiment 4 — BiGRU + positional/sequential approaches
  results_summary.csv          Aggregated summary across experiments

  noa_exp1_feature_comparison.csv  Teammate's Experiment 1 results (different format)
  noa_exp3_ablation.csv            Teammate's Experiment 3 results
  noa_exp4_sequential.csv          Teammate's Experiment 4 results (pos_split, surp_weighted,
                                   critical_span, seq_enhanced, bigru) — source script missing,
                                   see CONFLICTS below

figures/
  fig1–fig8_*.png   Named report figures (used in Eye Project.docx)
  extra_*.png       Supplementary diagnostic plots
  noa_fig*.png      Figures from teammate repo


```


## How to Run

```bash
# Use the conda env (pip/venv xgboost breaks on macOS — missing libomp)
conda activate eyebench   # or: /Users/asi/miniforge3/envs/eyebench/bin/python

python create_trials.py        # build data/trials_9718_with_folds.csv
python eda.py                  # generate EDA figures
python model.py                # run Experiments 1–3
python model_sequential.py     # run Experiment 4
```

**Note:** `data/ia_Paragraph.csv` (5.2 GB raw eye-tracking export) is gitignored and not included. Obtain it from the EyeBench OneStop preprocessing pipeline.

---
