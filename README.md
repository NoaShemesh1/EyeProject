# EyeBench: Predicting Reading Comprehension from Early and Late Eye-Movement Features

**Course:** Language, Computation and Cognition — Technion, Spring 2026  
**Authors:** Asaf Grinshtein, Noa Shemesh

---

## Research Question

> How much predictive information is contained in early versus late eye-movement measures for reading comprehension prediction?

We investigate whether early (first-pass) vs. late (re-reading/regression) gaze features carry different predictive signal for binary reading-comprehension outcomes, using the **OneStop** eye-tracking dataset under the **EyeBench** evaluation protocol (10-fold CV, three generalization regimes: unseen_reader, unseen_text, unseen_both).

---

## Repository Structure

```
create_trials.py       Builds data/trials_9718.csv from the raw EyeBench OneStop data
eda.py                 Exploratory analysis — label distribution, feature correlations,
                       early-vs-late feature distributions
feature_models.py      Experiments 1–3: LR / RF / XGB on feature-set comparison,
                       permutation importance, and ablation study
sequential_models.py   Experiment 4: BiGRU + position-aware approaches
                       (pos_split, surp_weighted, critical_span, combined A+B+C).
                       Needs combined_ia.csv from the raw eye-tracking export, which
                       isn't included in this repo — see note under How to Run.

data/
  trials_9718_with_folds.csv   9,718 participant×question trials, with 10 EyeBench
                               CV fold assignments already added

results/
  exp1_feature_comparison.csv      Experiment 1 — early/late/combined/text_only/combined_ling
  exp2_permutation_importance.csv  Experiment 2 — RF permutation importance on combined
  exp3_ablation.csv                Experiment 3 — ablation (remove one feature group at a time)
  exp4_sequential.csv              Experiment 4 — BiGRU / positional / sequential approaches
  results_summary.csv              Aggregated mean/std summary across experiments

figures/
  fig1_dataset_overview.png, fig2_kde_early_late.png, fig3_*.png, fig4_*.png,
  fig5_early_vs_late_scatter.png, fig8_seq_comparison.png     Report figures
  noa_fig1/2/4/5/6/7_*.png                                    Same analyses, second run
  extra_01–14_*.png                                           Supplementary diagnostic plots

environment.yml      Conda environment (recommended — see note under How to Run)
requirements.txt     Pip alternative
LICENSE              MIT
```

---

## How to Run

```bash
conda env create -f environment.yml && conda activate eyebench   # recommended
# or: pip install -r requirements.txt

python create_trials.py        # build data/trials_9718.csv
python eda.py                  # generate EDA figures
python feature_models.py       # run Experiments 1–3 (needs data/trials_9718_with_folds.csv)
python sequential_models.py    # run Experiment 4
```

**Notes:**
- `sequential_models.py` needs `combined_ia.csv` (raw per-word gaze features), which comes
  from the full EyeBench OneStop preprocessing pipeline and is too large to include here.
- xgboost via pip on macOS is known to break due to a missing `libomp` runtime; if you hit
  that, install it via conda or `brew install libomp`.

---

## License

MIT — see `LICENSE`.

Built on the [EyeBench](https://github.com/EyeBench/eyebench) framework (Shubi et al., NeurIPS 2025).
