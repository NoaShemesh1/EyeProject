"""
Early vs Late Eye-Movement Features — Modeling Pipeline
OneStop Dataset | EyeBench | Reading Comprehension

Input:  data/trials_9718_with_folds.csv
Output: results/results_table.csv
        figures/fig6_permutation_importance.png
        figures/fig7_regime_interaction.png
"""
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             recall_score, confusion_matrix)
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────
DATA_PATH   = Path('data/trials_9718_with_folds.csv')
RESULTS_DIR = Path('results')
FIGURES_DIR = Path('figures')
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── Feature groups ─────────────────────────────────────────────────
EARLY = [c for c in ['IA_FIRST_FIXATION_DURATION_mean','IA_FIRST_FIXATION_DURATION_std','IA_FIRST_FIXATION_DURATION_max',
                     'IA_FIRST_RUN_DWELL_TIME_mean','IA_FIRST_RUN_DWELL_TIME_std','IA_FIRST_RUN_DWELL_TIME_max',
                     'IA_FIRST_RUN_FIXATION_COUNT_mean','IA_FIRST_RUN_FIXATION_COUNT_std','IA_FIRST_RUN_FIXATION_COUNT_max',
                     'skip_rate']]

LATE = [c for c in ['IA_REGRESSION_IN_COUNT_mean','IA_REGRESSION_IN_COUNT_std','IA_REGRESSION_IN_COUNT_max',
                    'IA_REGRESSION_OUT_COUNT_mean','IA_REGRESSION_OUT_COUNT_std','IA_REGRESSION_OUT_COUNT_max',
                    'IA_REGRESSION_PATH_DURATION_mean','IA_REGRESSION_PATH_DURATION_std','IA_REGRESSION_PATH_DURATION_max',
                    'IA_LAST_RUN_DWELL_TIME_mean','IA_LAST_RUN_DWELL_TIME_std','IA_LAST_RUN_DWELL_TIME_max',
                    'IA_RUN_COUNT_mean','IA_RUN_COUNT_std','IA_RUN_COUNT_max',
                    'IA_DWELL_TIME_mean','IA_DWELL_TIME_std','IA_DWELL_TIME_max']]

LING = ['word_length_mean','wordfreq_frequency_mean','gpt2_surprisal_mean']

TRIAL_LEVEL = ['PARAGRAPH_RT','difficulty_level']

FEATURE_SETS = {
    'text_only':          LING,
    'early':              EARLY,
    'late':               LATE,
    'combined':           EARLY + LATE + TRIAL_LEVEL,
    'combined_ling':      EARLY + LATE + LING + TRIAL_LEVEL,
}

REGIMES   = ['unseen_reader', 'unseen_text', 'unseen_both']
N_FOLDS   = 10

# ── Load data ──────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df['difficulty_level'] = (df['difficulty_level'] == 'Adv').astype(int)
print(f"Loaded: {len(df):,} trials | {df.shape[1]} columns")

# ── Bias correction ────────────────────────────────────────────────
GAZE_FEATURES = EARLY + LATE  # only gaze features get bias-corrected

def bias_correct(train_df, test_df, features):
    """Remove participant and item bias computed on train only.
    Unseen participants/items get zero correction (no information to subtract)."""
    corrected_train = train_df.copy()
    corrected_test  = test_df.copy()
    grand_mean = train_df[features].mean()

    part_bias = train_df.groupby('participant_id')[features].mean().reset_index()
    item_bias = train_df.groupby(['article_id', 'paragraph_id'])[features].mean().reset_index()

    for df_out, df_in in [(corrected_train, train_df), (corrected_test, test_df)]:
        pb = (df_in[['participant_id']].merge(part_bias, on='participant_id', how='left')
              [features].fillna(0))
        pb.index = df_in.index

        ib = (df_in[['article_id', 'paragraph_id']].merge(
                  item_bias, on=['article_id', 'paragraph_id'], how='left')
              [features].fillna(0))
        ib.index = df_in.index

        for feat in features:
            df_out[feat] = (df_in[feat].values
                            - pb[feat].values
                            - ib[feat].values
                            + grand_mean[feat])
    return corrected_train, corrected_test

# ── Metrics ────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, y_prob):
    return {
        'auc':       roc_auc_score(y_true, y_prob),
        'macro_f1':  f1_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_0':  recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'accuracy':  accuracy_score(y_true, y_pred),
    }

# ── Main training loop ─────────────────────────────────────────────
records = []

MODELS = {
    'majority':  DummyClassifier(strategy='most_frequent'),
    'random':    DummyClassifier(strategy='stratified', random_state=42),
    'LR':        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'RF':        RandomForestClassifier(class_weight='balanced', n_estimators=100,
                                        random_state=42, n_jobs=-1),
    'XGB':       XGBClassifier(n_estimators=100, eval_metric='logloss',
                               random_state=42, n_jobs=-1, verbosity=0),
}

print("\nRunning Experiment 1 — Early vs Late vs Combined...")
for fold in range(N_FOLDS):
    fold_col = f'fold_{fold}'
    train_df = df[df[fold_col] == 'train'].copy()

    for regime in REGIMES:
        test_df = df[df[fold_col] == regime].copy()
        if len(test_df) == 0:
            continue

        for fs_name, features in FEATURE_SETS.items():
            # verify features exist
            feats = [f for f in features if f in df.columns]

            for bc in [False, True]:
                condition = f'{fs_name}_bc' if bc else fs_name

                # skip bias correction for text-only (linguistic = text descriptors)
                if bc and fs_name == 'text_only':
                    continue

                tr = train_df.copy()
                te = test_df.copy()

                if bc:
                    gaze_feats = [f for f in feats if f in GAZE_FEATURES]
                    if gaze_feats:
                        tr, te = bias_correct(tr, te, gaze_feats)

                X_train = tr[feats].fillna(0)
                X_test  = te[feats].fillna(0)
                y_train = tr['is_correct'].values
                y_test  = te['is_correct'].values

                for model_name, model_proto in MODELS.items():
                    if model_name in ['majority','random'] and (bc or fs_name != 'early'):
                        continue  # baselines only once per fold/regime

                    import copy
                    model = copy.deepcopy(model_proto)

                    if model_name == 'LR':
                        scaler = StandardScaler()
                        X_tr_s = scaler.fit_transform(X_train)
                        X_te_s = scaler.transform(X_test)
                    elif model_name == 'XGB':
                        ratio = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)
                        model.set_params(scale_pos_weight=ratio)
                        X_tr_s, X_te_s = X_train.values, X_test.values
                    else:
                        X_tr_s, X_te_s = X_train.values, X_test.values

                    model.fit(X_tr_s, y_train)
                    y_pred = model.predict(X_te_s)
                    y_prob = (model.predict_proba(X_te_s)[:,1]
                              if hasattr(model,'predict_proba') else y_pred.astype(float))

                    metrics = evaluate(y_test, y_pred, y_prob)
                    records.append({
                        'fold': fold, 'regime': regime,
                        'condition': condition if model_name not in ['majority','random'] else model_name,
                        'model': model_name,
                        **metrics
                    })

        print(f"  fold {fold} | {regime} done", end='\r')

print("\nDone. Saving results...")
results = pd.DataFrame(records)
results.to_csv(RESULTS_DIR / 'results_table.csv', index=False)

# Summary table
summary = (results.groupby(['condition','model','regime'])
           [['auc','macro_f1','recall_0','accuracy']]
           .agg(['mean','std'])
           .round(4))
summary.to_csv(RESULTS_DIR / 'results_summary.csv')
print("Saved results/results_table.csv and results/results_summary.csv")

# ── Experiment 2 — Permutation Importance ─────────────────────────
print("\nRunning Experiment 2 — Permutation Importance...")
COMBINED_FEATS = [f for f in EARLY + LATE + TRIAL_LEVEL if f in df.columns]
perm_records = []

for fold in range(N_FOLDS):
    fold_col = f'fold_{fold}'
    train_df = df[df[fold_col] == 'train'].copy()
    test_df  = df[df[fold_col] == 'unseen_reader'].copy()
    if len(test_df) == 0:
        continue

    X_train = train_df[COMBINED_FEATS].fillna(0).values
    X_test  = test_df[COMBINED_FEATS].fillna(0).values
    y_train = train_df['is_correct'].values
    y_test  = test_df['is_correct'].values

    rf = RandomForestClassifier(class_weight='balanced', n_estimators=100,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    perm = permutation_importance(rf, X_test, y_test,
                                  n_repeats=5, random_state=42,
                                  scoring='roc_auc', n_jobs=-1)
    for i, feat in enumerate(COMBINED_FEATS):
        perm_records.append({
            'fold': fold, 'feature': feat,
            'importance_mean': perm.importances_mean[i],
            'importance_std':  perm.importances_std[i],
        })
    print(f"  fold {fold} permutation done", end='\r')

perm_df = pd.DataFrame(perm_records)
perm_summary = (perm_df.groupby('feature')['importance_mean']
                .mean()
                .sort_values(ascending=False)
                .reset_index())
perm_summary['group'] = perm_summary['feature'].apply(
    lambda f: 'Early' if any(e in f for e in ['FIRST','skip'])
    else ('Late' if any(l in f for l in ['REGRESSION','LAST_RUN','RUN_COUNT','DWELL'])
    else 'Trial-level'))
perm_summary.to_csv(RESULTS_DIR / 'permutation_importance.csv', index=False)

# Figure 6
group_colors = {'Early':'#3498db','Late':'#e67e22','Trial-level':'#95a5a6'}
top20 = perm_summary.head(20)
colors = [group_colors.get(g,'#bdc3c7') for g in top20['group']]

plt.figure(figsize=(10, 7))
plt.barh(top20['feature'].str.replace('IA_','').str.replace('_mean',''),
         top20['importance_mean'], color=colors, edgecolor='white')
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Permutation Feature Importance (RF, Combined model, Unseen Reader)\nTop 20 features', fontweight='bold')
plt.xlabel('Mean AUC drop when feature shuffled')
for g, c in group_colors.items():
    plt.barh([], [], color=c, label=g)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig6_permutation_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved fig6_permutation_importance.png")

# ── Experiment 3 — Ablation ────────────────────────────────────────
print("\nRunning Experiment 3 — Ablation Study...")
REGRESSION_FEATS = [f for f in LATE if 'REGRESSION' in f]
FIRST_FIX_FEATS  = [f for f in EARLY if 'FIRST_FIXATION' in f or 'FIRST_RUN' in f]

ablation_sets = {
    'combined_full':      EARLY + LATE + TRIAL_LEVEL,
    'no_regression':      [f for f in EARLY + LATE + TRIAL_LEVEL if f not in REGRESSION_FEATS],
    'no_first_fixation':  [f for f in EARLY + LATE + TRIAL_LEVEL if f not in FIRST_FIX_FEATS],
    'no_paragraph_rt':    [f for f in EARLY + LATE + TRIAL_LEVEL if f != 'PARAGRAPH_RT'],
    'no_late':            EARLY + TRIAL_LEVEL,
    'no_early':           LATE + TRIAL_LEVEL,
}

ablation_records = []
for fold in range(N_FOLDS):
    fold_col = f'fold_{fold}'
    train_df = df[df[fold_col] == 'train'].copy()

    for regime in REGIMES:
        test_df = df[df[fold_col] == regime].copy()
        if len(test_df) == 0:
            continue

        for abl_name, feats in ablation_sets.items():
            feats = [f for f in feats if f in df.columns]
            X_train = train_df[feats].fillna(0).values
            X_test  = test_df[feats].fillna(0).values
            y_train = train_df['is_correct'].values
            y_test  = test_df['is_correct'].values

            rf = RandomForestClassifier(class_weight='balanced', n_estimators=100,
                                        random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            y_prob = rf.predict_proba(X_test)[:,1]

            metrics = evaluate(y_test, y_pred, y_prob)
            ablation_records.append({'fold':fold,'regime':regime,'ablation':abl_name,**metrics})

    print(f"  ablation fold {fold} done", end='\r')

ablation_df = pd.DataFrame(ablation_records)
ablation_df.to_csv(RESULTS_DIR / 'ablation_results.csv', index=False)
print("\nSaved ablation_results.csv")

# ── Experiment 4 — Regime × Early/Late Interaction ─────────────────
print("\nRunning Experiment 4 — Regime x Early/Late Interaction...")

palette = {'early':'#3498db','late':'#e67e22','combined':'#2ecc71','text_only':'#9b59b6'}
order   = ['unseen_reader','unseen_text','unseen_both']
order_labels = ['Unseen\nReader','Unseen\nText','Unseen\nBoth']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, model_name in zip(axes, ['RF', 'XGB']):
    sub_results = (results[results['model'] == model_name]
                   [results['condition'].isin(['early','late','combined','text_only'])]
                   .groupby(['condition','regime'])['auc']
                   .agg(['mean','std'])
                   .reset_index())
    for cond in ['early','late','combined','text_only']:
        sub = sub_results[sub_results['condition']==cond].set_index('regime')
        vals = [sub.loc[r,'mean'] if r in sub.index else np.nan for r in order]
        errs = [sub.loc[r,'std']  if r in sub.index else np.nan for r in order]
        ax.errorbar(order_labels, vals, yerr=errs, label=cond,
                    color=palette[cond], marker='o', linewidth=2, capsize=4)
    ax.set_xlabel('Generalization Regime')
    ax.set_ylabel('Mean AUC-ROC')
    ax.set_title(f'Regime × Feature Group Interaction ({model_name})\nKey: does early/late gap change across regimes?',
                 fontweight='bold')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig7_regime_interaction.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig7_regime_interaction.png")

# ── Final summary printout ─────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS SUMMARY (RF + XGB, mean across 10 folds)")
print("="*60)
rf_summary = (results[results['model'].isin(['RF','XGB'])]
              .groupby(['condition','model','regime'])['auc']
              .mean()
              .unstack()
              .round(4))
print(rf_summary.to_string())
print("\nAll outputs saved to results/ and figures/")

# ══════════════════════════════════════════════════════════════════
# POST-RESULTS ANALYSIS & NEXT STEPS
# (based on observed near-chance gaze AUC, text_only winning)
# ══════════════════════════════════════════════════════════════════
#
# EXPECTED PATTERN IF RESULTS ARE NEAR-CHANCE FOR GAZE:
#   text_only AUC ≈ 0.56–0.64  (text difficulty is real signal)
#   early/late/combined AUC ≈ 0.49–0.55  (indistinct from random)
#   bias correction (_bc) ≈ no consistent improvement
#   ablation: removing regression features does not hurt (H3 disproven)
#
# TWO PLAUSIBLE EXPLANATIONS (not mutually exclusive):
#
#   1. POSITIONAL SIGNAL LOSS (most likely culprit)
#      Aggregating regression counts over the whole paragraph mixes
#      diagnostic re-reading (in the answer-bearing sentences) with
#      noise (hard words elsewhere). A reader who regresses into the
#      last sentence — where the answer usually lives — is qualitatively
#      different from one who regresses into the opening. Mean/std
#      cannot distinguish these two profiles.
#
#   2. GENUINE WEAK SIGNAL
#      81% of trials are correct. The readers who fail may do so for
#      reasons with no stable gaze signature (misread question, lucky
#      guess, momentary lapse). LR getting ~0.53–0.55 while RF/XGB
#      stays at ~0.49–0.51 suggests the true signal, if present, is
#      a weak linear combination — not an interaction trees can exploit.
#
# RECOMMENDED NEXT STEP — POSITIONAL FEATURES (cheap, high-impact):
#   From the raw IA-level file, compute per trial:
#
#   a) regression_ratio_late =
#         mean(IA_REGRESSION_IN_COUNT for words in top 33% of IA_ID)
#         / (mean(IA_REGRESSION_IN_COUNT for all words) + 1e-6)
#      Captures whether regressions are CONCENTRATED in the final
#      paragraph region (answer-bearing) vs spread over the whole text.
#
#   b) dwell_ratio_late =
#         mean(IA_DWELL_TIME for words in top 33% of IA_ID)
#         / (mean(IA_DWELL_TIME for all words) + 1e-6)
#      Same idea for total reading time.
#
#   c) first_fixation_late_vs_early =
#         mean(IA_FIRST_FIXATION_DURATION for top 33% IA_ID)
#         - mean(IA_FIRST_FIXATION_DURATION for bottom 33% IA_ID)
#      Positive = first-pass slowing in the late region.
#
#   All three require only: groupby(participant, article, paragraph),
#   split IA_ID into thirds, compute means. ~15 lines of pandas.
#   Then retrain this pipeline unchanged with the new feature columns.
#
# ALTERNATIVE AGGREGATIONS (if rebuilding create_trials.py):
#
#   Instead of / in addition to global mean+std+max+median, consider:
#
#   A) POSITION-SPLIT MEANS — for each feature, compute mean separately
#      for early/mid/late thirds of the paragraph (3× the features).
#      Directly captures where in the text the behavior occurs.
#      Best bang-for-buck upgrade.
#
#   B) SURPRISAL-WEIGHTED MEAN — weight each word's gaze measure by
#      its GPT-2 surprisal before averaging. Upweights unexpected words,
#      which are the ones where comprehension effort is most diagnostic.
#         weighted_dwell = sum(dwell_i * surprisal_i) / sum(surprisal_i)
#
#   C) ANOMALY COUNT — count of words where a gaze measure exceeds
#      2 SD above the participant's own mean for that feature in that
#      trial. Captures "stumbling" events rather than average behavior.
#         n_anomalous_fixations = sum(dwell_i > mu + 2*sigma)
#
#   D) RE-READING BOUT COUNT — number of distinct contiguous backward
#      passes (not just total regression count). A reader with 3 clean
#      re-reading bouts is different from one with 3 scattered regressions.
#      Requires sequential IA_ID order; count sign-flips in IA_ID sequence.
#
#   Of these, A (position split) is the fastest to implement and most
#   directly tied to the positional-loss hypothesis. B (surprisal-weighted)
#   is the most theoretically motivated. C and D are interesting but
#   add implementation complexity for uncertain gain.
#
# HYPERPARAMETER SEARCH: low priority.
#   The gaze AUC gap vs text_only is 5–10 percentage points.
#   Tuning realistically recovers 1–3% at best. Does not change narrative.
#   If attempted: tune only LR's C ∈ {0.01, 0.1, 1, 10} via CV on train.
# ══════════════════════════════════════════════════════════════════
