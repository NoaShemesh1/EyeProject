"""
model_seq.py — Sequential & Position-Aware Models
Tests the positional-signal-loss hypothesis three ways:

  A) Position-split means  — gaze features in early/mid/late thirds of paragraph
  B) Surprisal-weighted means — upweight unexpected words (GPT-2 surprisal as weight)
  C) Critical-span features — regression/dwell in annotated answer-bearing span vs rest
  D) Bidirectional GRU    — reads raw word sequence (12 features/word), learns which
                             positions matter without hand-crafted splits

IMPORTANT: requires combined_ia.csv — run in main eyebench repo, not the shared repo.

Input:  data/OneStop/precomputed_reading_measures/combined_ia.csv
        data/trials_9718_with_folds.csv
Output: results/results_seq.csv
        figures/fig8_seq_comparison.png
"""
import warnings; warnings.filterwarnings('ignore')
import ast, copy
import matplotlib; matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score
from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ─────────────────────────────────────────────────────────────
IA_PATH     = Path('data/OneStop/precomputed_reading_measures/combined_ia.csv')
FOLDS_PATH  = Path('data/trials_9718_with_folds.csv')
RESULTS_DIR = Path('results');  RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = Path('figures');  FIGURES_DIR.mkdir(exist_ok=True)

N_FOLDS    = 10
REGIMES    = ['unseen_reader', 'unseen_text', 'unseen_both']
MAX_LEN    = 160   # covers 99%+ of paragraphs (p95 = 147 words)
GRU_HIDDEN  = 32
GRU_DROPOUT = 0.3
GRU_EPOCHS  = 40
GRU_LR      = 1e-3
BATCH_SIZE  = 64
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EARLY_COLS = ['IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME',
              'IA_FIRST_RUN_FIXATION_COUNT']
LATE_COLS  = ['IA_REGRESSION_IN_COUNT', 'IA_REGRESSION_OUT_COUNT',
              'IA_REGRESSION_PATH_DURATION', 'IA_LAST_RUN_DWELL_TIME',
              'IA_RUN_COUNT', 'IA_DWELL_TIME']
GAZE_COLS  = EARLY_COLS + LATE_COLS   # 10 total — GRU channels 0:10
GROUP      = ['participant_id', 'article_id', 'paragraph_id']

# ── Load & filter ──────────────────────────────────────────────────────
print("Loading combined_ia.csv (~2.6M rows, takes ~2 min)...")
NEEDED = GROUP + ['IA_ID', 'IA_SKIP', 'practice_trial', 'question_preview',
                  'repeated_reading_trial', 'critical_span_indices',
                  'gpt2_surprisal', 'is_correct'] + GAZE_COLS

ia = pd.read_csv(IA_PATH, usecols=NEEDED)
ia = ia[(ia['practice_trial'] == False) &
        (ia['question_preview'] == False) &
        (ia['repeated_reading_trial'] == False)].copy()

for c in LATE_COLS:           # skipped word = zero late activity
    ia[c] = ia[c].fillna(0)

n_trials = ia.groupby(GROUP).ngroups
print(f"Filtered: {len(ia):,} rows | {n_trials:,} trials")

# ── Critical-span parser ───────────────────────────────────────────────
def parse_span(s):
    if pd.isna(s): return []
    try:    return ast.literal_eval(str(s))
    except: return []

def make_critical_flag(ia_ids, span_str):
    spans = parse_span(span_str)
    mask = np.zeros(len(ia_ids), dtype=bool)
    for s, e in spans:
        mask |= (ia_ids >= s) & (ia_ids <= e)
    return mask.astype(np.float32)

# ── Per-trial feature computation + raw sequence building ──────────────
print("Computing A/B/C features and building GRU sequences...")

trial_rows   = []
raw_sequences = {}   # key → (n_words × 13) float32 array

for key, grp in ia.groupby(GROUP):
    grp   = grp.sort_values('IA_ID')
    n     = len(grp)
    ia_ids = grp['IA_ID'].values.astype(np.float32)
    surp  = grp['gpt2_surprisal'].fillna(0).values
    crit  = make_critical_flag(ia_ids, grp['critical_span_indices'].iloc[0])

    thirds = [np.arange(n) < n // 3,
              (np.arange(n) >= n // 3) & (np.arange(n) < 2 * n // 3),
              np.arange(n) >= 2 * n // 3]

    row = {k: v for k, v in zip(GROUP, key)}

    for col in GAZE_COLS:
        vals = grp[col].values.astype(float)

        # A: position-split means (early_reg / mid_reg / late_reg)
        for mask, tag in zip(thirds, ['early_reg', 'mid_reg', 'late_reg']):
            v = vals[mask]
            row[f'{col}_{tag}'] = float(np.nanmean(v)) if v.size > 0 else 0.0

        # B: surprisal-weighted mean
        w     = surp.copy()
        w_sum = w.sum()
        v0    = np.where(np.isnan(vals), 0.0, vals)
        row[f'{col}_surp_w'] = float(np.dot(v0, w) / w_sum) if w_sum > 0 else 0.0

    # C: critical-span vs non-critical (regression + dwell)
    crit_mask  = crit.astype(bool)
    other_mask = ~crit_mask
    for col in ['IA_REGRESSION_IN_COUNT', 'IA_DWELL_TIME']:
        vals = grp[col].values.astype(float)
        c_mean = float(np.nanmean(vals[crit_mask]))  if crit_mask.sum()  > 0 else 0.0
        o_mean = float(np.nanmean(vals[other_mask])) if other_mask.sum() > 0 else 0.0
        row[f'{col}_crit']       = c_mean
        row[f'{col}_crit_ratio'] = c_mean / (o_mean + 1e-6)

    trial_rows.append(row)

    # GRU raw sequence: 10 gaze + IA_SKIP + norm_position + critical_flag = 13
    gaze = np.stack([grp[c].fillna(0).values for c in GAZE_COLS], axis=1).astype(np.float32)
    skip = grp['IA_SKIP'].fillna(0).values.reshape(-1, 1).astype(np.float32)
    pos  = ((ia_ids - 1) / max(n - 1, 1)).reshape(-1, 1)
    raw_sequences[key] = np.hstack([gaze, skip, pos, crit.reshape(-1, 1)])

feat_df = pd.DataFrame(trial_rows)
print(f"Features: {feat_df.shape[1]} columns for {len(feat_df):,} trials")

# ── Merge fold assignments ─────────────────────────────────────────────
folds   = pd.read_csv(FOLDS_PATH)
feat_df = folds.merge(feat_df, on=GROUP, how='left')

label_dict = {(r.participant_id, r.article_id, r.paragraph_id): r.is_correct
              for r in feat_df[GROUP + ['is_correct']].itertuples(index=False)}

# ── Feature set definitions ────────────────────────────────────────────
POS_FEATS   = [c for c in feat_df.columns
               if any(c.endswith(f'_{t}') for t in ['early_reg','mid_reg','late_reg'])]
SURP_FEATS  = [c for c in feat_df.columns if c.endswith('_surp_w')]
CRIT_FEATS  = [c for c in feat_df.columns if '_crit' in c]
ALL_FEATS   = POS_FEATS + SURP_FEATS + CRIT_FEATS

FEATURE_SETS = {
    'pos_split':     POS_FEATS,
    'surp_weighted': SURP_FEATS,
    'critical_span': CRIT_FEATS,
    'seq_enhanced':  ALL_FEATS,
}

# ── Metrics helper ─────────────────────────────────────────────────────
def evaluate(y_true, y_pred, y_prob):
    return {'auc':      roc_auc_score(y_true, y_prob),
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_0': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred)}

MODELS = {
    'LR':  LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'RF':  RandomForestClassifier(class_weight='balanced', n_estimators=100,
                                  random_state=42, n_jobs=-1),
    'XGB': XGBClassifier(n_estimators=100, eval_metric='logloss',
                         random_state=42, n_jobs=-1, verbosity=0),
}

# ══════════════════════════════════════════════════════════════════
# PART 1 — Traditional ML on A+B+C features
# ══════════════════════════════════════════════════════════════════
print("\n[1/2] Traditional ML on A+B+C features...")
tab_records = []

for fold in range(N_FOLDS):
    fc       = f'fold_{fold}'
    train_df = feat_df[feat_df[fc] == 'train'].copy()

    for regime in REGIMES:
        test_df = feat_df[feat_df[fc] == regime].copy()
        if len(test_df) == 0:
            continue

        for fs_name, feats in FEATURE_SETS.items():
            feats   = [f for f in feats if f in feat_df.columns]
            X_tr    = train_df[feats].fillna(0)
            X_te    = test_df[feats].fillna(0)
            y_tr    = train_df['is_correct'].values
            y_te    = test_df['is_correct'].values

            for mname, mproto in MODELS.items():
                m = copy.deepcopy(mproto)
                if mname == 'LR':
                    sc = StandardScaler()
                    Xtr_s = sc.fit_transform(X_tr)
                    Xte_s = sc.transform(X_te)
                elif mname == 'XGB':
                    ratio = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)
                    m.set_params(scale_pos_weight=ratio)
                    Xtr_s, Xte_s = X_tr.values, X_te.values
                else:
                    Xtr_s, Xte_s = X_tr.values, X_te.values

                m.fit(Xtr_s, y_tr)
                y_pred = m.predict(Xte_s)
                y_prob = m.predict_proba(Xte_s)[:, 1]
                tab_records.append({'fold': fold, 'regime': regime,
                                    'condition': fs_name, 'model': mname,
                                    **evaluate(y_te, y_pred, y_prob)})

    print(f"  tabular fold {fold} done", end='\r')

print()

# ══════════════════════════════════════════════════════════════════
# PART 2 — Bidirectional GRU
# ══════════════════════════════════════════════════════════════════
print("[2/2] Bidirectional GRU (10 folds × 3 regimes)...")
N_GRU_FEATS = 12   # 9 gaze + skip + norm_position + critical_flag

class GazeDataset(Dataset):
    def __init__(self, seqs, labels):
        self.X = np.zeros((len(seqs), MAX_LEN, N_GRU_FEATS), dtype=np.float32)
        self.L = np.ones(len(seqs), dtype=np.int64)
        self.y = np.array(labels, dtype=np.float32)
        for i, seq in enumerate(seqs):
            l = min(len(seq), MAX_LEN)
            self.X[i, :l] = seq[:l]
            self.L[i] = max(l, 1)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i):
        return (torch.tensor(self.X[i]), torch.tensor(self.L[i]), torch.tensor(self.y[i]))

class BiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru  = nn.GRU(N_GRU_FEATS, GRU_HIDDEN, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(GRU_DROPOUT)
        self.fc   = nn.Linear(GRU_HIDDEN * 2, 1)
    def forward(self, x, lengths):
        packed  = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(),
                                                     batch_first=True, enforce_sorted=False)
        _, h    = self.gru(packed)
        h       = torch.cat([h[0], h[1]], dim=1)   # (batch, hidden*2)
        return self.fc(self.drop(h)).squeeze(1)

def fold_keys(fc, split):
    sub = feat_df[feat_df[fc] == split]
    return [tuple(r) for r in sub[GROUP].itertuples(index=False)]

gru_records = []

for fold in range(N_FOLDS):
    fc         = f'fold_{fold}'
    train_keys = [k for k in fold_keys(fc, 'train') if k in raw_sequences]

    # Fit scaler on training gaze features only (channels 0:10)
    train_gaze = np.vstack([raw_sequences[k][:, :9] for k in train_keys])
    scaler     = StandardScaler().fit(train_gaze)

    def scale_seq(k):
        s = raw_sequences[k].copy()
        s[:, :9] = scaler.transform(s[:, :9])
        return s

    # Build train dataset
    tr_seqs   = [scale_seq(k) for k in train_keys]
    tr_labels = [label_dict[k] for k in train_keys]
    tr_set    = GazeDataset(tr_seqs, tr_labels)
    tr_loader = DataLoader(tr_set, batch_size=BATCH_SIZE, shuffle=True)

    y_tr      = np.array(tr_labels)
    pos_w     = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    model     = BiGRU().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LR)

    # Train
    model.train()
    for epoch in range(GRU_EPOCHS):
        for X_b, L_b, y_b in tr_loader:
            X_b, L_b, y_b = X_b.to(DEVICE), L_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b, L_b), y_b)
            loss.backward()
            optimizer.step()

    # Evaluate on each regime
    model.eval()
    with torch.no_grad():
        for regime in REGIMES:
            te_keys   = [k for k in fold_keys(fc, regime) if k in raw_sequences]
            if not te_keys:
                continue
            te_seqs   = [scale_seq(k) for k in te_keys]
            te_labels = [label_dict[k] for k in te_keys]
            te_set    = GazeDataset(te_seqs, te_labels)
            te_loader = DataLoader(te_set, batch_size=256, shuffle=False)

            all_prob, all_pred, all_true = [], [], []
            for X_b, L_b, y_b in te_loader:
                X_b, L_b = X_b.to(DEVICE), L_b.to(DEVICE)
                logits = model(X_b, L_b).cpu().numpy()
                probs  = 1 / (1 + np.exp(-logits))
                preds  = (probs >= 0.5).astype(int)
                all_prob.extend(probs); all_pred.extend(preds); all_true.extend(y_b.numpy())

            gru_records.append({'fold': fold, 'regime': regime,
                                'condition': 'bigru', 'model': 'BiGRU',
                                **evaluate(np.array(all_true),
                                           np.array(all_pred),
                                           np.array(all_prob))})

    print(f"  GRU fold {fold} done", end='\r')

print()

# ── Save results ───────────────────────────────────────────────────────
all_records = pd.DataFrame(tab_records + gru_records)
all_records.to_csv(RESULTS_DIR / 'results_seq.csv', index=False)
print("Saved results/results_seq.csv")

# ── Summary table ──────────────────────────────────────────────────────
summary = (all_records.groupby(['condition', 'model', 'regime'])['auc']
           .agg(['mean', 'std']).round(4).unstack('regime'))
print("\n" + "="*70)
print("AUC-ROC SUMMARY (mean across 10 folds)")
print("="*70)
print(summary.to_string())

# ── Figure 8: seq results vs model.py baseline ─────────────────────────
baseline_path = RESULTS_DIR / 'results_table.csv'
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, regime in zip(axes, REGIMES):
    # seq model results
    sub = all_records[all_records['regime'] == regime]
    auc_means = (sub.groupby(['condition', 'model'])['auc']
                 .mean().reset_index())

    # pick best model per condition
    best = (auc_means.sort_values('auc', ascending=False)
            .groupby('condition').first().reset_index())

    # optionally add baseline RF combined from model.py
    if baseline_path.exists():
        base = pd.read_csv(baseline_path)
        base_auc = (base[(base['model'] == 'RF') &
                         (base['condition'] == 'combined') &
                         (base['regime'] == regime)]['auc'].mean())
        baseline_row = pd.DataFrame([{'condition': 'baseline_RF_combined',
                                       'model': 'RF', 'auc': base_auc}])
        best = pd.concat([best, baseline_row], ignore_index=True)

    colors = {'pos_split': '#3498db', 'surp_weighted': '#9b59b6',
              'critical_span': '#e74c3c', 'seq_enhanced': '#2ecc71',
              'bigru': '#f39c12', 'baseline_RF_combined': '#95a5a6'}

    bars = ax.bar(best['condition'], best['auc'],
                  color=[colors.get(c, '#bdc3c7') for c in best['condition']],
                  edgecolor='white', width=0.6)
    ax.set_title(regime.replace('_', ' ').title(), fontweight='bold')
    ax.set_xlabel('')
    ax.set_xticklabels(best['condition'], rotation=35, ha='right', fontsize=8)
    ax.axhline(0.5, color='black', linewidth=0.8, linestyle='--', label='chance')
    ax.set_ylim(0.45, max(best['auc'].max() + 0.05, 0.65))
    if ax == axes[0]:
        ax.set_ylabel('Mean AUC-ROC (best model per condition)')

axes[0].legend(fontsize=8)
plt.suptitle('Sequential & Position-Aware Models vs Baseline\n(each bar = best model for that condition)',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig8_seq_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/fig8_seq_comparison.png")
print("\nDone.")
