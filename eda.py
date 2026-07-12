# EDA — Early vs. Late Eye-Movement Features for Reading Comprehension
# OneStop Dataset | EyeBench Project

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')
FIGURES_DIR = 'figures/'

# ── Data loading ─────────────────────────────────���──────────────────────────
COLS = [
    'participant_id','article_batch','article_id','difficulty_level','paragraph_id',
    'practice_trial','question_preview','repeated_reading_trial',
    'is_correct','PARAGRAPH_RT','IA_SKIP',
    'IA_FIRST_FIXATION_DURATION','IA_FIRST_RUN_DWELL_TIME','IA_FIRST_RUN_FIXATION_COUNT',
    'IA_REGRESSION_IN_COUNT','IA_REGRESSION_OUT_COUNT','IA_REGRESSION_PATH_DURATION',
    'IA_LAST_RUN_DWELL_TIME','IA_RUN_COUNT','IA_DWELL_TIME',
    'word_length','wordfreq_frequency','gpt2_surprisal'
]

DATA_PATH = 'data/OneStop/precomputed_reading_measures/combined_ia.csv'

print('Loading data...')
df = pd.read_csv(DATA_PATH, usecols=COLS)
df = df[
    (df['practice_trial'] == False) &
    (df['question_preview'] == False) &
    (df['repeated_reading_trial'] == False)
].copy()
df['is_correct'] = df['is_correct'].astype(int)
print(f'Word-level rows after filtering: {len(df):,}')

# ── Feature groups ───────────────────────────────────────────────────────────
EARLY = ['IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME', 'IA_FIRST_RUN_FIXATION_COUNT']
LATE  = ['IA_REGRESSION_IN_COUNT', 'IA_REGRESSION_OUT_COUNT', 'IA_REGRESSION_PATH_DURATION',
         'IA_LAST_RUN_DWELL_TIME', 'IA_RUN_COUNT', 'IA_DWELL_TIME']
LING  = ['word_length', 'wordfreq_frequency', 'gpt2_surprisal']
GROUP_COLS = ['participant_id', 'article_batch', 'article_id', 'difficulty_level', 'paragraph_id']

# ── NaN handling: LATE fills 0 (skipped word = no re-reading) ───────────────
for c in LATE:
    df[c] = df[c].fillna(0)
# EARLY: leave NaN — excluded from mean automatically

# ── Trial-level aggregation ──────────────────────────────────────────────────
agg = {c: 'mean' for c in EARLY + LATE + LING}
agg.update({'is_correct': 'first', 'PARAGRAPH_RT': 'first', 'IA_SKIP': 'mean'})

trials = df.groupby(GROUP_COLS).agg(agg).reset_index()
trials.rename(columns={'IA_SKIP': 'skip_rate'}, inplace=True)
print(f'Trial-level rows: {len(trials):,} | Participants: {trials["participant_id"].nunique()}')

# ─────────────────────────────���──────────────────────────────���───────────────
# FIGURE 1 — Dataset overview: class balance + accuracy by difficulty
# ────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Panel A: class balance
cb = trials['is_correct'].value_counts().sort_index()
bars = axes[0].bar(['Incorrect', 'Correct'], cb.values,
                   color=['#e74c3c', '#2ecc71'], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, cb.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f'{val:,}\n({val/len(trials):.1%})', ha='center', va='bottom', fontsize=11)
axes[0].set_title('A. Target Variable Distribution', fontweight='bold')
axes[0].set_ylabel('Number of Trials')
axes[0].set_ylim(0, cb.max() * 1.15)

# Panel B: accuracy by difficulty level
diff_acc = trials.groupby('difficulty_level')['is_correct'].mean().reset_index()
diff_acc.columns = ['Level', 'Accuracy']
axes[1].bar(diff_acc['Level'], diff_acc['Accuracy'],
            color=['#3498db', '#9b59b6'], edgecolor='white', linewidth=1.5)
for i, row in diff_acc.iterrows():
    axes[1].text(i, row['Accuracy'] + 0.005, f'{row["Accuracy"]:.1%}',
                 ha='center', va='bottom', fontsize=11)
axes[1].set_title('B. Accuracy by Text Difficulty Level', fontweight='bold')
axes[1].set_ylabel('Proportion Correct')
axes[1].set_ylim(0, 1.0)
axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
axes[1].legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR + 'fig1_dataset_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig1')

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — KDE: Early vs Late features by is_correct (2×2)
# ────────────────────────────────────────────────────────────────────────────
EARLY_PLOT = [
    ('IA_FIRST_FIXATION_DURATION', 'First Fixation Duration (ms)', 'A'),
    ('IA_FIRST_RUN_DWELL_TIME',    'Gaze Duration / First-Pass Dwell (ms)', 'B'),
]
LATE_PLOT = [
    ('IA_REGRESSION_PATH_DURATION', 'Go-Past Time (ms)', 'C'),
    ('IA_REGRESSION_IN_COUNT',      'Regression-In Count', 'D'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
colors = {'correct': '#2ecc71', 'incorrect': '#e74c3c'}

for ax, (feat, label, panel) in zip(axes.flat[:2], EARLY_PLOT):
    data_c = trials[trials['is_correct'] == 1][feat].dropna()
    data_i = trials[trials['is_correct'] == 0][feat].dropna()
    sns.kdeplot(data_c, ax=ax, label='Correct', fill=True,
                color=colors['correct'], alpha=0.5)
    sns.kdeplot(data_i, ax=ax, label='Incorrect', fill=True,
                color=colors['incorrect'], alpha=0.5)
    ax.set_title(f'{panel}. {label} [EARLY]', fontweight='bold')
    ax.set_xlabel(label)
    ax.legend()

for ax, (feat, label, panel) in zip(axes.flat[2:], LATE_PLOT):
    data_c = trials[trials['is_correct'] == 1][feat].dropna()
    data_i = trials[trials['is_correct'] == 0][feat].dropna()
    sns.kdeplot(data_c, ax=ax, label='Correct', fill=True,
                color=colors['correct'], alpha=0.5)
    sns.kdeplot(data_i, ax=ax, label='Incorrect', fill=True,
                color=colors['incorrect'], alpha=0.5)
    ax.set_title(f'{panel}. {label} [LATE]', fontweight='bold')
    ax.set_xlabel(label)
    ax.legend()

fig.suptitle('Early vs. Late Feature Distributions by Comprehension Outcome\n(Trial-Level Means)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR + 'fig2_kde_early_late.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig2')

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Correlation: heatmap + bar chart of avg |r| by group
# ────────────────────────────────────────────────────────────────────────────
ALL_FEATS = EARLY + LATE + ['PARAGRAPH_RT', 'skip_rate']
corr_mat  = trials[ALL_FEATS + ['is_correct']].corr()
target_corr = corr_mat['is_correct'].drop('is_correct')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: heatmap
feat_labels = {
    'IA_FIRST_FIXATION_DURATION': 'First Fix Dur',
    'IA_FIRST_RUN_DWELL_TIME':    'Gaze Dur',
    'IA_FIRST_RUN_FIXATION_COUNT':'First Run Fix Count',
    'IA_REGRESSION_IN_COUNT':     'Regression In',
    'IA_REGRESSION_OUT_COUNT':    'Regression Out',
    'IA_REGRESSION_PATH_DURATION':'Go-Past Time',
    'IA_LAST_RUN_DWELL_TIME':     'Last Run Dwell',
    'IA_RUN_COUNT':               'Run Count',
    'IA_DWELL_TIME':              'Total Dwell',
    'PARAGRAPH_RT':               'Paragraph RT',
    'skip_rate':                  'Skip Rate',
    'is_correct':                 'is_correct'
}
sub_feats = EARLY + LATE + ['is_correct']
sub_corr  = corr_mat.loc[sub_feats, sub_feats].rename(
    index=feat_labels, columns=feat_labels)
mask = np.triu(np.ones_like(sub_corr, dtype=bool), k=1)
sns.heatmap(sub_corr, ax=axes[0], annot=True, fmt='.2f',
            cmap='coolwarm', center=0, vmin=-1, vmax=1,
            mask=mask, linewidths=0.3,
            annot_kws={'size': 7})
axes[0].set_title('A. Feature Correlation Matrix', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45, labelsize=8)
axes[0].tick_params(axis='y', rotation=0,  labelsize=8)

# Panel B: avg |r| with is_correct by group
group_corr = pd.DataFrame({
    'Group': ['Early', 'Late', 'Total (RT+Skip)'],
    'Avg |r| with is_correct': [
        target_corr[EARLY].abs().mean(),
        target_corr[LATE].abs().mean(),
        target_corr[['PARAGRAPH_RT','skip_rate']].abs().mean()
    ]
})
bar_colors = ['#3498db', '#e67e22', '#95a5a6']
axes[1].bar(group_corr['Group'], group_corr['Avg |r| with is_correct'],
            color=bar_colors, edgecolor='white', linewidth=1.5)
for i, val in enumerate(group_corr['Avg |r| with is_correct']):
    axes[1].text(i, val + 0.001, f'{val:.3f}', ha='center', va='bottom', fontsize=11)
axes[1].set_title('B. Average |r| with Comprehension by Feature Group', fontweight='bold')
axes[1].set_ylabel('Mean |Pearson r|')
axes[1].set_ylim(0, max(group_corr['Avg |r| with is_correct']) * 1.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR + 'fig3_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig3')

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Participant-level variance (motivates bias correction)
# ────────────────────────────────────────────────────────────────────────────
part = trials.groupby('participant_id').agg(
    accuracy=('is_correct', 'mean'),
    avg_rt=('PARAGRAPH_RT', 'mean'),
    n_trials=('is_correct', 'count')
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: participant accuracy distribution
sns.histplot(part['accuracy'], bins=20, kde=True, ax=axes[0],
             color='#3498db', edgecolor='white')
axes[0].axvline(part['accuracy'].mean(), color='red', linestyle='--',
                label=f'Mean = {part["accuracy"].mean():.1%}')
axes[0].set_title('A. Participant-Level Accuracy Distribution', fontweight='bold')
axes[0].set_xlabel('Proportion Correct')
axes[0].set_ylabel('Number of Participants')
axes[0].legend()

# Panel B: RT vs accuracy scatter
sc = axes[1].scatter(part['avg_rt']/1000, part['accuracy'],
                     s=part['n_trials']*2, alpha=0.7,
                     c=part['accuracy'], cmap='RdYlGn', edgecolors='gray', linewidth=0.5)
plt.colorbar(sc, ax=axes[1], label='Accuracy')
axes[1].set_title('B. Reading Speed vs. Accuracy (per Participant)\n(bubble size = number of trials)',
                  fontweight='bold')
axes[1].set_xlabel('Mean Paragraph Reading Time (s)')
axes[1].set_ylabel('Proportion Correct')

plt.tight_layout()
plt.savefig(FIGURES_DIR + 'fig4_participant_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig4')

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Trial-level scatter: Early vs Late colored by is_correct
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

for label, color, marker in [(0, '#e74c3c', 'x'), (1, '#2ecc71', 'o')]:
    sub = trials[trials['is_correct'] == label]
    ax.scatter(sub['IA_FIRST_RUN_DWELL_TIME'], sub['IA_REGRESSION_IN_COUNT'],
               alpha=0.25, s=15, c=color, marker=marker,
               label='Correct' if label == 1 else 'Incorrect')

ax.set_xlabel('Mean First-Pass Dwell Time / Trial (ms)  [EARLY]')
ax.set_ylabel('Mean Regression-In Count / Trial  [LATE]')
ax.set_title('Early vs. Late Eye-Movement Features\n(Trial-Level, Colored by Comprehension Outcome)',
             fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR + 'fig5_early_vs_late_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig5')

print('\nAll figures saved to', FIGURES_DIR)
