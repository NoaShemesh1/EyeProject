"""Aggregate combined_ia.csv to trial-level (9,718 rows) and save."""
import pandas as pd

DATA_PATH = "data/OneStop/precomputed_reading_measures/combined_ia.csv"
OUT_PATH  = "data/trials_9718.csv"

COLS = [
    'participant_id','article_batch','article_id','difficulty_level','paragraph_id',
    'practice_trial','question_preview','repeated_reading_trial',
    'is_correct','PARAGRAPH_RT','IA_SKIP',
    'IA_FIRST_FIXATION_DURATION','IA_FIRST_RUN_DWELL_TIME','IA_FIRST_RUN_FIXATION_COUNT',
    'IA_REGRESSION_IN_COUNT','IA_REGRESSION_OUT_COUNT','IA_REGRESSION_PATH_DURATION',
    'IA_LAST_RUN_DWELL_TIME','IA_RUN_COUNT','IA_DWELL_TIME',
    'word_length','wordfreq_frequency','gpt2_surprisal',
]
EARLY = ['IA_FIRST_FIXATION_DURATION','IA_FIRST_RUN_DWELL_TIME','IA_FIRST_RUN_FIXATION_COUNT']
LATE  = ['IA_REGRESSION_IN_COUNT','IA_REGRESSION_OUT_COUNT','IA_REGRESSION_PATH_DURATION',
         'IA_LAST_RUN_DWELL_TIME','IA_RUN_COUNT','IA_DWELL_TIME']
LING  = ['word_length','wordfreq_frequency','gpt2_surprisal']
GROUP = ['participant_id','article_id','paragraph_id']

print("Loading...")
df = pd.read_csv(DATA_PATH, usecols=COLS)

print("Filtering...")
df = df[
    (df['practice_trial']         == False) &
    (df['question_preview']       == False) &
    (df['repeated_reading_trial'] == False)
].copy()
df['is_correct'] = df['is_correct'].astype(int)
print(f"Word-level rows after filter: {len(df):,}")

# NaN handling
for c in LATE:
    df[c] = df[c].fillna(0)   # skipped word = 0 late activity
# EARLY: leave NaN — excluded from mean automatically

print("Aggregating...")
agg = {}
for c in EARLY + LATE + LING:
    agg[f'{c}_mean']   = (c, 'mean')
    agg[f'{c}_std']    = (c, 'std')
    agg[f'{c}_max']    = (c, 'max')
    agg[f'{c}_median'] = (c, 'median')

# trial-level scalars
agg['is_correct']    = ('is_correct',    'first')
agg['PARAGRAPH_RT']  = ('PARAGRAPH_RT',  'first')
agg['skip_rate']     = ('IA_SKIP',       'mean')
agg['difficulty_level'] = ('difficulty_level', 'first')
agg['article_batch'] = ('article_batch', 'first')
agg['n_words']       = ('is_correct',    'count')   # words per trial

trials = df.groupby(GROUP).agg(**agg).reset_index()

print(f"Trials: {len(trials):,} | Columns: {len(trials.columns)}")
print(f"Participants: {trials['participant_id'].nunique()}")

trials.to_csv(OUT_PATH, index=False)
print(f"Saved to {OUT_PATH}")
