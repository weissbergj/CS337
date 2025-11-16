import pandas as pd

df = pd.read_csv("data/phase2_phase3_pairs.csv")

print("\n=== BASIC SHAPE ===")
print("Rows:", df.shape[0])
print("Columns:", list(df.columns))

print("\n=== HEAD ===")
print(df.head())

print("\n=== NULL COUNTS ===")
print(df.isnull().sum())

print("\n=== LABEL DISTRIBUTION ===")
print(df["label_success"].value_counts())
print("\n% Positive (success):", df["label_success"].mean())

print("\n=== UNIQUE TRIAL FEATURES ===")
print("Unique Phase values (raw):", df["Phases"].unique())
print("Unique Phase 3 Status:", df["Overall Status_ph3"].unique())

print("\n=== SAMPLE PHASE 2 → PHASE 3 LINKS ===")
sample_cols = [
    "Interventions_clean",
    "Brief Title",
    "Overall Status_ph3",
    "label_success"
]

print(df[sample_cols].sample(5, random_state=0))
