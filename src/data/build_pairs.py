import pandas as pd

# 1. Load
df = pd.read_csv("data/clin_trials.csv")
print("Loaded:", df.shape)

# 2. Look at unique phases first
df["Phases_clean"] = df["Phases"].str.upper().fillna("")
print("Unique phases:", df["Phases_clean"].unique()[:10])

# 3. Oncology subset check
onc = df[df["Medical Subject Headings"].str.contains("Neoplasm", case=False, na=False)]
print("Oncology subset:", onc.shape)

# 4. Count Phase II / Phase III inside oncology
is_phase2 = onc["Phases_clean"].str.contains("PHASE2")
is_phase3 = onc["Phases_clean"].str.contains("PHASE3")
print("Phase 2 oncology count:", is_phase2.sum())
print("Phase 3 oncology count:", is_phase3.sum())

# 5. Extract phase 2 and phase 3 sets
ph2 = onc[is_phase2].copy()
ph3 = onc[is_phase3].copy()

# Show examples of interventions before cleaning
print("\nSample interventions Phase 2:", ph2["Interventions"].head().tolist())
print("Sample interventions Phase 3:", ph3["Interventions"].head().tolist())

# 6. Clean interventions
ph2["Interventions_clean"] = ph2["Interventions"].str.lower().str.strip()
ph3["Interventions_clean"] = ph3["Interventions"].str.lower().str.strip()

# 7. Show unique intervention counts
print("\nUnique Phase 2 interventions:", ph2["Interventions_clean"].nunique())
print("Unique Phase 3 interventions:", ph3["Interventions_clean"].nunique())

# 8. Find overlap
common = set(ph2["Interventions_clean"]).intersection(set(ph3["Interventions_clean"]))
print("Common interventions:", len(common))

# 9. Filter down
ph2 = ph2[ph2["Interventions_clean"].isin(common)]
ph3 = ph3[ph3["Interventions_clean"].isin(common)]

print("\nPairs after intersection:")
print("Phase 2:", ph2.shape)
print("Phase 3:", ph3.shape)

# 10. Merge
merged = ph2.merge(
    ph3[["Interventions_clean", "Overall Status"]],
    on="Interventions_clean",
    suffixes=("_ph2", "_ph3")
)

print("Merged pairs:", merged.shape)

# 11. Label success
merged["label_success"] = (merged["Overall Status_ph3"] == "COMPLETED").astype(int)

# 12. Save
merged.to_csv("data/phase2_phase3_pairs.csv", index=False)
print("\nSaved: data/phase2_phase3_pairs.csv")
