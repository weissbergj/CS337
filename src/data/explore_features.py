import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/phase2_phase3_pairs.csv")

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

# -------------------------------
# 1. Basic Label Distribution
# -------------------------------
print("\n=== LABEL DISTRIBUTION ===")
print(df["label_success"].value_counts(normalize=True))

# -------------------------------
# 2. Text Lengths (Phase 2)
# -------------------------------
df["title_len"] = df["Brief Title"].apply(lambda x: len(str(x)))
df["interv_len"] = df["Interventions_clean"].apply(lambda x: len(str(x)))
df["cond_len"] = df["Conditions"].apply(lambda x: len(str(x)))

print("\n=== TEXT LENGTH SUMMARY ===")
print(df[["title_len", "interv_len", "cond_len"]].describe())

# -------------------------------
# 3. Common Interventions
# -------------------------------
print("\n=== MOST COMMON INTERVENTIONS ===")
print(Counter(df["Interventions_clean"]).most_common(15))

# -------------------------------
# 4. Common Conditions (diseases)
# -------------------------------
# many conditions are comma-separated
all_conditions = []
for c in df["Conditions"]:
    parts = [x.strip().lower() for x in str(c).split(",")]
    all_conditions.extend(parts)

print("\n=== MOST COMMON CONDITIONS ===")
print(Counter(all_conditions).most_common(20))

# -------------------------------
# 5. Sponsor Types
# -------------------------------
print("\n=== SPONSOR TYPE COUNT ===")
print(df["Organization Class"].value_counts())

# -------------------------------
# 6. TF–IDF Feature Exploration
# -------------------------------
print("\n=== TOP TF-IDF TERMS (INTERVENTIONS) ===")

vectorizer = TfidfVectorizer(stop_words="english", max_features=30)
tfidf = vectorizer.fit_transform(df["Interventions_clean"])

print(vectorizer.get_feature_names_out())

# -------------------------------
# 7. Correlation with success
# -------------------------------
print("\n=== SUCCESS RATES BY SPONSOR TYPE ===")
print(df.groupby("Organization Class")["label_success"].mean())

print("\n=== SUCCESS RATES BY PRIMARY PURPOSE ===")
print(df.groupby("Primary Purpose")["label_success"].mean())

print("\n=== SUCCESS RATES BY STUDY TYPE ===")
print(df.groupby("Study Type")["label_success"].mean())

print("\nDone.")
