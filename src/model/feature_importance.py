import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load("baseline_model.joblib")

# Pull out components
pre = model.named_steps["preprocess"]
clf = model.named_steps["clf"]

# -------------------------------------------------------
# Get feature names
# -------------------------------------------------------
tfidf = pre.named_transformers_["tfidf"]
cat = pre.named_transformers_["cat"]

tfidf_features = tfidf.get_feature_names_out()
cat_features = cat.get_feature_names_out()

all_features = np.concatenate([tfidf_features, cat_features])

# Coefficients (1D)
coefs = clf.coef_[0]

# -------------------------------------------------------
# Top positive predictors (success ↑)
# -------------------------------------------------------
top_pos_idx = np.argsort(coefs)[-20:][::-1]
top_pos = [(all_features[i], coefs[i]) for i in top_pos_idx]

print("\n=== TOP POSITIVE FEATURES (predict success) ===")
for name, coef in top_pos:
    print(f"{name:25s}  {coef:.4f}")

# -------------------------------------------------------
# Top negative predictors (success ↓)
# -------------------------------------------------------
top_neg_idx = np.argsort(coefs)[:20]
top_neg = [(all_features[i], coefs[i]) for i in top_neg_idx]

print("\n=== TOP NEGATIVE FEATURES (predict failure) ===")
for name, coef in top_neg:
    print(f"{name:25s}  {coef:.4f}")
