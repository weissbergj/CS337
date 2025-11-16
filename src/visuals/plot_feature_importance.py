import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = joblib.load("baseline_model.joblib")

# Extract components
pre = model.named_steps["preprocess"]
clf = model.named_steps["clf"]

tfidf = pre.named_transformers_["tfidf"]
cat = pre.named_transformers_["cat"]

tfidf_features = tfidf.get_feature_names_out()
cat_features = cat.get_feature_names_out()

all_features = np.concatenate([tfidf_features, cat_features])
coefs = clf.coef_[0]

# Top + and - features
top_pos_idx = np.argsort(coefs)[-10:][::-1]
top_neg_idx = np.argsort(coefs)[:10]

feat_names = list(all_features[top_pos_idx]) + list(all_features[top_neg_idx])
feat_values = list(coefs[top_pos_idx]) + list(coefs[top_neg_idx])

# Plot
plt.figure(figsize=(10, 6))
y_pos = np.arange(len(feat_names))

colors = ["green"] * 10 + ["red"] * 10

plt.barh(y_pos, feat_values, color=colors)
plt.yticks(y_pos, feat_names, fontsize=9)
plt.xlabel("Coefficient Weight")
plt.title("Top Positive & Negative Predictors of Phase III Success")

plt.tight_layout()
plt.gca().invert_yaxis()  # highest at top

plt.savefig("feature_importance.png", dpi=300)
print("Saved plot to feature_importance.png")
