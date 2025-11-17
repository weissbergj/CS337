import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# Load data
df = pd.read_csv("data/phase2_phase3_pairs.csv")

label_col = "label_success"

# Build a combined text field
# (Phase II only: interventions + title + conditions + outcome text)
df["combined_text"] = (
    df["Interventions_clean"].fillna("") + " "
    + df["Brief Title"].fillna("") + " "
    + df["Conditions"].fillna("") + " "
    + df["Outcome Measure"].fillna("")
)

text_col = "combined_text"
cat_cols = ["Organization Class", "Primary Purpose"]

X = df[[text_col] + cat_cols]
y = df[label_col]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Transformers
text_tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
)

cat_encoder = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("text", text_tfidf, text_col),
        ("cat", cat_encoder, cat_cols),
    ]
)

# Build model pipeline
model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
    ]
)

# Train
print("Training model...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n=== UPDATED MODEL RESULTS ===")
print(f"Accuracy: {acc:.3f}")
print(f"ROC-AUC:  {auc:.3f}")

# Save model
joblib.dump(model, "model.joblib")
print("\nSaved: model.joblib")