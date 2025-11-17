# CS337 BioDiligence – Predicting Phase III Success from Phase II Data

## 1. Overview

We train a logistic regression model to estimate whether an oncology Phase III trial
will **successfully complete**, using only information available at **Phase II**.
The model relies entirely on public metadata from ClinicalTrials.gov.

---

## 2. Research Question

**Can Phase II metadata (drug/intervention text, sponsor type, cancer context, purpose, outcomes) predict whether the corresponding Phase III trial for that same intervention will complete?**

---

## 3. Data

### 3.1 Source

- **Dataset:** ClinicalTrials.gov (Kaggle export)  
- **Rows:** 496k+ trials  
- We filter to oncology (MeSH includes *Neoplasm*) and extract all Phase II and Phase III trials.

### 3.2 Phase II → Phase III Matching

We pair Phase II and Phase III trials that share the **same cleaned intervention**.
Success is defined as Phase III `Overall Status == "COMPLETED"`.

- Output file: `phase2_phase3_pairs.csv`
- **Rows:** 5,071 matched pairs  
- Label distribution: ~49% success / 51% non-success  

---

## 4. Features & Inputs

The model uses **six Phase II–level inputs**:

### Text (TF-IDF)
1. `Interventions_clean`  
2. `Brief Title`  
3. `Conditions`  
4. `Outcome Measure`  

These are concatenated into **`combined_text`** → TF-IDF (max_features=5000).

### Categorical (One-Hot)
5. `Organization Class`  
6. `Primary Purpose`

**Target:** `label_success` ∈ {0,1}

---

## 5. Modeling

- Train/test split: 80/20, stratified  
- Preprocessing: TF-IDF + OneHotEncoder  
- Model: Logistic Regression (`max_iter=500`, `class_weight="balanced"`)  
- Training script: `src/model/train.py`  
- Saved model: `model.joblib`

---

## 6. Results

Test-set performance:

- **Accuracy:** 0.731  
- **ROC-AUC:** 0.787

---

## 7. Repository Structure

A minimal structure (omitting some files for brevity):

```bash
CS337/
├── README.md
├── data/
│   └── phase2_phase3_pairs.csv         # Cleaned, paired Phase2→Phase3 oncology dataset used for modeling
│
├── src/
│   ├── data/
│   │   ├── build_pairs.py              # Builds Phase 2 ↔ Phase 3 matched intervention dataset, labels success
│   │
│   ├── model/
│   │   ├── train.py                    # Trains logistic regression model on TF-IDF features
│   │
│   └── visuals/
│       ├── plot_feature_importance.py  # Generates feature importance barplot from trained model
│
├── model.joblib                        # Saved trained logistic regression model
│
└── visuals/
    └── feature_importance.png          # Output figure of top positive/negative predictive drug terms
