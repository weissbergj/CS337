# CS337 Project – Predicting Phase III Trial Success from Phase II Data

## 1. Overview

This project builds a simple, interpretable model to estimate whether an oncology Phase III clinical trial will **successfully complete** based on information available at **Phase II**.

We use public trial metadata (from ClinicalTrials.gov via a Kaggle dump) to learn which **drugs** and **sponsors** are historically associated with successful Phase III completion.

---

## 2. Research Question

> **Given metadata from an oncology Phase II trial (especially the intervention/drug and sponsor type), can we predict whether a corresponding Phase III trial for the same intervention will successfully complete?**

Concretely, we define:

- **Input:** Phase II trial intervention text (drug/regimen) and sponsor class  
- **Target:** Whether the matched Phase III trial’s **Overall Status** is `COMPLETED` vs anything else  
- **Task:** Binary classification (success vs non-success)

---

## 3. Data

### 3.1 Source

**Modeling dataset summary:**  
- **Final dataset:** **5,071** matched **oncology Phase II → Phase III** trial pairs  
- **Label balance:** ~49% success / 51% non-success  

We use the Kaggle dataset:

- **Dataset:** *ClinicalTrials.gov Clinical Trials dataset*  
- **Author:** Daniel Ansted  
- **File:** `data/clin_trials.csv`  
- **Rows:** 496,615 trials  
- **Columns (17 total):**  
  - Trial sponsor/org info  
  - Titles (brief / full)  
  - Overall Status  
  - Start Date, Standard Age  
  - Conditions  
  - Primary Purpose, Study Type, Phases  
  - Interventions + Intervention Description  
  - Outcome Measure  
  - Medical Subject Headings  

This dataset contains **metadata only** (no numeric efficacy/safety outcomes).

### 3.2 Task-Specific Dataset Construction

We build a Phase II → Phase III oncology dataset in several steps:

1. **Oncology subset**  
   - Keep trials where `Medical Subject Headings` contains `"Neoplasm"`.

2. **Phase cleaning**  
   - Standardize `Phases` to uppercase (`PHASE2`, `PHASE3`, `PHASE2, PHASE3`, etc.).
   - Define:
     - Phase II trials: `Phases_clean` contains `"PHASE2"`
     - Phase III trials: `Phases_clean` contains `"PHASE3"`

3. **Intervention normalization**  
   - Create `Interventions_clean`:
     - lowercase
     - stripped whitespace

4. **Phase II ↔ Phase III pairing**  
   - Split oncology subset into Phase II (`ph2`) and Phase III (`ph3`).
   - Find interventions that appear in **both** Phase II and Phase III:
     - `common = set(ph2.Interventions_clean) ∩ set(ph3.Interventions_clean)`
   - Keep only trials whose `Interventions_clean` is in `common`.
   - Merge Phase II and Phase III on `Interventions_clean`:

     - Phase II columns retained as features  
     - Phase III `Overall Status` kept as outcome (`Overall Status_ph3`)

5. **Label definition (success vs non-success)**  
   - **Success (1):** `Overall Status_ph3 == "COMPLETED"`  
   - **Non-success (0):** all other statuses (`RECRUITING`, `ACTIVE_NOT_RECRUITING`, `TERMINATED`, `WITHDRAWN`, `SUSPENDED`, `NOT_YET_RECRUITING`, etc.)

6. **Final modeling dataset**

   - Output file: `data/phase2_phase3_pairs.csv`  
   - **Rows:** 5,071 Phase II → Phase III pairs  
   - **Label balance:**  
     - Success (1): 2,481 (~49%)  
     - Non-success (0): 2,590 (~51%)  

---

## 4. Features & Inputs

For the baseline model, we intentionally use **simple, interpretable features**:

### 4.1 Inputs (X)

From **Phase II** trials:

1. **Intervention text** – `Interventions_clean`  
   - Examples: `capecitabine`, `nivolumab`, `bortezomib`, `radiation therapy`, `capecitabine, oxaliplatin`  
   - Processed using **TF–IDF** (`max_features=2000`, English stopwords removed)

2. **Sponsor type** – `Organization Class`  
   - Categories include: `INDUSTRY`, `NIH`, `NETWORK`, `OTHER`, etc.  
   - Encoded via **one-hot encoding** with `handle_unknown="ignore"`.

### 4.2 Target (y)

- **`label_success`** ∈ {0, 1}  
  - **1** = Phase III overall status is `COMPLETED`  
  - **0** = Phase III is not completed (any other status)

No patient-level data, no Phase II outcomes, and no numeric effect sizes are used.  
The model is learning **drug/sponsor–level development patterns**, not biology per se.

---

## 5. Modeling

### 5.1 Train/Test Split

- Train/test split on `phase2_phase3_pairs.csv`:
  - `test_size = 0.2`  
  - `random_state = 42`  
  - `stratify = y` (preserve label balance)

### 5.2 Pipeline

We use a scikit-learn `Pipeline`:

1. **Preprocessing (`ColumnTransformer`):**
   - `TfidfVectorizer` on `Interventions_clean` (`max_features=2000`)  
   - `OneHotEncoder` on `Organization Class`  

2. **Classifier:**
   - `LogisticRegression` (with `max_iter=500`, `class_weight="balanced"`)

### 5.3 Training Script

- Script: `src/model/train_baseline.py`  
- Output model: `baseline_model.joblib`

---

## 6. Results

On the held-out test set (20% of data):

- **Accuracy:** 0.761  
- **ROC–AUC:** 0.841  

These numbers indicate the model can **reliably distinguish** Phase III trials that complete vs those that do not, using only Phase II intervention text and sponsor type.

---

## 7. Interpretability & Visuals

### 7.1 Feature Importance

We extract coefficients from the logistic regression model to see which features most strongly influence predicted success:

- Script: `src/model/feature_importance.py`  
- The model assigns high positive weights to interventions such as:  
  - `nivolumab`, `letrozole`, `bortezomib`, `olaparib`, `trastuzumab`, etc.  
  These correspond to **well-established oncology drugs** with many successful Phase III programs.

- Some drugs/regimens receive strong negative weights (e.g., `ipilimumab`, `durvalumab`, `pembrolizumab`, certain regimens), often reflecting:
  - more ongoing/not-yet-completed Phase III trials, or  
  - historically difficult indications with lower success rates.

### 7.2 Visual: Top Features Plot

We generate a single, compact visual:

- Script: `src/model/plot_feature_importance.py`  
- Output: `feature_importance.png`  
- Content:
  - Horizontal bar chart of the **top 10 positive** and **top 10 negative** features (by coefficient magnitude).
  - Green bars: features that **increase** model-predicted Phase III success.
  - Red bars: features that **decrease** model-predicted Phase III success.

**Why this visual matters:**

- Shows that the model has learned **clinically sensible patterns** (e.g., well-known drugs and sponsor types).  
- Makes the model **interpretable** and supports the narrative that we are capturing real development-risk signals rather than noise.

---

## 8. Repository Structure

A minimal structure (omitting some files for brevity):

```bash
CS337/
├── README.md
├── data/
│   ├── clin_trials.csv                 # Raw clinical trials dataset from ClinicalTrials.gov
│   └── phase2_phase3_pairs.csv         # Cleaned, paired Phase2→Phase3 oncology dataset used for modeling
│
├── src/
│   ├── data/
│   │   ├── build_pairs.py              # Builds Phase 2 ↔ Phase 3 matched intervention dataset, labels success
│   │   ├── check_pairs.py              # Sanity checks for dataset integrity, nulls, structure
│   │   └── explore_features.py         # Exploratory data analysis (EDA) + summaries of interventions, conditions, etc.
│   │
│   ├── model/
│   │   ├── train_baseline.py           # Trains baseline logistic regression model on TF-IDF features
│   │   ├── feature_importance.py       # Extracts and saves model feature weights (drug signals)
│   │   └── __init__.py
│   │
│   └── visuals/
│       ├── plot_feature_importance.py  # Generates feature importance barplot from trained model
│       └── __init__.py
│
├── models/
│   └── baseline_model.joblib           # Saved trained logistic regression model
│
└── visuals/
    └── feature_importance.png          # Output figure of top positive/negative predictive drug terms
