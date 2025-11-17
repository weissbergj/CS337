# BioDiligence: Predicting Drug Development Success

**CS337 AI for Healthcare | Team PhaseForward**

## What We're Building

This project tackles a real problem in drug development: most Phase III clinical trials fail, but there's no good way to predict which ones will succeed before investing millions of dollars and years of work. We're building a tool that uses historical data from ClinicalTrials.gov to predict whether a Phase II oncology trial is likely to successfully complete Phase III.

## The Big Question

Can we use early-stage trial data (drug names, cancer types, sponsor info, outcomes) to predict if that drug will make it through Phase III? Turns out, yes—at least somewhat.

## The Data

We started with a massive dataset from ClinicalTrials.gov (via Kaggle)—over 496,000 clinical trials across all diseases. We filtered down to oncology trials only (anything tagged with "Neoplasm" in MeSH terms) and pulled out all Phase II and Phase III studies.

The tricky part was matching them up. We paired Phase II and Phase III trials that tested the same intervention (after cleaning up drug names to handle typos and formatting). If the Phase III trial's status was "COMPLETED", we labeled that as success. Otherwise, it's a failure.

We ended up with **5,071 matched pairs**—basically a 50/50 split between successes and failures, which is actually pretty close to real-world rates.

## What Goes Into the Model

We're working with six features from the Phase II trial:

**Text features** (we use TF-IDF to turn these into numbers):
- Drug/intervention names
- Trial title
- Cancer conditions being treated
- Primary outcome measures

**Categorical features**:
- Who's sponsoring it (industry, NIH, academic, etc.)
- Trial purpose (treatment, prevention, diagnostic, etc.)

We concatenate all the text fields together, run TF-IDF with a 5,000 feature limit, and one-hot encode the categorical stuff.

## The Model

Nothing fancy here—just logistic regression. We tried a few things, but honestly a simple model worked best for this dataset. We use:
- 80/20 train/test split (stratified so we keep the 50/50 balance)
- Balanced class weights (since we care about both successes and failures equally)
- 500 max iterations to make sure it converges

The model gets saved to `model.joblib` and you can retrain it by running `src/model/train.py`.

## How Well Does It Work?

On the test set:
- **Accuracy: 73%** (better than random!)
- **ROC-AUC: 0.79** (pretty decent for predicting something this uncertain)

It's not perfect, but it's way better than guessing. The model picks up on patterns like certain drug classes that tend to fail, or specific cancer types that are harder to treat successfully.

## What's in This Repo

Here's how everything's organized:

```
CS337/
├── app.py                              # Main Streamlit app with calculator & dashboard
├── model.joblib                        # Our trained model
├── requirements.txt                    # Python dependencies
│
├── data/
│   └── phase2_phase3_pairs.csv        # 5,071 matched Phase II→III pairs
│
└── src/
    ├── app/
    │   ├── dashboard.py               # Historical insights dashboard
    │   ├── data_loader.py             # Data loading utilities
    │   └── mock_data.py               # Sample data for demo
    │
    ├── data/
    │   ├── build_pairs.py             # Script to create Phase II→III matches
    │   └── explore_features.py        # Data exploration notebooks
    │
    ├── model/
    │   ├── train.py                   # Model training pipeline
    │   └── feature_importance.py      # Extract important features
    │
    └── visuals/
        └── plot_feature_importance.py # Generate visualizations
```

## Try It Out

We built a Streamlit app with two main features:

### 1. Success Predictor
Enter details about a Phase II trial (drug name, cancer type, sponsor, outcomes) and get an instant probability estimate for Phase III success. The interface is simple—just fill in the fields and hit predict.

### 2. Historical Insights Dashboard
This is where things get interesting. We built an interactive dashboard with 10 different analysis views to explore what actually drives success rates in oncology trials:

- **📊 Overview** - High-level stats on trial counts, success rates, and basic distribution patterns
- **🏢 Sponsor Analysis** - Compare success rates across different sponsor types (industry vs academic vs government)
- **💊 Top Interventions** - Which drugs and drug combinations have the highest success rates? See the top performers and biggest failures
- **📈 Advanced Analytics** - Dig into model performance with ROC curves, feature importance, and prediction distributions
- **🎯 Cancer Type Deep Dive** - Success rates vary wildly by cancer type. See which cancers are easier vs harder to treat successfully
- **🔬 Intervention Patterns** - Analyze patterns in drug naming, combination therapies, and treatment approaches
- **📅 Temporal Trends** - How have success rates changed over time? Are we getting better at predicting winners?
- **📋 Trial Status Analysis** - Break down trials by their current status (completed, terminated, withdrawn, etc.)
- **🌐 Organization Insights** - Geographic patterns and institutional success rates
- **🔗 Correlation Matrix** - Explore relationships between different features and success rates

Each tab has interactive visualizations built with Plotly, so you can hover, zoom, and explore the data yourself.

### Running It

The app is deployed on Streamlit Cloud (link in repo) or you can run it locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Team

**PhaseForward:** Charles Chen, Chelsea Hu, Meghana Paturu, and Jared Weissberg

Built for Stanford CS337 – AI for Healthcare, Fall 2024
