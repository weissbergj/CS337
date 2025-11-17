import streamlit as st
import pandas as pd
import joblib
from src.app.dashboard import render_dashboard

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Phase III Success Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# Navigation
# ===============================
page = st.sidebar.radio(
    "Navigation",
    ["🧪 Calculator", "📊 Historical Insights"],
    index=0
)

# ===============================
# Route to appropriate page
# ===============================
if page == "📊 Historical Insights":
    render_dashboard()
    st.stop()

# ===============================
# Load model
# ===============================
@st.cache_resource
def load_model():
    # This must match the filename you saved from train.py
    return joblib.load("model.joblib")

model = load_model()

# ===============================
# Title + Description
# ===============================
st.title("Phase III Success Predictor (Oncology)")

st.markdown("""
This demo predicts the probability that an oncology **Phase II trial**  
will successfully complete **Phase III**, based on patterns learned from  
**5,071 historical Phase II → Phase III pairs** derived from ClinicalTrials.gov metadata.
""")

# User Inputs
st.header("Enter Phase II Trial Information")

intervention = st.text_input(
    "Intervention(s) / Drug(s)",
    placeholder="e.g., nivolumab, capecitabine + oxaliplatin"
)

brief_title = st.text_input(
    "Brief trial title",
    placeholder="e.g., A Phase II Study of Nivolumab in Metastatic NSCLC"
)

conditions = st.text_input(
    "Cancer type / condition(s)",
    placeholder="e.g., metastatic non-small cell lung cancer"
)

outcome = st.text_area(
    "Primary outcome summary",
    placeholder="e.g., Overall response rate at 6 months."
)

org_class = st.selectbox(
    "Sponsor type (Organization Class)",
    [
        "INDUSTRY",
        "NIH",
        "NETWORK",
        "OTHER",
        "OTHER_GOV",
        "UNKNOWN",
        "FED",
        "INDIV",
    ],
    index=0,
)

primary_purpose = st.selectbox(
    "Primary Purpose",
    [
        "TREATMENT",
        "PREVENTION",
        "DIAGNOSTIC",
        "HEALTH_SERVICES_RESEARCH",
        "SCREENING",
        "SUPPORTIVE_CARE",
        "BASIC_SCIENCE",
        "OTHER",
        "ECT",
        "Unknown",
    ],
    index=0,
)

# Predict
if st.button("Predict Phase III Success"):
    # Basic validation
    if intervention.strip() == "":
        st.error("Please enter at least an intervention/drug name.")
    else:
        # Build combined_text exactly like in training:
        # combined_text = Interventions_clean + Brief Title + Conditions + Outcome Measure
        combined_text = " ".join([
            intervention.strip(),
            brief_title.strip(),
            conditions.strip(),
            outcome.strip(),
        ])

        # Build input DataFrame with the exact columns the model expects
        X_input = pd.DataFrame([{
            "combined_text": combined_text,
            "Organization Class": org_class,
            "Primary Purpose": primary_purpose,
        }])

        # Run model
        prob_success = model.predict_proba(X_input)[0, 1]
        pred_label = model.predict(X_input)[0]

        st.subheader("Prediction Results")
        st.write(f"**Predicted Probability of Phase III Success:** `{prob_success:.2%}`")

        if pred_label == 1:
            st.success("Model prediction: **Likely to succeed ✅**")
        else:
            st.error("Model prediction: **Likely NOT to succeed ⚠️**")

        st.caption(
            "Note: This is a proof-of-concept model using historical metadata only. "
            "It is **not** intended for clinical or investment decision-making."
        )
