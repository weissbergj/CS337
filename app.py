import streamlit as st
import pandas as pd
import joblib

# ===============================
# Blue Tab Bar Styling
# ===============================
st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            background-color: #002b80;
            border-radius: 6px;
            padding: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            color: white;
            font-weight: 700;
            font-size: 18px;
            padding: 10px 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0041cc !important;
            border-radius: 4px;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()

# ===============================
# Tabs
# ===============================
tab1, tab2 = st.tabs(["Predictor", "About Us"])

# ===============================
# TAB 1 — MAIN APP
# ===============================
with tab1:

    # Title + Description
    st.title("Phase III Success Predictor (Oncology)")

    st.markdown("""
    This demo predicts the probability that an oncology **Phase II trial**  
    will successfully complete **Phase III**, based on patterns learned from  
    **5,071 historical Phase II → Phase III pairs** derived from ClinicalTrials.gov metadata.
    """)

    # ===============================
    # User Inputs
    # ===============================
    st.header("Enter Phase II Trial Information")

    intervention = st.text_input(
        "Intervention(s) / Drug(s)",
        placeholder="e.g., Nivolumab, Capecitabine + Oxaliplatin"
    )

    brief_title = st.text_input(
        "Brief Trial Title",
        placeholder="e.g., A Phase II Study of Nivolumab in Metastatic NSCLC"
    )

    conditions = st.text_input(
        "Cancer Type / Condition(s)",
        placeholder="e.g., Metastatic Non-Small Cell Lung Cancer"
    )

    outcome = st.text_area(
        "Primary Outcome Summary",
        placeholder="e.g., Overall Response Rate at 6 months."
    )

    org_class = st.selectbox(
        "Sponsor Type (Organization Class)",
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

    # ===============================
    # Predict
    # ===============================
    if st.button("Predict Phase III Success"):

        if intervention.strip() == "":
            st.error("Please enter at least an intervention/drug name.")
        else:
            combined_text = " ".join([
                intervention.strip(),
                brief_title.strip(),
                conditions.strip(),
                outcome.strip(),
            ])

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

# ===============================
# TAB 2 — ABOUT US
# ===============================
with tab2:

    st.markdown("<h1 style='text-align: center;'>About Us</h1>", unsafe_allow_html=True)

    st.markdown("**Team Name: PhaseForward** (Charles Chen, Chelsea Hu, Meghana Paturu, and Jared Weissberg)")

    st.write("**Course:** Stanford CS337 – AI for Healthcare")

    st.markdown("**Project Abstract:**")

    st.write("""
    Our goal for this project is to build a quantitative tool that predicts the probability that a new drug candidate will succeed in Phase III trials and receive FDA approval. Typically, healthcare investors rely on subjective expert opinions and broad historical averages to guide their decisions, but these methods provide limited drug-specific insight because they ignore the candidate’s actual Phase II efficacy and safety results. Our approach is to collect past and ongoing clinical trial data, including Phase II efficacy results, safety outcomes, endpoints, and trial design details, to train a predictive model that estimates the drug-level likelihood of success. An example workflow of using our tool is that a user could input a candidate drug, and our system will compare its early-phase profile to similar historical cases to generate a dashboard filled with evidence-based probability scores, confidence intervals, and similarity metrics. In the past few weeks, our group has identified primary data sources, outlined scraping methods to fill gaps in publicly available information, and defined the core features we need to start building up our consolidated dataset for developing""")
