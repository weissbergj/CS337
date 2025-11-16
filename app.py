import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load model
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()

# ===============================
# Title + Description
# ===============================
st.title("Phase III Success Predictor (Oncology)")

st.markdown("""
This demo predicts the probability that an oncology **Phase II intervention**  
will successfully complete **Phase III**, based on patterns learned from  
**5,071 historical Phase II → Phase III pairs**.
""")

# ===============================
# User Inputs
# ===============================
st.header("Enter Trial Information")

intervention = st.text_input(
    "Intervention Name",
    placeholder="e.g., nivolumab, capecitabine, trastuzumab"
)

org_class = st.selectbox(
    "Sponsor Type",
    [
        "INDUSTRY", "NIH", "NETWORK", "OTHER", "OTHER_GOV",
        "UNKNOWN", "FED", "INDIV"
    ]
)

# ===============================
# Predict
# ===============================
if st.button("Predict Phase III Success"):
    if intervention.strip() == "":
        st.error("Please enter an intervention name.")
    else:
        # Build input dataframe
        X_input = pd.DataFrame([{
            "Interventions_clean": intervention.lower().strip(),
            "Organization Class": org_class
        }])

        # Predict
        prob_success = model.predict_proba(X_input)[0][1]
        label = "Likely to succeed" if prob_success >= 0.5 else "Likely to fail"

        # Display results
        st.subheader("Prediction Results")
        st.write(f"**Predicted Probability of Success:** `{prob_success:.2f}`")
        
        if prob_success >= 0.5:
            st.success(label)
        else:
            st.error(label)

# ===============================
# Footer
# ===============================
st.markdown("""
---
**Note:**  
This model is a proof-of-concept trained on historical clinical trial metadata.  
It should **not** be used for real clinical decision-making.
""")
