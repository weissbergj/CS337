"""
Load and process trial data for the dashboard - both real historical data and predictions.
"""
import pandas as pd
import numpy as np
import joblib
import os

def load_real_data_with_predictions(model_path="model.joblib", data_path="data/phase2_phase3_pairs.csv"):
    """
    Generate n realistic mock Phase II trials with predicted Phase III success probabilities.
    
    Returns:
        pd.DataFrame with columns matching the calculator inputs plus predictions
    """
    np.random.seed(seed)
    
    # Realistic interventions (oncology drugs and combinations)
    interventions = [
        "Pembrolizumab", "Nivolumab", "Atezolizumab", "Durvalumab",
        "Ipilimumab + Nivolumab", "Carboplatin + Paclitaxel", 
        "Cisplatin + Gemcitabine", "FOLFOX", "FOLFIRI",
        "Bevacizumab + Paclitaxel", "Trastuzumab + Pertuzumab",
        "Osimertinib", "Erlotinib", "Gefitinib", "Crizotinib",
        "Dabrafenib + Trametinib", "Vemurafenib", "Encorafenib + Binimetinib",
        "Olaparib", "Rucaparib", "Niraparib", "Talazoparib",
        "Lenvatinib", "Sorafenib", "Regorafenib", "Cabozantinib",
        "Palbociclib", "Ribociclib", "Abemaciclib",
        "Venetoclax", "Ibrutinib", "Acalabrutinib",
        "CAR-T Cell Therapy", "Blinatumomab", "Inotuzumab",
        "Enasidenib", "Ivosidenib", "Midostaurin",
        "Tucatinib + Trastuzumab", "Lapatinib + Capecitabine",
        "Abiraterone + Prednisone", "Enzalutamide", "Darolutamide",
        "Temozolomide + Radiation", "Irinotecan + Temozolomide",
        "Docetaxel", "Paclitaxel", "Nab-paclitaxel",
        "Gemcitabine + Cisplatin", "Pemetrexed + Cisplatin"
    ]
    
    # Cancer types
    cancer_types = [
        "Non-Small Cell Lung Cancer", "Small Cell Lung Cancer",
        "Metastatic Melanoma", "Advanced Melanoma",
        "Metastatic Breast Cancer", "Triple-Negative Breast Cancer", "HER2+ Breast Cancer",
        "Metastatic Colorectal Cancer", "Advanced Colorectal Cancer",
        "Renal Cell Carcinoma", "Advanced Renal Cell Carcinoma",
        "Hepatocellular Carcinoma", "Advanced Hepatocellular Carcinoma",
        "Pancreatic Adenocarcinoma", "Metastatic Pancreatic Cancer",
        "Ovarian Cancer", "Platinum-Resistant Ovarian Cancer",
        "Prostate Cancer", "Metastatic Castration-Resistant Prostate Cancer",
        "Glioblastoma Multiforme", "Recurrent Glioblastoma",
        "Acute Myeloid Leukemia", "Relapsed/Refractory AML",
        "Chronic Lymphocytic Leukemia", "Relapsed CLL",
        "Multiple Myeloma", "Relapsed/Refractory Multiple Myeloma",
        "Gastric Cancer", "Gastroesophageal Junction Cancer",
        "Bladder Cancer", "Urothelial Carcinoma",
        "Head and Neck Squamous Cell Carcinoma",
        "Esophageal Cancer", "Thyroid Cancer"
    ]
    
    # Trial title templates
    title_templates = [
        "A Phase II Study of {intervention} in {condition}",
        "Phase II Trial Evaluating {intervention} for {condition}",
        "Efficacy and Safety of {intervention} in Patients with {condition}",
        "A Multicenter Phase II Study of {intervention} in {condition}",
        "Phase II Open-Label Study of {intervention} in Advanced {condition}",
        "Randomized Phase II Trial of {intervention} versus Standard Care in {condition}",
        "Single-Arm Phase II Study of {intervention} for {condition}",
        "Phase II Investigation of {intervention} in Previously Treated {condition}",
    ]
    
    # Outcome measure templates
    outcome_templates = [
        "Overall response rate at {months} months",
        "Progression-free survival at {months} months",
        "Objective response rate by RECIST 1.1 criteria",
        "Disease control rate at {months} months",
        "Complete response rate at {months} months",
        "Duration of response assessed up to {months} months",
        "Time to progression assessed up to {months} months",
        "Overall survival at {months} months",
        "Pathologic complete response rate",
        "6-month progression-free survival rate",
    ]
    
    org_classes = ["INDUSTRY", "NIH", "NETWORK", "OTHER", "OTHER_GOV", "FED"]
    primary_purposes = ["TREATMENT", "PREVENTION", "SUPPORTIVE_CARE", "DIAGNOSTIC"]
    
    # Generate trials
    trials = []
    for i in range(n):
        intervention = np.random.choice(interventions)
        condition = np.random.choice(cancer_types)
        
        # Create title
        template = np.random.choice(title_templates)
        brief_title = template.format(
            intervention=intervention,
            condition=condition
        )
        
        # Create outcome
        outcome_template = np.random.choice(outcome_templates)
        months = np.random.choice([3, 6, 9, 12, 18, 24])
        outcome = outcome_template.format(months=months)
        
        org_class = np.random.choice(org_classes, p=[0.6, 0.15, 0.1, 0.08, 0.05, 0.02])
        primary_purpose = np.random.choice(primary_purposes, p=[0.85, 0.05, 0.05, 0.05])
        
        # Generate realistic probability
        # Industry trials tend to have slightly higher success rates
        # Immunotherapy tends to have higher success
        base_prob = 0.35
        
        if org_class == "INDUSTRY":
            base_prob += 0.10
        elif org_class == "NIH":
            base_prob += 0.05
            
        if any(drug in intervention for drug in ["Pembrolizumab", "Nivolumab", "Atezolizumab", "Durvalumab"]):
            base_prob += 0.15
            
        if "Combination" in brief_title or "+" in intervention:
            base_prob += 0.05
            
        # Add random variation
        prob_success = np.clip(base_prob + np.random.normal(0, 0.15), 0.05, 0.95)
        
        # Determine predicted label (threshold at 0.5)
        predicted_success = 1 if prob_success >= 0.5 else 0
        
        trial = {
            "trial_id": f"NCT{np.random.randint(10000000, 99999999):08d}",
            "intervention": intervention,
            "brief_title": brief_title,
            "conditions": condition,
            "primary_outcome": outcome,
            "org_class": org_class,
            "primary_purpose": primary_purpose,
            "predicted_probability": prob_success,
            "predicted_success": predicted_success,
        }
        trials.append(trial)
    
    df = pd.DataFrame(trials)
    
    # Sort by probability descending for better initial view
    df = df.sort_values("predicted_probability", ascending=False).reset_index(drop=True)
    
    return df
