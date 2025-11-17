"""
Load and process real historical trial data for the dashboard.
"""
import pandas as pd
import os


def load_historical_trials(data_path="data/phase2_phase3_pairs.csv"):
    """
    Load real historical Phase II → Phase III trial data.
    
    Returns:
        pd.DataFrame with processed trial data and outcome information
    """
    # Load the CSV data
    df = pd.read_csv(data_path)
    
    # Create a clean dataframe with relevant columns
    processed_df = pd.DataFrame({
        'trial_index': df['Unnamed: 0'],
        'organization_name': df['Organization Full Name'],
        'org_class': df['Organization Class'],
        'brief_title': df['Brief Title'],
        'conditions': df['Conditions'],
        'interventions': df['Interventions_clean'],
        'primary_purpose': df['Primary Purpose'],
        'primary_outcome': df['Outcome Measure'],
        'start_date': df['Start Date'],
        'phase2_status': df['Overall Status_ph2'],
        'phase3_status': df['Overall Status_ph3'],
        'actual_success': df['label_success'],
    })
    
    # Add a column indicating if final outcome is known
    # Outcome is known if label_success is 0 or 1 (not NaN) and phase3_status is meaningful
    processed_df['outcome_known'] = processed_df['actual_success'].notna()
    
    # Create a readable success label
    def format_outcome(row):
        if pd.isna(row['actual_success']):
            return 'Unknown'
        elif row['actual_success'] == 1:
            return '✅ Success'
        else:
            return '⚠️ Failure'
    
    processed_df['outcome_label'] = processed_df.apply(format_outcome, axis=1)
    
    # Fill NaN values for display
    processed_df['interventions'] = processed_df['interventions'].fillna('Unknown')
    processed_df['conditions'] = processed_df['conditions'].fillna('Unknown')
    processed_df['primary_outcome'] = processed_df['primary_outcome'].fillna('Unknown')
    processed_df['brief_title'] = processed_df['brief_title'].fillna('Unknown')
    processed_df['start_date'] = processed_df['start_date'].fillna('Unknown')
    processed_df['org_class'] = processed_df['org_class'].fillna('UNKNOWN')
    processed_df['primary_purpose'] = processed_df['primary_purpose'].fillna('Unknown')
    
    return processed_df
