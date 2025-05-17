import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load scaler
scaler = joblib.load("models/scaler.pkl")

def preprocess_input(input_df):
    df = pd.get_dummies(input_df, columns=['gender', 'smoking_status', 'physical_activity'], drop_first=True)
    
    # Ensure all required columns exist
    expected_cols = ['age', 'bmi', 'blood_pressure', 'glucose_level', 'cholesterol',
                     'diabetes_history', 'heart_disease',
                     'gender_Male', 'gender_Other',
                     'smoking_status_Former', 'smoking_status_Never',
                     'physical_activity_Low', 'physical_activity_Moderate']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_cols]
    df_scaled = scaler.transform(df)
    return df_scaled
