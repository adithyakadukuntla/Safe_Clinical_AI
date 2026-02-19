import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

def preprocess_data(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # Target column
    target = 'hospital_death'
    
    # Drop IDs and obvious leakage columns
    drop_cols = ['encounter_id', 'patient_id', 'hospital_id', 'icu_id', 
                 'apache_4a_icu_death_prob', 'apache_4a_hospital_death_prob']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Extract Categorical and Numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(exclude=['object']).columns.tolist()
    numerical_cols.remove(target)
    
    print(f"Categorical columns: {categorical_cols}")
    
    # --- Advanced Preprocessing ---
    
    # 1. Handle Categorical Values with OneHotEncoding
    # Using handle_unknown='ignore' for robust production inference
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = ohe.fit_transform(df[categorical_cols].fillna('Missing'))
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_cats, columns=cat_feature_names)
    
    # 2. Handle Numerical Values with Median Imputation
    imputer = SimpleImputer(strategy='median')
    imputed_nums = imputer.fit_transform(df[numerical_cols])
    imputed_df = pd.DataFrame(imputed_nums, columns=numerical_cols)
    
    # 3. Combine Processed Features
    X = pd.concat([imputed_df, encoded_df], axis=1)
    y = df[target]
    
    # 4. Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Save artifacts for inference
    os.makedirs('models', exist_ok=True)
    joblib.dump(ohe, 'models/encoder.joblib')
    joblib.dump(imputer, 'models/imputer.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(X.columns.tolist(), 'models/feature_names.joblib')
    
    return X_scaled, y, X.columns.tolist()

def get_top_18_features():
    """
    Returns the top 18 features most relevant for the ICU mortality prediction
    based on clinical standards and WiDS 2020 importance metrics.
    """
    # Mapping for Frontend Display
    features_info = {
        'age': {'name': 'Age', 'info': 'Patient age in years.'},
        'bmi': {'name': 'BMI', 'info': 'Body Mass Index.'},
        'gcs_eyes_apache': {'name': 'GCS Eyes', 'info': 'Glasgow Coma Scale Eyes component.'},
        'gcs_motor_apache': {'name': 'GCS Motor', 'info': 'Glasgow Coma Scale Motor component.'},
        'gcs_verbal_apache': {'name': 'GCS Verbal', 'info': 'Glasgow Coma Scale Verbal component.'},
        'heart_rate_apache': {'name': 'Heart Rate', 'info': 'Peak heart rate during the first 24 hours.'},
        'resprate_apache': {'name': 'Respiratory Rate', 'info': 'Peak respiratory rate during the first 24 hours.'},
        'temp_apache': {'name': 'Temperature', 'info': 'Peak temperature in Celsius.'},
        'map_apache': {'name': 'Mean Arterial Pressure', 'info': 'Lowest mean arterial pressure.'},
        'bun_apache': {'name': 'BUN', 'info': 'Blood Urea Nitrogen level.'},
        'creatinine_apache': {'name': 'Creatinine', 'info': 'Serium creatinine level.'},
        'glucose_apache': {'name': 'Glucose', 'info': 'Highest glucose level.'},
        'hematocrit_apache': {'name': 'Hematocrit', 'info': 'Lowest hematocrit level.'},
        'wbc_apache': {'name': 'WBC Count', 'info': 'Highest white blood cell count.'},
        'sodium_apache': {'name': 'Sodium', 'info': 'Highest sodium level.'},
        'potassium_apache': {'name': 'Potassium', 'info': 'Highest potassium level.'},
        'bilirubin_apache': {'name': 'Bilirubin', 'info': 'Highest bilirubin level.'},
        'albumin_apache': {'name': 'Albumin', 'info': 'Lowest albumin level.'}
    }
    return features_info

if __name__ == "__main__":
    X, y, features = preprocess_data('data/DataSet.csv')
    print(f"Data preprocessed. Final shape: {X.shape}")
