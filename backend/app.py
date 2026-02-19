from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Load artifacts and training statistics
try:
    encoder = joblib.load('models/encoder.joblib')
    imputer = joblib.load('models/imputer.joblib')
    scaler = joblib.load('models/scaler.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    
    # Load ensemble
    model_xgb = joblib.load('models/xgb_model.joblib')
    model_lgb = joblib.load('models/lgb_model.joblib')
    model_cat = joblib.load('models/cat_model.joblib')
    
    # Load original dataset to get median values for missing features
    print("Loading training statistics...")
    df_train = pd.read_csv('data/DataSet.csv')
    
    # Drop ID columns
    drop_cols = ['encounter_id', 'patient_id', 'hospital_id', 'icu_id', 
                 'apache_4a_icu_death_prob', 'apache_4a_hospital_death_prob', 'hospital_death']
    df_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    
    # Get categorical and numerical columns
    categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df_train.select_dtypes(exclude=['object']).columns.tolist()
    
    # Calculate median values for all numerical features
    median_values = df_train[numerical_cols].median().to_dict()
    
    # Get mode for categorical features
    mode_values = {col: df_train[col].mode()[0] if len(df_train[col].mode()) > 0 else 'Missing' 
                   for col in categorical_cols}
    
    print(f"Loaded {len(median_values)} numerical medians and {len(mode_values)} categorical modes")
    
except Exception as e:
    print(f"Warning: Models not found. Error: {e}")



# Initialize SHAP explainer
try:
    import shap
    print("Initializing SHAP explainer...")
    # Initialize with the XGBoost model
    explainer = shap.TreeExplainer(model_xgb)
except Exception as e:
    print(f"Warning: Could not initialize SHAP explainer. Error: {e}")
    explainer = None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Create a full feature DataFrame with all original columns
        full_input = pd.DataFrame(columns=numerical_cols + categorical_cols)
        
        # Fill with user input (only the 18 features they provided)
        for key, value in data.items():
            if key in full_input.columns:
                full_input.loc[0, key] = value
        
        # Fill missing numerical features with training medians
        for col in numerical_cols:
            if pd.isna(full_input.loc[0, col]):
                full_input.loc[0, col] = median_values.get(col, 0)
        
        # Fill missing categorical features with training modes
        for col in categorical_cols:
            if pd.isna(full_input.loc[0, col]) or full_input.loc[0, col] == '':
                full_input.loc[0, col] = mode_values.get(col, 'Missing')
        
        # Now apply the SAME preprocessing pipeline as training
        # 1. OneHotEncode categorical
        encoded_cats = encoder.transform(full_input[categorical_cols].fillna('Missing'))
        cat_feature_names = encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_cats, columns=cat_feature_names)
        
        # 2. Impute numerical (should be no missing now, but for safety)
        imputed_nums = imputer.transform(full_input[numerical_cols])
        imputed_df = pd.DataFrame(imputed_nums, columns=numerical_cols)
        
        # 3. Combine
        X = pd.concat([imputed_df, encoded_df], axis=1)
        
        # 4. Scale
        X_scaled = scaler.transform(X)
        
        # Ensemble Prediction (Hybrid Gradient Ensemble)
        prob_xgb = model_xgb.predict_proba(X_scaled)[:, 1][0]
        prob_lgb = model_lgb.predict_proba(X_scaled)[:, 1][0]
        prob_cat = model_cat.predict_proba(X_scaled)[:, 1][0]
        
        # Weighted fusion based on training performance
        try:
            ensemble_weights = joblib.load('models/ensemble_weights.joblib')
        except FileNotFoundError:
            # Fallback to equal weights if ensemble_weights not found
            print("Warning: ensemble_weights.joblib not found, using equal weights")
            ensemble_weights = {'xgb': 0.333, 'lgb': 0.333, 'cat': 0.334}
        
        final_prob = (
            ensemble_weights['xgb'] * prob_xgb +
            ensemble_weights['lgb'] * prob_lgb +
            ensemble_weights['cat'] * prob_cat
        )
        
        # Generate SHAP Force Plot
        shap_html = None
        if explainer:
            try:
                print("Calculating SHAP values...")
                # Calculate SHAP values for the instance
                shap_values = explainer.shap_values(X_scaled)
                print(f"SHAP values calculated. Shape: {shap_values.shape if hasattr(shap_values, 'shape') else len(shap_values)}")
                
                # Check formatting of shap_values (list or array)
                # XGBoost binary usually returns a single array, but safer to handle list
                if isinstance(shap_values, list):
                    # For binary, index 1 is usually the positive class
                    sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                else:
                    sv = shap_values
                
                # Ensure 1D array for single prediction
                if len(sv.shape) > 1:
                    sv = sv[0]

                print("Generating SHAP force plot...")
                # Generate force plot
                # We use the unscaled X for display values (features argument)
                # Use X (pandas DataFrame) to show feature names and original values
                p = shap.force_plot(
                    explainer.expected_value, 
                    sv,
                    X.iloc[0,:],
                    matplotlib=False,
                    show=False,
                    link="logit" 
                )
                
                # Save to HTML string
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode='w', encoding='utf-8') as f:
                    shap.save_html(f.name, p)
                    tmp_name = f.name
                
                with open(tmp_name, "r", encoding="utf-8") as f:
                    shap_html = f.read()
                
                print(f"SHAP HTML generated. Length: {len(shap_html)}")
                os.remove(tmp_name)
            except Exception as shap_e:
                print(f"Error generating SHAP plot: {shap_e}")
                import traceback
                traceback.print_exc()
        else:
            print("SHAP explainer is None, skipping plot generation.")
        
        return jsonify({
            'mortality_risk': float(final_prob),
            'status': 'Deceased' if final_prob > 0.5 else 'Survived',
            'confidence': {
                'xgboost': float(prob_xgb),
                'lightgbm': float(prob_lgb),
                'catboost': float(prob_cat)
            },
            'ensemble_weights': ensemble_weights,
            'shap_html': shap_html
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in prediction: {error_trace}")
        return jsonify({'error': str(e), 'trace': error_trace}), 400

@app.route('/features', methods=['GET'])
def get_features():
    from preprocessing import get_top_18_features
    return jsonify(get_top_18_features())

import shutil
import requests
import time
from train_model import train_and_evaluate

# --- Feedback & Retraining Endpoints ---

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    try:
        # data should contain: features (dict), actual_label (int), predicted_label (int)
        # We need to map features back to CSV columns
        # For simplicity, we assume 'features' keys match CSV columns
        
        feedback_file = 'data/feedback_data.csv'
        
        # Check if file exists to write header
        file_exists = os.path.isfile(feedback_file)
        
        # Prepare row
        row = data.get('features', {}).copy()
        row['hospital_death'] = data.get('actual_label')
        row['timestamp'] = pd.Timestamp.now()
        
        # Append to CSV
        df_feedback = pd.DataFrame([row])
        
        # Align columns with DataSet.csv if possible, mostly for consistency
        # For now, just append what we have. Retraining logic will handle valid columns.
        df_feedback.to_csv(feedback_file, mode='a', header=not file_exists, index=False)
        
        return jsonify({'message': 'Feedback received', 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/retrain/dry-run', methods=['POST'])
def retrain_dry_run():
    try:
        job_id = request.json.get('jobId')
        
        # 1. Load Data
        print("Loading original and feedback data...")
        df_orig = pd.read_csv('data/DataSet.csv')
        
        if os.path.exists('data/feedback_data.csv'):
            df_feedback = pd.read_csv('data/feedback_data.csv')
            # Ensure columns match
            common_cols = list(set(df_orig.columns) & set(df_feedback.columns))
            df_combined = pd.concat([df_orig[common_cols], df_feedback[common_cols]])
        else:
            df_combined = df_orig
            
        # Save temp combined file
        temp_data_path = 'data/temp_combined.csv'
        df_combined.to_csv(temp_data_path, index=False)
        
        # 2. Train Candidate Model
        candidate_dir = 'models_candidate'
        if os.path.exists(candidate_dir):
            shutil.rmtree(candidate_dir)
        
        print("Starting candidate training...")
        # Now returns a dict with 'individual_models', 'ensembles', 'weights'
        results = train_and_evaluate(temp_data_path, output_dir=candidate_dir)
        
        # New metrics for the "Optimized Ensemble"
        new_metrics = results['Phase2_Constrained']['metrics']['Ensemble_Weighted']
        
        # Notify Node.js Backend about completion
        if job_id:
            node_backend_url = f"http://localhost:3000/api/admin/update-job/{job_id}"
            requests.post(node_backend_url, json={
                'status': 'completed',
                'metrics_new': {
                    'accuracy': new_metrics['Accuracy'],
                    'f1_score': new_metrics['F1-Score'],
                    'roc_auc': new_metrics['ROC-AUC']
                },
                'detailed_results': results,
                'log_message': "Dry run training completed with Weight Optimization."
            })
        
        return jsonify({
            'message': 'Dry run complete', 
            'metrics_new': new_metrics,
            'full_results': results
        })
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(trace)
        if job_id:
             try:
                requests.post(f"http://localhost:3000/api/admin/update-job/{job_id}", json={
                    'status': 'failed',
                    'log_message': f"Training failed: {str(e)}"
                })
             except:
                 pass
        return jsonify({'error': str(e), 'trace': trace}), 500

@app.route('/retrain/approve', methods=['POST'])
def approve_retrain():
    try:
        # Backup current models
        backup_dir = f'models_backup_{int(time.time())}'
        if os.path.exists('models'):
            shutil.copytree('models', backup_dir)
            
        # Move candidate to models
        if os.path.exists('models_candidate'):
            # shutil.move cannot overwrite directories easily on Windows
            if os.path.exists('models'):
                shutil.rmtree('models')
            shutil.move('models_candidate', 'models')
            
            # Restart/Reload models in memory?
            # Ideally, we should reload the models global variables here.
            # For simplicity, we tell the admin to restart the server or we implement a reload function.
            # Let's try to reload.
            global model_xgb, model_lgb, model_cat, encoder, imputer, scaler, feature_names
            model_xgb = joblib.load('models/xgb_model.joblib')
            model_lgb = joblib.load('models/lgb_model.joblib')
            model_cat = joblib.load('models/cat_model.joblib')
            encoder = joblib.load('models/encoder.joblib')
            imputer = joblib.load('models/imputer.joblib')
            scaler = joblib.load('models/scaler.joblib')
            feature_names = joblib.load('models/feature_names.joblib')
            
            return jsonify({'message': 'Model approved and deployed successfully. Backup created.'})
        else:
            return jsonify({'error': 'No candidate model found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain/reject', methods=['POST'])
def reject_retrain():
    try:
        if os.path.exists('models_candidate'):
            shutil.rmtree('models_candidate')
        return jsonify({'message': 'Candidate model rejected and deleted.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)

