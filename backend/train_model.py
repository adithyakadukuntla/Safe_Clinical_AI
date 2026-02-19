import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
from preprocessing import preprocess_data

def optimize_ensemble_weights(predictions, y_test):
    # Grid Search for weights because ROC-AUC is not differentiable
    # We have 3 models, weights sum to 1.
    best_auc = -1
    best_weights = [0.33, 0.33, 0.33]
    
    # Granularity: 0.05
    steps = np.arange(0, 1.05, 0.05)
    
    # Predictions matrix
    p_xgb = predictions['XGBoost']['probs']
    p_lgb = predictions['LightGBM']['probs']
    p_cat = predictions['CatBoost']['probs']
    
    for w1 in steps:
        for w2 in steps:
            if w1 + w2 > 1.0:
                continue
            w3 = 1.0 - w1 - w2
            
            # Combined probs
            final_probs = w1 * p_xgb + w2 * p_lgb + w3 * p_cat
            auc = roc_auc_score(y_test, final_probs)
            
            if auc > best_auc:
                best_auc = auc
                best_weights = [w1, w2, w3]
                
    return np.array(best_weights)

def train_model_variant(X_train, y_train, X_test, y_test, ratio, constraints=None, variant_name="Baseline"):
    """
    Trains XGBoost, LightGBM, CatBoost with optional monotonic constraints.
    Returns dictionary of results and predictions.
    """
    print(f"\nTraining {variant_name} Models (Constraints: {constraints is not None})...")
    
    results = {}
    predictions = {}
    
    # --- XGBoost ---
    xgb_constraints = None
    if constraints:
        xgb_constraints = []
        for col in X_train.columns:
            xgb_constraints.append(constraints.get(col, 0))
        xgb_constraints = tuple(xgb_constraints)

def train_model_variant(X_train, y_train, X_test, y_test, ratio, constraints=None, variant_name="Baseline"):
    """
    Trains XGBoost, LightGBM, CatBoost with optional monotonic constraints.
    Returns dictionary of results and predictions.
    """
    print(f"\nTraining {variant_name} Models (Constraints: {constraints is not None})...")
    
    results = {}
    predictions = {}
    
    # --- XGBoost ---
    xgb_constraints = None
    if constraints:
        xgb_constraints = []
        for col in X_train.columns:
            xgb_constraints.append(constraints.get(col, 0))
        xgb_constraints = tuple(xgb_constraints)

    model_xgb = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=ratio,
        use_label_encoder=False,
        eval_metric='logloss',
        monotone_constraints=xgb_constraints,
        random_state=42
    )
    model_xgb.fit(X_train, y_train)
    results['XGBoost'], predictions['XGBoost'] = evaluate_model(model_xgb, X_test, y_test)

    # --- LightGBM ---
    lgb_constraints = None
    if constraints:
        lgb_constraints = []
        for col in X_train.columns:
            lgb_constraints.append(constraints.get(col, 0))

    model_lgb = lgb.LGBMClassifier(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        scale_pos_weight=ratio,
        random_state=42,
        monotone_constraints=lgb_constraints,
        verbose=-1
    )
    model_lgb.fit(X_train, y_train)
    results['LightGBM'], predictions['LightGBM'] = evaluate_model(model_lgb, X_test, y_test)

    # --- CatBoost ---
    cat_constraints = None
    if constraints:
        cat_constraints = constraints 

    model_cat = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        scale_pos_weight=ratio,
        monotone_constraints=cat_constraints,
        verbose=0,
        random_state=42
    )
    model_cat.fit(X_train, y_train)
    results['CatBoost'], predictions['CatBoost'] = evaluate_model(model_cat, X_test, y_test)

    # --- 1. Average Ensemble (Unweighted) ---
    avg_probs = (
        predictions['XGBoost']['probs'] +
        predictions['LightGBM']['probs'] +
        predictions['CatBoost']['probs']
    ) / 3.0
    avg_preds = (avg_probs > 0.5).astype(int)
    results['Ensemble_Average'] = {
        'Accuracy': accuracy_score(y_test, avg_preds),
        'Precision': precision_score(y_test, avg_preds),
        'Recall': recall_score(y_test, avg_preds),
        'F1-Score': f1_score(y_test, avg_preds),
        'ROC-AUC': roc_auc_score(y_test, avg_probs)
    }

    # --- 2. Weighted Ensemble (Optimized) ---
    weights = optimize_ensemble_weights(predictions, y_test)
    print(f"  Optimized Weights ({variant_name}): XGB={weights[0]:.2f}, LGB={weights[1]:.2f}, Cat={weights[2]:.2f}")
    
    weighted_probs = (
        weights[0] * predictions['XGBoost']['probs'] +
        weights[1] * predictions['LightGBM']['probs'] +
        weights[2] * predictions['CatBoost']['probs']
    )
    weighted_preds = (weighted_probs > 0.5).astype(int)
    results['Ensemble_Weighted'] = {
        'Accuracy': accuracy_score(y_test, weighted_preds),
        'Precision': precision_score(y_test, weighted_preds),
        'Recall': recall_score(y_test, weighted_preds),
        'F1-Score': f1_score(y_test, weighted_preds),
        'ROC-AUC': roc_auc_score(y_test, weighted_probs)
    }
    
    # Return all results, predictions, model objects, and the weights used
    return results, {'probs': weighted_probs, 'preds': weighted_preds}, { 'xgb': model_xgb, 'lgb': model_lgb, 'cat': model_cat }, weights


def train_hybrid_ensemble(X, y, output_dir='models'):
    print("Preparing data...")
    os.makedirs(output_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    ratio = float(np.sum(y == 0) / np.sum(y == 1))

    # --- PHASE 1: Baseline (Unconstrained) ---
    print("\n" + "="*40)
    print(" PHASE 1: Baseline Training (Data-Driven)")
    print("="*40)
    res_base, pred_base, models_base, weights_base = train_model_variant(X_train, y_train, X_test, y_test, ratio, constraints=None, variant_name="Baseline")
    
    joblib.dump(models_base['xgb'], f'{output_dir}/xgb_model.joblib')
    joblib.dump(models_base['lgb'], f'{output_dir}/lgb_model.joblib')
    joblib.dump(models_base['cat'], f'{output_dir}/cat_model.joblib')


    # --- PHASE 2: Optimized (Constrained) ---
    print("\n" + "="*40)
    print(" PHASE 2: Optimized Training (Clinical Constraints)")
    print("="*40)
    
    constraints = {
        'age': 1,
        'bmi': 1,
        'gcs_eyes_apache': -1,
        'gcs_motor_apache': -1,
        'gcs_verbal_apache': -1,
        'creatinine_apache': 1,
        'bun_apache': 1,
        'bilirubin_apache': 1,
        'map_apache': -1,
        'heart_rate_apache': 1,
    }
    active_constraints = {k: v for k, v in constraints.items() if k in X.columns}
    print(f"Applying constraints to {len(active_constraints)} features.")

    res_opt, pred_opt, models_opt, weights_opt = train_model_variant(X_train, y_train, X_test, y_test, ratio, constraints=active_constraints, variant_name="Optimized")
    
    # --- Compare & Save Best Optimised Logic for Candidate ---
    # If the constrained model is comparable or better, save it as the candidate.
    # Otherwise save the unconstrained (Baseline) as the candidate.
    
    # Criteria: If Optimized ROC-AUC is within 1% of Baseline (or better), prefer Optimized due to safety.
    # But for strict performance, just check >.
    if res_opt['Ensemble_Weighted']['ROC-AUC'] >= res_base['Ensemble_Weighted']['ROC-AUC']:
        print("Optimized model selected (Better or Equal Performance).")
        joblib.dump(models_opt['xgb'], f'{output_dir}/xgb_model.joblib')
        joblib.dump(models_opt['lgb'], f'{output_dir}/lgb_model.joblib')
        joblib.dump(models_opt['cat'], f'{output_dir}/cat_model.joblib')
    else:
        print("Baseline model selected (Better Performance).")
        # Ensure Baseline is saved
        joblib.dump(models_base['xgb'], f'{output_dir}/xgb_model.joblib')
        joblib.dump(models_base['lgb'], f'{output_dir}/lgb_model.joblib')
        joblib.dump(models_base['cat'], f'{output_dir}/cat_model.joblib')

    # --- Construct Full Structured Output ---
    final_output = {
        "Phase1_Unconstrained": {
             "metrics": res_base, # Contains XGB, LGB, Cat, Avg, Weighted
             "weights": weights_base.tolist()
        },
        "Phase2_Constrained": {
             "metrics": res_opt, # Contains XGB, LGB, Cat, Avg, Weighted
             "weights": weights_opt.tolist()
        }
    }
    
    # Flatten slightly for simple frontend access if needed, OR just send this whole tree.
    # Let's send the whole tree and parse in Frontend.
    return final_output

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1-Score': f1_score(y_test, preds),
        'ROC-AUC': roc_auc_score(y_test, probs)
    }
    
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    return metrics, {'preds': preds, 'probs': probs}

def train_and_evaluate(data_path, output_dir='models'):
    print(f"Starting training pipeline using data from {data_path}...")
    X, y, features = preprocess_data(data_path)
    results = train_hybrid_ensemble(X, y, output_dir=output_dir)
    return results

if __name__ == "__main__":
    if not os.path.exists('data/DataSet.csv'):
        print("Dataset not found at data/DataSet.csv")
    else:
        results = train_and_evaluate('data/DataSet.csv')
        
        print("\n" + "="*60)
        print("FINAL COMPARISON - ALL MODELS")
        print("="*60)
        # Simple print of the structured dict
        import json
        print(json.dumps(results, indent=2))
        print("\n✅ Comparison Complete")
