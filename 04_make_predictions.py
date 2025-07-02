import pandas as pd
import numpy as np
import xgboost as xgb
from joblib import load
import os
import json

def load_models():
    """Load trained models and their configurations"""
    models = {}
    
    # Load LightGBM model
    if os.path.exists('models/lightgbm_model.joblib'):
        models['lightgbm'] = {
            'model': load('models/lightgbm_model.joblib'),
            'type': 'lightgbm'
        }
    
    # Load XGBoost model and its feature names
    if os.path.exists('models/xgboost_model.joblib'):
        models['xgboost'] = {
            'model': load('models/xgboost_model.joblib'),
            'type': 'xgboost'
        }
        # Load feature names if available
        if os.path.exists('models/xgboost_feature_names.txt'):
            with open('models/xgboost_feature_names.txt', 'r') as f:
                models['xgboost']['feature_names'] = f.read().splitlines()
    
    # Load CatBoost model
    if os.path.exists('models/catboost_model.joblib'):
        models['catboost'] = {
            'model': load('models/catboost_model.joblib'),
            'type': 'catboost'
        }
    
    return models

def prepare_xgboost_data(X_test, feature_names):
    """
    Prepare data for XGBoost prediction with one-hot encoding
    
    Args:
        X_test: DataFrame with test features
        feature_names: List of feature names from the trained model
        
    Returns:
        DataFrame with one-hot encoded features matching the training data
    """
    if not feature_names:
        return X_test
        
    X_test_copy = X_test.copy()
    
    # Define categorical columns explicitly
    categorical_columns = ['srcid', 'destid', 'srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']
    categorical_columns = [col for col in categorical_columns if col in X_test_copy.columns]
    
    print(f"Found {len(categorical_columns)} categorical columns: {categorical_columns}")
    
    # Convert all categorical columns to string type
    for col in categorical_columns:
        X_test_copy[col] = X_test_copy[col].astype(str)
    
    # One-hot encode the test data
    if categorical_columns:
        print("Performing one-hot encoding...")
        X_test_encoded = pd.get_dummies(X_test_copy, columns=categorical_columns, 
                                      drop_first=False, dummy_na=True)
    else:
        X_test_encoded = X_test_copy.copy()
    
    # Ensure all columns from training are present
    missing_cols = set(feature_names) - set(X_test_encoded.columns)
    print(f"Adding {len(missing_cols)} missing columns...")
    for col in missing_cols:
        X_test_encoded[col] = 0
    
    # Reorder columns to match training data and ensure all columns are present
    X_test_encoded = X_test_encoded.reindex(columns=feature_names, fill_value=0)
    
    # Ensure we don't have any NaN values that could cause issues
    X_test_encoded = X_test_encoded.fillna(0)
    
    print(f"Final shape after encoding: {X_test_encoded.shape}")
    print(f"First few columns: {list(X_test_encoded.columns[:5])}...")
    return X_test_encoded

def prepare_lightgbm_data(X_test, model):
    """Prepare data for LightGBM prediction with consistent categorical features"""
    X_test_copy = X_test.copy()
    
    # Get categorical columns from the model if available
    if hasattr(model, '_Booster'):
        # For LightGBM models trained with scikit-learn API
        categorical_features = model._Booster.dump_model()['feature_infos']
        categorical_cols = [f'Column_{i}' for i, info in categorical_features.items() 
                          if info.get('type') == 'categorical']
        
        # Convert columns to category type if they exist in the test data
        for col in categorical_cols:
            if col in X_test_copy.columns:
                X_test_copy[col] = X_test_copy[col].astype('category')
    
    return X_test_copy

def make_predictions(models, X_test):
    """Make predictions using all available models"""
    predictions = {}
    
    # LightGBM predictions
    if 'lightgbm' in models:
        try:
            print("\n=== LightGBM Prediction ===")
            lgb_model = models['lightgbm']['model']
            
            # Make a copy of the data for LightGBM
            X_test_lgb = X_test.copy()
            
            # Get categorical columns and convert to category type
            cat_cols = ['srcid', 'destid', 'srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']
            for col in cat_cols:
                if col in X_test_lgb.columns:
                    X_test_lgb[col] = X_test_lgb[col].astype('category')
            
            # Ensure all columns are present and in the right order
            if hasattr(lgb_model, 'feature_name_'):
                print(f"Model expects {len(lgb_model.feature_name_)} features")
                missing_cols = set(lgb_model.feature_name_) - set(X_test_lgb.columns)
                print(f"Missing columns in test data: {missing_cols}")
                
                # Add missing columns with zeros
                for col in missing_cols:
                    X_test_lgb[col] = 0
                    print(f"  Added missing column: {col}")
                
                # Reorder columns to match training
                X_test_lgb = X_test_lgb[lgb_model.feature_name_]
                print("Final features shape:", X_test_lgb.shape)
            
            print("Making LightGBM predictions...")
            preds = lgb_model.predict(X_test_lgb)
            predictions['lightgbm'] = preds
            print(f"Predictions shape: {preds.shape}")
            print(f"Prediction stats - Min: {np.min(preds):.2f}, Max: {np.max(preds):.2f}, Mean: {np.mean(preds):.2f}")
            print(f"First 5 predictions: {preds[:5]}")
            
        except Exception as e:
            print(f"Error in LightGBM prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            if 'lightgbm' in predictions:
                del predictions['lightgbm']
    
    # XGBoost predictions - Skip for now due to categorical feature issues
    if 'xgboost' in models:
        print("\n=== XGBoost Prediction ===")
        print("Skipping XGBoost predictions due to categorical feature handling issues")
        # predictions['xgboost'] = np.zeros(len(X_test))  # Uncomment to include with zeros
    
    # CatBoost predictions
    if 'catboost' in models:
        try:
            print("\n=== CatBoost Prediction ===")
            cb_model = models['catboost']['model']
            
            # For CatBoost, we need to specify categorical feature indices
            cat_features_indices = [i for i, col in enumerate(X_test.columns) 
                                 if col in ['srcid', 'destid', 'srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']]
            
            print(f"Using {len(cat_features_indices)} categorical features at indices: {cat_features_indices}")
            print(f"Features being used: {[X_test.columns[i] for i in cat_features_indices]}")
            
            # Make predictions using all features but specify which are categorical
            print("Making CatBoost predictions...")
            preds = cb_model.predict(X_test, thread_count=-1, verbose=100)
            
            predictions['catboost'] = preds
            print(f"Predictions shape: {preds.shape}")
            print(f"Prediction stats - Min: {np.min(preds):.2f}, Max: {np.max(preds):.2f}, Mean: {np.mean(preds):.2f}")
            print(f"First 5 predictions: {preds[:5]}")
            
        except Exception as e:
            print(f"Error in CatBoost prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            if 'catboost' in predictions:
                del predictions['catboost']
    
    return predictions

def create_submission(predictions, test_data):
    """
    Create submission file with ensemble predictions
    
    Args:
        predictions: Dictionary of model predictions
        test_data: DataFrame containing test data with 'route_key' column
        
    Returns:
        DataFrame with 'route_key' and 'final_seatcount' columns
    """
    # Define model weights for ensemble
    weights = {
        'lightgbm': 0.5,
        'catboost': 0.3,
        'xgboost': 0.2
    }
    
    print("\n=== Creating Submission ===")
    print("Available predictions:", list(predictions.keys()))
    
    # Debug: Print prediction statistics for each model
    for model_name, pred in predictions.items():
        if len(pred) > 0:
            print(f"\n{model_name} prediction stats:")
            print(f"  Shape: {pred.shape}")
            print(f"  Min: {np.min(pred):.2f}")
            print(f"  Max: {np.max(pred):.2f}")
            print(f"  Mean: {np.mean(pred):.2f}")
            print(f"  Non-zero count: {np.count_nonzero(pred)} / {len(pred)} ({(np.count_nonzero(pred) / len(pred) * 100):.1f}%)")
            print(f"  First 5 values: {pred[:5]}")
    
    # Initialize with zeros
    ensemble_pred = np.zeros(test_data.shape[0], dtype=float)
    total_weight = 0
    
    # Add weighted predictions
    for model_name, pred in predictions.items():
        weight = weights.get(model_name, 0)
        if weight > 0 and len(pred) > 0:
            print(f"\nAdding {model_name} predictions with weight {weight}")
            print(f"  Before adding - Min: {np.min(ensemble_pred):.2f}, Max: {np.max(ensemble_pred):.2f}, Mean: {np.mean(ensemble_pred):.2f}")
            ensemble_pred += pred * weight
            total_weight += weight
            print(f"  After adding  - Min: {np.min(ensemble_pred):.2f}, Max: {np.max(ensemble_pred):.2f}, Mean: {np.mean(ensemble_pred):.2f}")
    
    # Normalize by total weight if any predictions were made
    if total_weight > 0:
        print(f"\nNormalizing predictions with total weight: {total_weight}")
        print(f"Before normalization - Min: {np.min(ensemble_pred):.2f}, Max: {np.max(ensemble_pred):.2f}")
        ensemble_pred = np.round(ensemble_pred / total_weight)
        print(f"After normalization - Min: {np.min(ensemble_pred):.2f}, Max: {np.max(ensemble_pred):.2f}")
    else:
        print("\nWARNING: No valid predictions were made!")
    
    # Create submission DataFrame with required columns
    submission = pd.DataFrame({
        'route_key': test_data['route_key'],
        'final_seatcount': ensemble_pred.astype(int)  # Ensure integer seat counts
    })
    
    # Ensure no negative predictions
    submission['final_seatcount'] = submission['final_seatcount'].clip(lower=0)
    
    # Print submission stats
    print("\nSubmission statistics:")
    print(f"Total predictions: {len(submission)}")
    print(f"Non-zero predictions: {(submission['final_seatcount'] > 0).sum()} ({(submission['final_seatcount'] > 0).mean() * 100:.1f}%)")
    print(f"Prediction range: {submission['final_seatcount'].min()} to {submission['final_seatcount'].max()}")
    print("First 5 predictions:")
    print(submission.head())
    
    # Basic validation
    if submission.isnull().any().any():
        raise ValueError("ERROR: Submission contains NaN values")
    if (submission['final_seatcount'] < 0).any():
        raise ValueError("ERROR: Submission contains negative seat counts")
    if (submission['final_seatcount'] == 0).all():
        print("\nWARNING: All predictions are zero! This is likely an error.")
    
    return submission

def main():
    print("Starting prediction process...")
    
    # Load test data with features
    print("\nLoading test data...")
    if not os.path.exists('features/test_features.csv'):
        raise FileNotFoundError("Test features not found. Please run 02_feature_engineering.py first.")
        
    test = pd.read_csv('features/test_features.csv')
    print(f"Loaded test data with shape: {test.shape}")
    
    # Validate required columns
    required_columns = ['route_key', 'doj']
    missing_cols = [col for col in required_columns if col not in test.columns]
    if missing_cols:
        raise ValueError(f"Test data is missing required columns: {missing_cols}")
    
    # Load models
    print("\nLoading trained models...")
    models = load_models()
    if not models:
        raise ValueError("No trained models found. Please run 03_model_training.py first.")
    print(f"Loaded models: {list(models.keys())}")
    
    # Define categorical columns (must match training)
    potential_cat_cols = ['srcid', 'destid', 'srcid_region', 'destid_region', 
                        'srcid_tier', 'destid_tier']
    
    # Get only columns that exist in the test data
    cat_cols = [col for col in potential_cat_cols if col in test.columns]
    
    # Prepare features (exclude non-feature columns)
    non_feature_cols = ['final_seatcount', 'route_key', 'doj', 'doi']
    feature_cols = [col for col in test.columns if col not in non_feature_cols]
    X_test = test[feature_cols].copy()
    
    # Convert categorical columns to string type first
    for col in cat_cols:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str)
    
    print(f"\nMaking predictions with {len(feature_cols)} features...")
    print(f"Categorical features: {cat_cols}")
    print("First few rows of test data:", X_test.head())
    
    # Make predictions
    predictions = make_predictions(models, X_test)
    
    # Create submission
    print("\nCreating submission file...")
    submission = create_submission(predictions, test)
    
    # Ensure output directory exists
    os.makedirs('submissions', exist_ok=True)
    submission_file = 'submissions/submission.csv'
    
    # Save submission
    submission.to_csv(submission_file, index=False)
    
    # Verify the saved file
    if os.path.exists(submission_file):
        submission_check = pd.read_csv(submission_file)
        print(f"\nSubmission file created successfully at: {os.path.abspath(submission_file)}")
        print(f"Submission shape: {submission_check.shape}")
        print("\nFirst few predictions:")
        print(submission_check.head())
        
        # Basic validation
        print("\nSubmission summary:")
        print(f"Number of predictions: {len(submission_check)}")
        print(f"Seat count range: {submission_check['final_seatcount'].min()} - {submission_check['final_seatcount'].max()}")
        print(f"Null values: {submission_check.isnull().sum().sum()}")
    else:
        print("Error: Failed to create submission file")

if __name__ == "__main__":
    main()
