import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
from joblib import dump
import os

def load_features():
    """Load processed features"""
    train = pd.read_csv('features/train_features.csv')
    test = pd.read_csv('features/test_features.csv')
    return train, test

def prepare_data(train, test):
    """Prepare data for training"""
    # Define columns to drop
    drop_cols = ['final_seatcount', 'route_key', 'doj', 'doi']
    
    # Get categorical columns that exist in the dataframe
    potential_cat_cols = ['srcid', 'destid', 'srcid_region', 'destid_region', 
                        'srcid_tier', 'destid_tier']
    cat_cols = [col for col in potential_cat_cols if col in train.columns]
    
    # Get all feature columns (exclude drop_cols and non-numeric columns that aren't categorical)
    feature_cols = []
    for col in train.columns:
        if col not in drop_cols:
            # Keep numeric columns and categorical columns
            if pd.api.types.is_numeric_dtype(train[col]) or col in cat_cols:
                feature_cols.append(col)
    
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Separate features and target
    X = train[feature_cols].copy()
    y = train['final_seatcount'].copy()
    
    # Convert categorical columns to category type
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Prepare test data
    X_test = test[feature_cols].copy()
    for col in cat_cols:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('category')
    
    return X_train, X_val, y_train, y_val, X_test, feature_cols, cat_cols

def train_lightgbm(X_train, X_val, y_train, y_val, cat_cols):
    """Train LightGBM model with hyperparameter tuning"""
    print("Training LightGBM model...")
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 1000  # Add fixed number of estimators
        }
        
        model = lgb.LGBMRegressor(**params)
        
        # Use callbacks for early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=1),
            lgb.log_evaluation(period=100)
        ]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=callbacks
        )
        
        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        return rmse
    
    # Optimize hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)  # Reduced to 5 trials for faster execution
    
    # Train final model with best params
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 1000
    })
    
    model = lgb.LGBMRegressor(**best_params)
    
    # Use callbacks for early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=1),
        lgb.log_evaluation(period=100)
    ]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=callbacks
    )
    
    return model

def train_xgboost(X_train, X_val, y_train, y_val):
    """Train XGBoost model with one-hot encoding for categorical variables"""
    print("Training XGBoost model...")
    
    # Create a copy of the data to avoid modifying the original
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    
    # Identify categorical columns
    categorical_columns = X_train_xgb.select_dtypes(include=['category', 'object']).columns.tolist()
    
    # Convert all columns to string type for consistent one-hot encoding
    for col in categorical_columns:
        X_train_xgb[col] = X_train_xgb[col].astype(str)
        X_val_xgb[col] = X_val_xgb[col].astype(str)
    
    # Get all possible categorical values from training data
    categorical_values = {}
    for col in categorical_columns:
        categorical_values[col] = X_train_xgb[col].unique()
    
    # Perform one-hot encoding on training data
    X_train_encoded = pd.get_dummies(X_train_xgb, columns=categorical_columns, drop_first=False)
    
    # For validation data, ensure all categories from training are present
    X_val_encoded = X_val_xgb.copy()
    for col in categorical_columns:
        # Add missing categories to validation set
        missing_cats = set(categorical_values[col]) - set(X_val_encoded[col].unique())
        for cat in missing_cats:
            X_val_encoded[f"{col}_{cat}"] = 0
        
        # One-hot encode each categorical column
        dummies = pd.get_dummies(X_val_encoded[col], prefix=col, drop_first=False)
        X_val_encoded = pd.concat([X_val_encoded.drop(columns=[col]), dummies], axis=1)
    
    # Ensure validation set has all columns from training set
    for col in X_train_encoded.columns:
        if col not in X_val_encoded.columns:
            X_val_encoded[col] = 0
    
    # Reorder columns to match training data
    X_val_encoded = X_val_encoded[X_train_encoded.columns]
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train_encoded, label=y_train)
    dval = xgb.DMatrix(X_val_encoded, label=y_val)
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'tree_method': 'hist'
    }
    
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    
    model = xgb.train(
        params, 
        dtrain,
        num_boost_round=1000,
        evals=watchlist,
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Store the column names for later use in prediction
    model.feature_names = X_train_encoded.columns.tolist()
    
    return model, X_train_encoded.columns.tolist()

def train_catboost(X_train, X_val, y_train, y_val, cat_cols):
    """Train CatBoost model"""
    print("Training CatBoost model...")
    
    cat_features = [i for i, col in enumerate(X_train.columns) if col in cat_cols]
    
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        early_stopping_rounds=50,
        cat_features=cat_features,
        verbose=100
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    
    return model

def evaluate_model(model, X_val, y_val, model_name, feature_names=None):
    """Evaluate model performance"""
    if hasattr(model, 'feature_names'):
        # Handle XGBoost model with feature names
        try:
            # Try to use the feature names directly
            dval = xgb.DMatrix(X_val[feature_names] if feature_names is not None else X_val)
        except KeyError:
            # If that fails, perform one-hot encoding
            X_val_encoded = X_val.copy()
            # Ensure all feature columns exist in validation data
            for col in feature_names:
                if col not in X_val_encoded.columns and col != 'const':
                    X_val_encoded[col] = 0
            # Reorder columns to match training data
            X_val_encoded = X_val_encoded[feature_names]
            dval = xgb.DMatrix(X_val_encoded)
        y_pred = model.predict(dval)
    else:
        y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"{model_name} RMSE: {rmse:.4f}")
    return rmse

def save_model(model, name, feature_importance=True):
    """Save model and optionally feature importance"""
    os.makedirs('models', exist_ok=True)
    
    # Save model
    dump(model, f'models/{name}.joblib')
    
    # Save feature importance if available
    if feature_importance and hasattr(model, 'feature_importances_'):
        # Get feature names
        if hasattr(model, 'feature_names_in_'):  # For scikit-learn models
            feature_names = model.feature_names_in_
        elif hasattr(model, 'feature_name_'):  # For LightGBM
            feature_names = model.feature_name_
        elif hasattr(model, 'feature_names_'):  # For CatBoost
            feature_names = model.feature_names_
        else:
            print(f"No feature names found for {name}")
            return
        
        # Create and save feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv(f'models/{name}_feature_importance.csv', index=False)
        
        # Save feature names for prediction
        with open(f'models/{name}_feature_names.txt', 'w') as f:
            f.write('\n'.join(feature_names))
    
    print(f"Model saved to models/{name}.joblib")

def main():
    # Load features
    train, test = load_features()
    
    # Prepare data
    X_train, X_val, y_train, y_val, X_test, feature_cols, cat_cols = prepare_data(train, test)
    
    # Train models
    models = {}
    
    # Train and evaluate LightGBM
    lgb_model = train_lightgbm(X_train, X_val, y_train, y_val, cat_cols)
    lgb_rmse = evaluate_model(lgb_model, X_val, y_val, 'LightGBM')
    save_model(lgb_model, 'lightgbm_model')
    models['lightgbm'] = lgb_model
    
    # Train and evaluate XGBoost
    xgb_model, xgb_feature_names = train_xgboost(X_train, X_val, y_train, y_val)
    
    # For XGBoost, we need to prepare the validation data the same way as in training
    X_val_xgb = X_val.copy()
    categorical_columns = X_val_xgb.select_dtypes(include=['category', 'object']).columns.tolist()
    
    # Convert all columns to string type for consistent one-hot encoding
    for col in categorical_columns:
        X_val_xgb[col] = X_val_xgb[col].astype(str)
    
    # One-hot encode validation data
    X_val_encoded = pd.get_dummies(X_val_xgb, columns=categorical_columns, drop_first=False)
    
    # Ensure all columns from training are present
    for col in xgb_feature_names:
        if col not in X_val_encoded.columns:
            X_val_encoded[col] = 0
    
    # Reorder columns to match training data
    X_val_encoded = X_val_encoded[xgb_feature_names]
    
    xgb_rmse = evaluate_model(xgb_model, X_val_encoded, y_val, 'XGBoost')
    save_model(xgb_model, 'xgboost_model', feature_importance=False)
    models['xgboost'] = (xgb_model, xgb_feature_names)  # Store both model and feature names
    
    # Train and evaluate CatBoost
    cb_model = train_catboost(X_train, X_val, y_train, y_val, cat_cols)
    cb_rmse = evaluate_model(cb_model, X_val, y_val, 'CatBoost')
    save_model(cb_model, 'catboost_model')
    models['catboost'] = cb_model
    
    # Prepare data for predictions
    X_val_copy = X_val.copy()
    
    # Get predictions from each model
    print("\nGenerating LightGBM predictions...")
    lgb_pred = models['lightgbm'].predict(X_val_copy)
    
    print("Generating CatBoost predictions...")
    cb_pred = models['catboost'].predict(X_val_copy)
    
    print("Generating XGBoost predictions...")
    xgb_model, xgb_feature_names = models['xgboost']
    
    # Prepare XGBoost validation data with the same one-hot encoding as training
    X_val_xgb = X_val_copy.copy()
    
    # Get categorical columns
    categorical_columns = X_val_xgb.select_dtypes(include=['category', 'object']).columns.tolist()
    
    # Convert all categorical columns to string type
    for col in categorical_columns:
        X_val_xgb[col] = X_val_xgb[col].astype(str)
    
    # One-hot encode the validation data
    X_val_encoded = pd.get_dummies(X_val_xgb, columns=categorical_columns, drop_first=False)
    
    # Ensure all columns from training are present
    for col in xgb_feature_names:
        if col not in X_val_encoded.columns:
            X_val_encoded[col] = 0
    
    # Reorder columns to match training data
    X_val_encoded = X_val_encoded[xgb_feature_names]
    
    # Make XGBoost predictions
    xgb_dval = xgb.DMatrix(X_val_encoded)
    xgb_pred = xgb_model.predict(xgb_dval)
    
    # Create weighted ensemble (adjust weights as needed)
    print("\nCalculating ensemble predictions...")
    ensemble_pred = 0.5 * lgb_pred + 0.3 * cb_pred + 0.2 * xgb_pred
    
    # Calculate and print ensemble RMSE
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    print(f"\nEnsemble RMSE: {ensemble_rmse:.4f}")
    print("Individual model RMSEs:")
    print(f"- LightGBM: {np.sqrt(mean_squared_error(y_val, lgb_pred)):.4f}")
    print(f"- CatBoost: {np.sqrt(mean_squared_error(y_val, cb_pred)):.4f}")
    print(f"- XGBoost: {np.sqrt(mean_squared_error(y_val, xgb_pred)):.4f}")

if __name__ == "__main__":
    main()