import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime, timedelta
import holidays
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'data_paths': {
        'train': 'train/train.csv',
        'test': 'test.csv',
        'transactions': 'train/transactions.csv',
        'output': 'submissions/submission.csv',
        'model_dir': 'models',
        'features_dir': 'features'
    },
    'model_params': {
        'n_trials': 5,  # Reduced for faster execution
        'early_stopping_rounds': 50,
        'random_state': 42,
        'use_lightgbm': True,
        'use_xgboost': True,
        'use_catboost': True
    }
}

# Ensure directories exist
os.makedirs(CONFIG['data_paths']['model_dir'], exist_ok=True)
os.makedirs(CONFIG['data_paths']['features_dir'], exist_ok=True)

class FeatureEngineer:
    """Handles all feature engineering tasks with robust feature handling"""
    
    @staticmethod
    def add_date_features(df, date_column):
        """Add advanced date-based features"""
        df = df.copy()
        df[f'{date_column}_month'] = df[date_column].dt.month
        df[f'{date_column}_day'] = df[date_column].dt.day
        df[f'{date_column}_dayofyear'] = df[date_column].dt.dayofyear
        df[f'{date_column}_weekday'] = df[date_column].dt.weekday
        df[f'{date_column}_week'] = df[date_column].dt.isocalendar().week
        df[f'{date_column}_quarter'] = df[date_column].dt.quarter
        df[f'{date_column}_year'] = df[date_column].dt.year
        df[f'{date_column}_is_weekend'] = (df[date_column].dt.weekday >= 5).astype(int)
        df[f'{date_column}_is_month_start'] = df[date_column].dt.is_month_start.astype(int)
        df[f'{date_column}_is_month_end'] = df[date_column].dt.is_month_end.astype(int)
        df[f'{date_column}_is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
        df[f'{date_column}_is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
        return df
    
    @staticmethod
    def add_holiday_features(df, date_column, country='IN'):
        """Add holiday-related features with more robust handling"""
        df = df.copy()
        years = range(df[date_column].dt.year.min(), df[date_column].dt.year.max() + 1)
        holiday_dict = {}
        
        for year in years:
            for date, name in holidays.India(years=year).items():
                holiday_dict[date] = name
        
        df[f'{date_column}_is_holiday'] = df[date_column].isin(holiday_dict.keys()).astype(int)
        
        # More robust days to next holiday calculation
        all_holidays = sorted(holiday_dict.keys())
        
        def days_to_next_holiday(date):
            # Convert input date to datetime.date for consistent comparison
            if hasattr(date, 'date'):
                date = date.date()
            
            # Find future holidays (convert holiday dates to date objects if needed)
            future_dates = [d.date() if hasattr(d, 'date') else d for d in all_holidays 
                          if (d.date() if hasattr(d, 'date') else d) >= date]
            
            if not future_dates:
                return 365
                
            next_holiday = min(future_dates)
            return (next_holiday - date).days
        
        df[f'{date_column}_days_to_holiday'] = df[date_column].apply(days_to_next_holiday)
        return df
    
    @staticmethod
    def create_lag_features(transactions, lags=[1, 2, 3, 7, 14, 21, 28]):
        """Create lag features with more robust handling"""
        if transactions.empty:
            return transactions
            
        transactions = transactions.sort_values(['srcid', 'destid', 'doj']).copy()
        
        # Define base columns to create lags for
        base_columns = []
        if 'seatcount' in transactions.columns:
            base_columns.append('seatcount')
        if 'searchcount' in transactions.columns:
            base_columns.append('searchcount')
        
        if not base_columns:
            print("Warning: No valid columns found for creating lag features")
            return transactions
        
        for lag in lags:
            for col in base_columns:
                # Create lag feature
                lag_col = f'{col}_lag_{lag}'
                transactions[lag_col] = transactions.groupby(['srcid', 'destid'])[col].shift(lag)
                
                # Add lagged rolling statistics if it's a numeric column
                if pd.api.types.is_numeric_dtype(transactions[col]):
                    for window in [3, 7, 14]:
                        roll_col = f'{col}_lag{lag}_rolling_mean{window}'
                        transactions[roll_col] = transactions.groupby(['srcid', 'destid'])[lag_col].transform(
                            lambda x: x.rolling(window=window, min_periods=1).mean()
                        )
        
        return transactions
    
    @staticmethod
    def create_rolling_features(transactions, windows=[3, 7, 14]):
        """Create rolling window features with more statistics"""
        if transactions.empty:
            return transactions
            
        transactions = transactions.sort_values(['srcid', 'destid', 'doj']).copy()
        
        # Define base columns to create rolling features for
        base_columns = []
        for col in ['seatcount', 'searchcount']:
            if col in transactions.columns and pd.api.types.is_numeric_dtype(transactions[col]):
                base_columns.append(col)
        
        if not base_columns:
            print("Warning: No valid numeric columns found for creating rolling features")
            return transactions
        
        for window in windows:
            for col in base_columns:
                # Skip if column doesn't exist or isn't numeric
                if col not in transactions.columns or not pd.api.types.is_numeric_dtype(transactions[col]):
                    continue
                    
                # Create rolling statistics
                group = transactions.groupby(['srcid', 'destid'])[col]
                
                # Mean
                transactions[f'{col}_rolling_mean_{window}'] = group.transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
                # Std
                transactions[f'{col}_rolling_std_{window}'] = group.transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
                )
                # Min
                transactions[f'{col}_rolling_min_{window}'] = group.transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
                )
                # Max
                transactions[f'{col}_rolling_max_{window}'] = group.transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
                )
        
        return transactions
    
    @classmethod
    def add_route_features(cls, df):
        """Add route-based features"""
        df = df.copy()
        # Create route ID
        df['route_id'] = df['srcid'].astype(str) + '_' + df['destid'].astype(str)
        
        # Add route statistics if target column exists
        if 'seatcount' in df.columns:
            route_stats = df.groupby('route_id')['seatcount'].agg(['mean', 'std', 'count']).add_prefix('route_')
            df = df.merge(route_stats, on='route_id', how='left')
        
        return df
    
    @classmethod
    def add_interaction_features(cls, df):
        """Add interaction features between source and destination"""
        df = df.copy()
        # Simple interaction features
        df['src_dest_interaction'] = df['srcid'].astype(str) + '_' + df['destid'].astype(str)
        return df
    
    @classmethod
    def prepare_features(cls, train, test, transactions):
        """Prepare features for training and testing with robust feature handling"""
        print("Starting feature engineering...")
        
        # Make copies to avoid modifying originals
        train = train.copy()
        test = test.copy()
        transactions = transactions.copy()
        
        # 1. Basic date features
        print("Adding date features...")
        train = cls.add_date_features(train, 'doj')
        test = cls.add_date_features(test, 'doj')
        
        # 2. Holiday features
        print("Adding holiday features...")
        train = cls.add_holiday_features(train, 'doj')
        test = cls.add_holiday_features(test, 'doj')
        
        # 3. Add route features before creating lags
        print("Adding route features...")
        train = cls.add_route_features(train)
        test = cls.add_route_features(test)
        
        # 4. Add interaction features
        print("Adding interaction features...")
        train = cls.add_interaction_features(train)
        test = cls.add_interaction_features(test)
        
        # 5. Process transactions
        print("Processing transaction features...")
        transactions = cls.create_lag_features(transactions)
        transactions = cls.create_rolling_features(transactions)
        
        # 6. Merge with transactions data (15 days before departure)
        print("Merging transaction data...")
        transactions_15d = transactions[transactions['dbd'] == 15].copy()
        
        # Ensure we have all required columns after merge
        merge_cols = ['doj', 'srcid', 'destid']
        
        # Save column lists before merge
        train_cols_before = set(train.columns)
        test_cols_before = set(test.columns)
        
        # Perform merges
        train = train.merge(transactions_15d, on=merge_cols, how='left')
        test = test.merge(transactions_15d, on=merge_cols, how='left')
        
        # 7. Handle missing values
        print("Handling missing values...")
        for df in [train, test]:
            # Fill numeric columns with 0
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
            # Fill categorical columns with 'missing'
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            df[cat_cols] = df[cat_cols].fillna('missing')
        
        # 8. Ensure test has same columns as train
        print("Aligning features between train and test...")
        missing_in_test = set(train.columns) - set(test.columns)
        extra_in_test = set(test.columns) - set(train.columns)
        
        # Add missing columns to test
        for col in missing_in_test:
            if col != 'seatcount':  # Skip target variable
                test[col] = 0
        
        # Remove extra columns from test
        test = test[train.columns]
        
        # 9. Ensure consistent dtypes
        print("Ensuring consistent data types...")
        for col in train.columns:
            if col in test.columns:
                test[col] = test[col].astype(train[col].dtype)
                
        print(f"Final train shape: {train.shape}, test shape: {test.shape}")
        print(f"Train columns: {len(train.columns)}, Test columns: {len(test.columns)}")
        
        return train, test
    
    @staticmethod
    def save_features(train, test, output_dir):
        """Save processed features"""
        os.makedirs(output_dir, exist_ok=True)
        train.to_csv(f'{output_dir}/train_features.csv', index=False)
        test.to_csv(f'{output_dir}/test_features.csv', index=False)


class ModelTrainer:
    """Handles model training, evaluation, and prediction with robust feature handling"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.feature_columns = None
        self.categorical_columns = None
        self.target_column = 'seatcount'
    
    def load_features(self):
        """Load processed features with error handling"""
        print("Loading processed features...")
        try:
            train_path = os.path.join(self.config['features_dir'], 'train_features.csv')
            test_path = os.path.join(self.config['features_dir'], 'test_features.csv')
            
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError("Feature files not found. Please run feature engineering first.")
                
            train = pd.read_csv(train_path, parse_dates=['doj'])
            test = pd.read_csv(test_path, parse_dates=['doj'])
            
            print(f"Loaded train shape: {train.shape}, test shape: {test.shape}")
            return train, test
            
        except Exception as e:
            print(f"Error loading features: {str(e)}")
            raise
    
    def _align_feature_columns(self, train, test, feature_cols):
        """Ensure train and test have the same feature columns"""
        # Add missing columns to test
        missing_in_test = set(feature_cols) - set(test.columns)
        if missing_in_test:
            print(f"Adding {len(missing_in_test)} missing columns to test data")
            for col in missing_in_test:
                if col in train.columns:
                    test[col] = train[col].iloc[0] if train[col].dtype == 'object' else 0
        
        # Remove extra columns from test
        extra_in_test = set(test.columns) - set(feature_cols)
        if extra_in_test:
            print(f"Removing {len(extra_in_test)} extra columns from test data")
            test.drop(columns=list(extra_in_test), inplace=True)
        
        # Ensure same column order
        test_cols = [col for col in feature_cols if col in test.columns]
        test = test[test_cols]
        
        # Ensure same dtypes
        for col in feature_cols:
            if col in train.columns and col in test.columns:
                test[col] = test[col].astype(train[col].dtype)
    
    def prepare_data(self, train, test):
        """Prepare data for training with robust feature handling"""
        print("\n=== Preparing Data for Training ===")
        
        # Make copies to avoid modifying originals
        train = train.copy()
        test = test.copy()
        
        # 1. Define non-feature columns
        non_feature_cols = [
            'doj', 'srcid', 'destid', 'srcid_region', 'destid_region',
            'srcid_tier', 'destid_tier', 'dbd', 'route_id', 'src_dest_interaction',
            self.target_column
        ]
        
        # 2. Get feature columns (exclude non-feature columns and any columns with all nulls)
        feature_cols = [
            col for col in train.columns 
            if col not in non_feature_cols and train[col].notna().any()
        ]
        
        print(f"Found {len(feature_cols)} feature columns")
        
        # 3. Handle missing values in features
        for col in feature_cols:
            if train[col].isna().any():
                if train[col].dtype in ['int64', 'float64']:
                    fill_val = train[col].median()
                    train[col].fillna(fill_val, inplace=True)
                    test[col].fillna(fill_val, inplace=True)
                else:
                    fill_val = train[col].mode()[0] if not train[col].mode().empty else 'missing'
                    train[col].fillna(fill_val, inplace=True)
                    test[col].fillna(fill_val, inplace=True)
        
        # 4. Identify categorical columns
        cat_cols = [
            col for col in feature_cols 
            if train[col].dtype == 'object' or train[col].nunique() < 20
        ]
        
        # 5. Convert categorical columns to string type
        for col in cat_cols:
            train[col] = train[col].astype(str)
            test[col] = test[col].astype(str)
        
        # 6. Ensure test has same columns as train
        self._align_feature_columns(train, test, feature_cols)
        
        # 7. Split into features and target
        X_train = train[feature_cols].copy()
        y_train = train[self.target_column].copy()
        X_test = test[feature_cols].copy()
        
        # 8. Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=0.2, 
            random_state=self.config['random_state'],
            shuffle=True
        )
        
        print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
        print(f"X_test shape: {X_test.shape}")
        
        # Store feature information for later use
        self.feature_columns = feature_cols
        self.categorical_columns = cat_cols
        
        return X_train, X_val, y_train, y_val, X_test, feature_cols, cat_cols
    
    def train_lightgbm(self, X_train, X_val, y_train, y_val):
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
                'random_state': self.config['random_state'],
                'n_estimators': 1000
            }
            
            model = lgb.LGBMRegressor(**params)
            
            callbacks = [
                lgb.early_stopping(stopping_rounds=self.config['early_stopping_rounds'], verbose=1),
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
        study.optimize(objective, n_trials=self.config['n_trials'])
        
        # Train final model with best params
        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': self.config['random_state'],
            'n_estimators': 1000
        })
        
        model = lgb.LGBMRegressor(**best_params)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=self.config['early_stopping_rounds'], verbose=1),
            lgb.log_evaluation(period=100)
        ]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=callbacks
        )
        
        return model
        
    def train_xgboost(self, X_train, X_val, y_train, y_val, cat_cols):
        """Train XGBoost model with one-hot encoding for categorical variables"""
        print("Training XGBoost model...")
        
        # Create a copy of the data to avoid modifying the original
        X_train_xgb = X_train.copy()
        X_val_xgb = X_val.copy()
        
        # Identify categorical columns
        categorical_columns = [col for col in cat_cols if col in X_train_xgb.columns]
        
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
            'seed': self.config['random_state'],
            'tree_method': 'hist'
        }
        
        watchlist = [(dtrain, 'train'), (dval, 'eval')]
        
        try:
            model = xgb.train(
                params, 
                dtrain,
                num_boost_round=1000,
                evals=watchlist,
                early_stopping_rounds=self.config['early_stopping_rounds'],
                verbose_eval=100
            )
            return model, list(X_train_encoded.columns)
        except Exception as e:
            print(f"Error training XGBoost model: {str(e)}")
            return None
    
    def train_catboost(self, X_train, X_val, y_train, y_val, cat_cols):
        """Train CatBoost model"""
        print("Training CatBoost model...")
        
        # Get categorical feature indices
        cat_features_indices = [i for i, col in enumerate(X_train.columns) if col in cat_cols]
        
        # Initialize CatBoost
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=self.config['random_state'],
            early_stopping_rounds=self.config['early_stopping_rounds'],
            verbose=100,
            cat_features=cat_features_indices
        )
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=100
        )
        
        return model
    
    def train_models(self, X_train, X_val, y_train, y_val, X_test, feature_cols, cat_cols):
        """Train all models"""
        models = {}
        
        # Train LightGBM
        if self.config.get('use_lightgbm', True):
            print("\n=== Training LightGBM ===")
            lgb_model = self.train_lightgbm(X_train, X_val, y_train, y_val)
            models['lightgbm'] = {
                'model': lgb_model,
                'type': 'lightgbm',
                'feature_names': feature_cols,
                'cat_cols': cat_cols
            }
        
        # Train XGBoost
        if self.config.get('use_xgboost', True):
            print("\n=== Training XGBoost ===")
            # xgb_model = self.train_xgboost(X_train, X_val, y_train, y_val, cat_cols)
            xgb_model, xgb_feature_names = self.train_xgboost(X_train, X_val, y_train, y_val, cat_cols)
            if xgb_model is not None:  # Only add if training was successful
                models['xgboost'] = {
                    'model': xgb_model,
                    'type': 'xgboost',
                    # 'feature_names': feature_cols,
                    'feature_names': xgb_feature_names,
                    'cat_cols': cat_cols
                }
        
        # Train CatBoost
        if self.config.get('use_catboost', True):
            print("\n=== Training CatBoost ===")
            cb_model = self.train_catboost(X_train, X_val, y_train, y_val, cat_cols)
            models['catboost'] = {
                'model': cb_model,
                'type': 'catboost',
                'feature_names': feature_cols,
                'cat_cols': cat_cols
            }
        
        # Save models
        self.save_models(models)
        
        return models, X_test
    
    def save_models(self, models):
        """Save trained models and their configurations"""
        os.makedirs(self.config['model_dir'], exist_ok=True)
        
        for name, model_info in models.items():
            model = model_info['model']
            model_path = os.path.join(self.config['model_dir'], f'{name}_model.joblib')
            
            # Save the model
            joblib.dump(model, model_path)
            
            # Save additional model-specific information
            if name == 'lightgbm' and hasattr(model, 'feature_name_'):
                # Save feature names for LightGBM
                feature_names_path = os.path.join(self.config['model_dir'], f'{name}_feature_names.txt')
                with open(feature_names_path, 'w') as f:
                    f.write('\n'.join(model.feature_name_))

            #####################################################################
            # elif name == 'xgboost':
            #     # Save feature names for XGBoost
            #     feature_names_path = os.path.join(self.config['model_dir'], f'{name}_feature_names.txt')
            #     with open(feature_names_path, 'w') as f:
            #         f.write('\n'.join(model.feature_names) if hasattr(model, 'feature_names') else '')
            #####################################################################

            elif name == 'xgboost':
                # Save feature names for XGBoost
                feature_names_path = os.path.join(self.config['model_dir'], f'{name}_feature_names.txt')
                feature_names = model_info.get('feature_names', [])
                with open(feature_names_path, 'w') as f:
                    f.write('\n'.join(feature_names))
            
            print(f"Saved {name} model to {model_path}")
    
    def load_models(self):
        """Load trained models"""
        models = {}
        
        # Helper function to load model with feature names
        def _load_model(name, model_type):
            model_path = os.path.join(self.config['model_dir'], f'{name}_model.joblib')
            if os.path.exists(model_path):
                model_info = {
                    'model': joblib.load(model_path),
                    'type': model_type
                }
                # Load feature names if available
                feature_names_path = os.path.join(self.config['model_dir'], f'{name}_feature_names.txt')
                if os.path.exists(feature_names_path):
                    with open(feature_names_path, 'r') as f:
                        model_info['feature_names'] = [line.strip() for line in f if line.strip()]
                return model_info
            return None
        
        # Load all models
        lgb_model = _load_model('lightgbm', 'lightgbm')
        if lgb_model:
            models['lightgbm'] = lgb_model
            
        xgb_model = _load_model('xgboost', 'xgboost')
        if xgb_model:
            models['xgboost'] = xgb_model
            
        cb_model = _load_model('catboost', 'catboost')
        if cb_model:
            models['catboost'] = cb_model
        
        print(f"Loaded {len(models)} models: {list(models.keys())}")
        return models
    
    def prepare_xgboost_data(self, X_test, feature_names):
        """Prepare data for XGBoost prediction with one-hot encoding"""
        # Make a copy to avoid modifying the original
        X_test_encoded = X_test.copy()
        
        # Identify categorical columns
        categorical_columns = ['srcid', 'destid', 'srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']
        categorical_columns = [col for col in categorical_columns if col in X_test_encoded.columns]
        
        print(f"Found {len(categorical_columns)} categorical columns: {categorical_columns}")
        
        # Convert categorical columns to string type for one-hot encoding
        for col in categorical_columns:
            X_test_encoded[col] = X_test_encoded[col].astype(str)
        
        # Perform one-hot encoding
        print("Performing one-hot encoding...")
        X_test_encoded = pd.get_dummies(X_test_encoded, columns=categorical_columns, drop_first=False)
        

        ###############################################################
        # Add any missing columns that were present during training
        # missing_cols = set(feature_names) - set(X_test_encoded.columns)
        # if missing_cols:
        #     print(f"Adding {len(missing_cols)} missing columns...")
        #     for col in missing_cols:
        #         X_test_encoded[col] = 0
        
        # # Ensure all training columns exist in test data
        # for col in feature_names:
        #     if col not in X_test_encoded.columns:
        #         X_test_encoded[col] = 0
        

        # Reorder columns to match training data and ensure all columns are present
        # X_test_encoded = X_test_encoded[feature_names]
        ###############################################################

        # Only select feature_names that exist after encoding
        missing_in_encoded = set(feature_names) - set(X_test_encoded.columns)

        if missing_in_encoded:
            print(f"Adding {len(missing_in_encoded)} missing columns: {missing_in_encoded}")
            for col in missing_in_encoded:
                X_test_encoded[col] = 0

        # Remove extra columns not in training feature list
        extra_in_encoded = set(X_test_encoded.columns) - set(feature_names)
        if extra_in_encoded:
            print(f"Removing {len(extra_in_encoded)} extra columns: {extra_in_encoded}")
            X_test_encoded.drop(columns=extra_in_encoded, inplace=True)

        # Reorder to match training
        X_test_encoded = X_test_encoded[feature_names]
        
        # Ensure we don't have any NaN values that could cause issues
        X_test_encoded = X_test_encoded.fillna(0)
        
        print(f"Final shape after encoding: {X_test_encoded.shape}")
        return X_test_encoded
    
    def make_predictions(self, models, X_test):
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
        
        # XGBoost predictions
        if 'xgboost' in models:
            try:
                print("\n=== XGBoost Prediction ===")
                xgb_model = models['xgboost']['model']
                
                # Get feature names used during training
                feature_names = models['xgboost'].get('feature_names', [])
                
                if not feature_names:
                    print("Warning: No feature names found for XGBoost model. Using test columns as-is.")
                    X_test_xgb = X_test.copy()
                else:
                    print(f"Preparing XGBoost data with {len(feature_names)} features...")
                    X_test_xgb = self.prepare_xgboost_data(X_test, feature_names)
                
                # Ensure we have the exact same features in the same order as training
                missing_cols = set(feature_names) - set(X_test_xgb.columns)
                if missing_cols:
                    print(f"Adding {len(missing_cols)} missing columns to test data...")
                    for col in missing_cols:
                        X_test_xgb[col] = 0

                ############################################
                # Reorder columns to match training exactly
                # X_test_xgb = X_test_xgb[feature_names]
                ############################################
                
                # Align one-hot encoded test columns with training columns (fix mismatch)
                train_columns_set = set(feature_names)
                test_columns_set = set(X_test_xgb.columns)

                extra_test_columns = test_columns_set - train_columns_set
                missing_test_columns = train_columns_set - test_columns_set

                if extra_test_columns:
                    print(f"Removing {len(extra_test_columns)} extra columns from test data not seen in training")
                    X_test_xgb.drop(columns=extra_test_columns, inplace=True)

                for col in missing_test_columns:
                    X_test_xgb[col] = 0  # Add missing columns

                # Reorder again after fixing
                X_test_xgb = X_test_xgb[feature_names]

                # Convert to DMatrix
                print("Converting to DMatrix...")
                dtest = xgb.DMatrix(X_test_xgb.values, feature_names=X_test_xgb.columns.tolist())
                
                print("Making XGBoost predictions...")
                preds = xgb_model.predict(dtest)
                predictions['xgboost'] = preds
                print(f"Predictions shape: {preds.shape}")
                print(f"Prediction stats - Min: {np.min(preds):.2f}, Max: {np.max(preds):.2f}, Mean: {np.mean(preds):.2f}")
                print(f"First 5 predictions: {preds[:5]}")
                
            except Exception as e:
                print(f"Error in XGBoost prediction: {str(e)}")
                import traceback
                traceback.print_exc()
        
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
        
        return predictions
    
    def create_submission(self, predictions, test_data):
        """Create submission file with ensemble predictions"""
        if not predictions:
            raise ValueError("No predictions available for submission")
        
        # Create ensemble predictions (simple average of all available models)
        print("\nCreating ensemble predictions...")
        all_preds = np.array(list(predictions.values()))
        
        # Calculate ensemble predictions (simple average)
        final_predictions = np.mean(all_preds, axis=0)
        
        # Print individual model contributions
        print("\nModel contributions to ensemble:")
        for model_name, preds in predictions.items():
            print(f"- {model_name}: {np.mean(preds):.2f} (mean prediction)")
        
        print(f"Ensemble mean prediction: {np.mean(final_predictions):.2f}")
        
        # Ensure predictions are non-negative integers (seat counts can't be negative)
        final_predictions = np.round(np.maximum(0, final_predictions)).astype(int)
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'route_key': test_data['route_key'],
            'final_seatcount': final_predictions
        })
        
        # Save submission file
        submission.to_csv(self.config['output'], index=False)
        print(f"\nSubmission file saved to {self.config['output']}")
        
        # Print submission summary
        print("\n=== Submission Summary ===")
        print(f"Total predictions: {len(submission)}")
        print(f"Predicted seat counts - Min: {submission['final_seatcount'].min()}, "
              f"Max: {submission['final_seatcount'].max()}, "
              f"Mean: {submission['final_seatcount'].mean():.2f}")
        
        return submission


def main():
    # Initialize configuration
    config = {
        'data_paths': {
            'train': 'train.csv',
            'test': 'test.csv',
            'transactions': 'train/transactions.csv',
            'output': 'submission.csv',
            'model_dir': 'models',
            'features_dir': 'features'
        },
        'model_params': {
            'n_trials': 5,
            'early_stopping_rounds': 50,
            'random_state': 42
        },
        **CONFIG  # Override with any command line args if needed
    }
    
    # Load data
    print("Loading data...")
    train = pd.read_csv(config['data_paths']['train'], parse_dates=['doj'])
    test = pd.read_csv(config['data_paths']['test'], parse_dates=['doj'])
    transactions = pd.read_csv(config['data_paths']['transactions'], parse_dates=['doj', 'doi'])
    print("Data loaded successfully")

    # Feature Engineering
    print("\n=== Feature Engineering ===")
    fe = FeatureEngineer()
    train, test = fe.prepare_features(train, test, transactions)
    fe.save_features(train, test, config['data_paths']['features_dir'])
    print("Features saved successfully")
    
    # Model Training and Prediction
    print("\n=== Model Training ===")
    trainer = ModelTrainer({
        **config['model_params'],
        'model_dir': config['data_paths']['model_dir'],
        'features_dir': config['data_paths']['features_dir'],
        'output': config['data_paths']['output']
    })
    
    # Prepare data for training
    X_train, X_val, y_train, y_val, X_test, feature_cols, cat_cols = trainer.prepare_data(train, test)
    
    # Train models
    models, X_test_processed = trainer.train_models(X_train, X_val, y_train, y_val, X_test, feature_cols, cat_cols)
    print("Models trained successfully")
    
    # Make predictions
    print("\n=== Making Predictions ===")
    predictions = trainer.make_predictions(models, X_test_processed)
    print("Predictions made successfully")
    
    # Create submission file
    submission = trainer.create_submission(predictions, test)
    print("\n=== Submission Summary ===")
    print(f"Total predictions: {len(submission)}")
    print(f"Predicted seat counts - Min: {submission['final_seatcount'].min()}, "
          f"Max: {submission['final_seatcount'].max()}, "
          f"Mean: {submission['final_seatcount'].mean():.2f}")


if __name__ == "__main__":
    main()
