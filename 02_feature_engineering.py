import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import os

def add_date_features(df, date_column):
    """Add basic date-based features"""
    df = df.copy()
    df[f'{date_column}_month'] = df[date_column].dt.month
    df[f'{date_column}_day'] = df[date_column].dt.day
    df[f'{date_column}_weekday'] = df[date_column].dt.weekday
    df[f'{date_column}_week'] = df[date_column].dt.isocalendar().week
    df[f'{date_column}_quarter'] = df[date_column].dt.quarter
    df[f'{date_column}_year'] = df[date_column].dt.year
    df[f'{date_column}_is_weekend'] = df[date_column].dt.weekday >= 5
    return df

def add_holiday_features(df, date_column, country='IN'):
    """Add holiday-related features"""
    df = df.copy()
    years = range(df[date_column].min().year - 1, df[date_column].max().year + 2)
    in_holidays = holidays.country_holidays(country, years=years)
    
    df[f'{date_column}_is_holiday'] = df[date_column].isin(in_holidays)
    
    # Add days to next holiday
    all_dates = pd.date_range(start=df[date_column].min() - timedelta(days=30), 
                            end=df[date_column].max() + timedelta(days=30))
    holidays_list = [d for d in all_dates if d in in_holidays]
    
    def days_to_next_holiday(date):
        next_holidays = [h for h in holidays_list if h >= date]
        return (min(next_holidays) - date).days if next_holidays else np.nan
    
    df[f'{date_column}_days_to_holiday'] = df[date_column].apply(days_to_next_holiday)
    return df

def create_lag_features(transactions, lags=[1, 2, 3, 7, 14, 21, 28]):
    """Create lag features from transaction data"""
    transactions = transactions.sort_values(['srcid', 'destid', 'doj', 'dbd'])
    
    # Create lag features for seatcount and searchcount
    for lag in lags:
        transactions[f'seatcount_lag_{lag}'] = transactions.groupby(['srcid', 'destid', 'dbd'])['cumsum_seatcount'].shift(lag)
        transactions[f'searchcount_lag_{lag}'] = transactions.groupby(['srcid', 'destid', 'dbd'])['cumsum_searchcount'].shift(lag)
    
    return transactions

def create_rolling_features(transactions, windows=[3, 7, 14]):
    """Create rolling window features"""
    transactions = transactions.sort_values(['srcid', 'destid', 'doj', 'dbd'])
    
    for window in windows:
        transactions[f'seatcount_rolling_mean_{window}'] = transactions.groupby(['srcid', 'destid', 'dbd'])['cumsum_seatcount'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        transactions[f'searchcount_rolling_mean_{window}'] = transactions.groupby(['srcid', 'destid', 'dbd'])['cumsum_searchcount'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    return transactions

def prepare_features(train, test, transactions):
    """Prepare features for training and testing"""
    print("Preparing features...")
    
    # Add date features
    train = add_date_features(train, 'doj')
    test = add_date_features(test, 'doj')
    
    # Add holiday features
    train = add_holiday_features(train, 'doj')
    test = add_holiday_features(test, 'doj')
    
    # Process transactions
    transactions = create_lag_features(transactions)
    transactions = create_rolling_features(transactions)
    
    # Filter transactions to 15 days before journey
    transactions_15d = transactions[transactions['dbd'] == 15].copy()
    
    # Merge with train/test data
    train = train.merge(transactions_15d, on=['doj', 'srcid', 'destid'], how='left')
    test = test.merge(transactions_15d, on=['doj', 'srcid', 'destid'], how='left')
    
    return train, test, transactions

def save_features(train, test, output_dir='features'):
    """Save processed features"""
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(f'{output_dir}/train_features.csv', index=False)
    test.to_csv(f'{output_dir}/test_features.csv', index=False)
    print(f"Features saved to {output_dir}/")

def main():
    # Load data
    train = pd.read_csv('train/train.csv', parse_dates=['doj'])
    test = pd.read_csv('test.csv', parse_dates=['doj'])
    transactions = pd.read_csv('train/transactions.csv', 
                             parse_dates=['doj', 'doi'])
    
    # Prepare features
    train, test, _ = prepare_features(train, test, transactions)
    
    # Save features
    save_features(train, test)

if __name__ == "__main__":
    main()