import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import holidays
import os

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load data
def load_data():
    print("Loading data...")
    train = pd.read_csv('train/train.csv', parse_dates=['doj'])
    test = pd.read_csv('test.csv', parse_dates=['doj'])
    transactions = pd.read_csv('train/transactions.csv', 
                             parse_dates=['doj', 'doi'])
    return train, test, transactions

def basic_eda(df, name):
    print(f"\n=== {name} Data ===")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nBasic statistics:")
    print(df.describe(include='all'))

def plot_time_series(train, transactions):
    print("\nPlotting time series...")
    # Plot daily bookings
    daily_bookings = train.groupby('doj')['final_seatcount'].sum().reset_index()
    
    plt.figure(figsize=(15, 6))
    plt.plot(daily_bookings['doj'], daily_bookings['final_seatcount'])
    plt.title('Daily Total Bookings Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Seats Booked')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('daily_bookings.png')
    plt.close()

def analyze_routes(train):
    print("\nAnalyzing routes...")
    # Top routes by demand
    route_demand = train.groupby(['srcid', 'destid'])['final_seatcount'].sum().reset_index()
    route_demand = route_demand.sort_values('final_seatcount', ascending=False)
    
    print("\nTop 10 busiest routes:")
    print(route_demand.head(10))
    
    # Save top routes for feature engineering
    route_demand.to_csv('top_routes.csv', index=False)

def main():
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Load data
    train, test, transactions = load_data()
    
    # Basic EDA
    basic_eda(train, 'Train')
    basic_eda(test, 'Test')
    basic_eda(transactions, 'Transactions')
    
    # Time series analysis
    plot_time_series(train, transactions)
    
    # Route analysis
    analyze_routes(train)
    
    print("\nData exploration completed. Check the output directory for visualizations.")

if __name__ == "__main__":
    main()
