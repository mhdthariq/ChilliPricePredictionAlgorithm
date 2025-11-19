"""
Inference Script for Chili Price Prediction
This script loads trained LSTM and ARIMA models and generates predictions in table format
Output: Actual Price | LSTM Price | ARIMA Price
"""

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import os
import sys

def main():
    print("="*80)
    print("CHILI PRICE PREDICTION - INFERENCE")
    print("="*80)
    
    # Check if models exist
    lstm_model_path = 'models/lstm/lstm_holiday_model_all_markets.h5'
    if not os.path.exists(lstm_model_path):
        print(f"\n❌ Error: LSTM model not found at {lstm_model_path}")
        print("Please run train_lstm.py first to train the model.")
        sys.exit(1)
    
    # Load data
    print("\nLoading data...")
    df_with_holidays = pd.read_csv('data/processed/data_with_holidays.csv', index_col=0, parse_dates=True)
    market_columns = ['Pasar Sukaramai', 'Pasar Aksara', 'Pasar Petisah', 'Pusat Pasar', 'Pasar Brayan']
    
    # Load saved results (contains predictions on test set)
    print("Loading model results...")
    lstm_results = joblib.load('result/metrics/lstm_detailed_results.pkl')
    arima_results = joblib.load('result/metrics/arima_detailed_results.pkl')
    
    # Parameters
    TEST_SIZE = 0.2
    SPLIT_INDEX = int(len(df_with_holidays) * (1 - TEST_SIZE))
    LOOK_BACK = 30
    
    test_data = df_with_holidays.iloc[SPLIT_INDEX:]
    test_dates = test_data.index[LOOK_BACK:LOOK_BACK + len(lstm_results['actual'])]
    
    print(f"\n✓ Data loaded")
    print(f"  Test period: {test_dates[0]} to {test_dates[-1]}")
    print(f"  Number of predictions: {len(test_dates)}")
    
    # Create comparison tables for each market
    print("\n" + "="*80)
    print("PREDICTION COMPARISON TABLES")
    print("="*80)
    
    for idx, market in enumerate(market_columns):
        print(f"\n{'='*80}")
        print(f"Market: {market}")
        print(f"{'='*80}")
        
        # Get predictions for this market
        actual = lstm_results['actual'][:, idx]
        lstm_pred = lstm_results['predictions_with_holiday'][:, idx]  # Use best LSTM (with holidays)
        
        # Get ARIMA predictions for this market
        arima_pred = arima_results[market]['arimax_forecast'].values  # Use ARIMAX (with holidays)
        
        # Align lengths (ARIMA may have different length)
        min_len = min(len(actual), len(lstm_pred), len(arima_pred))
        actual = actual[:min_len]
        lstm_pred = lstm_pred[:min_len]
        arima_pred = arima_pred[:min_len]
        dates = test_dates[:min_len]
        
        # Create DataFrame
        comparison_df = pd.DataFrame({
            'Date': dates,
            'Actual Price': actual,
            'LSTM Price': lstm_pred,
            'ARIMA Price': arima_pred
        })
        
        # Calculate errors
        comparison_df['LSTM Error'] = comparison_df['LSTM Price'] - comparison_df['Actual Price']
        comparison_df['ARIMA Error'] = comparison_df['ARIMA Price'] - comparison_df['Actual Price']
        comparison_df['LSTM Error %'] = (comparison_df['LSTM Error'] / comparison_df['Actual Price']) * 100
        comparison_df['ARIMA Error %'] = (comparison_df['ARIMA Error'] / comparison_df['Actual Price']) * 100
        
        # Display first 10 rows
        print("\nFirst 10 predictions:")
        print(comparison_df[['Date', 'Actual Price', 'LSTM Price', 'ARIMA Price']].head(10).to_string(index=False))
        
        # Display last 10 rows
        print("\nLast 10 predictions:")
        print(comparison_df[['Date', 'Actual Price', 'LSTM Price', 'ARIMA Price']].tail(10).to_string(index=False))
        
        # Summary statistics
        print(f"\n{'='*60}")
        print(f"Summary Statistics for {market}:")
        print(f"{'='*60}")
        print(f"{'Metric':<25} {'Actual':>12} {'LSTM':>12} {'ARIMA':>12}")
        print("-"*60)
        print(f"{'Mean Price (Rp)':<25} {actual.mean():>12,.0f} {lstm_pred.mean():>12,.0f} {arima_pred.mean():>12,.0f}")
        print(f"{'Std Dev (Rp)':<25} {actual.std():>12,.0f} {lstm_pred.std():>12,.0f} {arima_pred.std():>12,.0f}")
        print(f"{'Min Price (Rp)':<25} {actual.min():>12,.0f} {lstm_pred.min():>12,.0f} {arima_pred.min():>12,.0f}")
        print(f"{'Max Price (Rp)':<25} {actual.max():>12,.0f} {lstm_pred.max():>12,.0f} {arima_pred.max():>12,.0f}")
        
        # Error metrics
        lstm_mae = np.mean(np.abs(comparison_df['LSTM Error']))
        arima_mae = np.mean(np.abs(comparison_df['ARIMA Error']))
        lstm_mape = np.mean(np.abs(comparison_df['LSTM Error %']))
        arima_mape = np.mean(np.abs(comparison_df['ARIMA Error %']))
        
        print(f"\n{'Error Metrics':<25} {'LSTM':>12} {'ARIMA':>12}")
        print("-"*60)
        print(f"{'MAE (Rp)':<25} {lstm_mae:>12,.2f} {arima_mae:>12,.2f}")
        print(f"{'MAPE (%)':<25} {lstm_mape:>12,.2f} {arima_mape:>12,.2f}")
        
        # Save to CSV
        csv_filename = f'result/predictions_{market.replace(" ", "_")}.csv'
        comparison_df.to_csv(csv_filename, index=False)
        print(f"\n✓ Full predictions saved to: {csv_filename}")
    
    # Overall summary across all markets
    print("\n" + "="*80)
    print("OVERALL SUMMARY - ALL MARKETS")
    print("="*80)
    
    # Load summaries
    lstm_summary = joblib.load('result/metrics/lstm_summary.pkl')
    arima_summary = joblib.load('result/metrics/arima_summary.pkl')
    
    print(f"\n{'Model':<30} {'Avg RMSE':>15} {'Avg MAPE':>15}")
    print("-"*60)
    print(f"{'LSTM (with holidays)':<30} {lstm_summary['avg_rmse_with_holiday']:>15,.2f} {lstm_summary['avg_mape_with_holiday']:>14,.2f}%")
    print(f"{'ARIMA':<30} {arima_summary['avg_rmse']:>15,.2f} {arima_summary['avg_mape']:>14,.2f}%")
    
    # Calculate improvement
    rmse_improvement = ((arima_summary['avg_rmse'] - lstm_summary['avg_rmse_with_holiday']) / arima_summary['avg_rmse']) * 100
    mape_improvement = ((arima_summary['avg_mape'] - lstm_summary['avg_mape_with_holiday']) / arima_summary['avg_mape']) * 100
    
    print(f"\n{'='*80}")
    print(f"LSTM improves over ARIMA by:")
    print(f"  RMSE: {rmse_improvement:+.2f}%")
    print(f"  MAPE: {mape_improvement:+.2f}%")
    print(f"{'='*80}")
    
    print("\n✓ Inference completed successfully!")
    print(f"\n📊 Prediction tables saved to result/ directory")
    print(f"   - One CSV file per market with format: Actual | LSTM | ARIMA")

if __name__ == "__main__":
    main()
