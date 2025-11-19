"""
Visualization Script for Chili Price Prediction
Generates comparison plots with Actual, LSTM, and ARIMA predictions only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

def main():
    print("="*80)
    print("GENERATING PREDICTION COMPARISON PLOTS")
    print("="*80)
    
    # Load results
    print("\nLoading model results...")
    lstm_results = joblib.load('result/metrics/lstm_detailed_results.pkl')
    arima_results = joblib.load('result/metrics/arima_detailed_results.pkl')
    
    market_columns = ['Pasar Sukaramai', 'Pasar Aksara', 'Pasar Petisah', 'Pusat Pasar', 'Pasar Brayan']
    
    # Load data for dates
    df_with_holidays = pd.read_csv('data/processed/data_with_holidays.csv', index_col=0, parse_dates=True)
    TEST_SIZE = 0.2
    SPLIT_INDEX = int(len(df_with_holidays) * (1 - TEST_SIZE))
    LOOK_BACK = 30
    test_data = df_with_holidays.iloc[SPLIT_INDEX:]
    test_dates = test_data.index[LOOK_BACK:LOOK_BACK + len(lstm_results['actual'])]
    
    print(f"✓ Results loaded")
    print(f"  Number of markets: {len(market_columns)}")
    print(f"  Test period: {test_dates[0]} to {test_dates[-1]}")
    
    # Create figure with subplots for each market
    fig, axes = plt.subplots(len(market_columns), 1, figsize=(14, 3 * len(market_columns)))
    fig.suptitle('Price Prediction Comparison - Pasar (Markets)', fontsize=16, fontweight='bold', y=0.995)
    
    if len(market_columns) == 1:
        axes = [axes]
    
    for idx, market in enumerate(market_columns):
        ax = axes[idx]
        
        # Get predictions for this market
        actual = lstm_results['actual'][:, idx]
        lstm_pred = lstm_results['predictions_with_holiday'][:, idx]  # Use best LSTM
        arima_pred = arima_results[market]['arimax_forecast'].values  # Use best ARIMA
        
        # Align lengths
        min_len = min(len(actual), len(lstm_pred), len(arima_pred))
        actual = actual[:min_len]
        lstm_pred = lstm_pred[:min_len]
        arima_pred = arima_pred[:min_len]
        dates = test_dates[:min_len]
        
        # Plot
        ax.plot(dates, actual, 'k-', linewidth=2, label='Actual', alpha=0.8)
        ax.plot(dates, lstm_pred, 'b--', linewidth=1.5, label='LSTM', alpha=0.7)
        ax.plot(dates, arima_pred, 'r:', linewidth=1.5, label='ARIMA', alpha=0.7)
        
        # Formatting
        ax.set_title(f'Price Prediction Comparison - {market}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Price (Rp)', fontsize=10)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'result/plots/prediction_comparison_lstm_arima.png'
    os.makedirs('result/plots', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    
    # Create a single combined plot (all markets on one chart) - alternative view
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, market in enumerate(market_columns):
        actual = lstm_results['actual'][:, idx]
        lstm_pred = lstm_results['predictions_with_holiday'][:, idx]
        
        min_len = min(len(actual), len(lstm_pred))
        actual = actual[:min_len]
        lstm_pred = lstm_pred[:min_len]
        dates = test_dates[:min_len]
        
        # Plot actual as solid line
        ax2.plot(dates, actual, color=colors[idx], linewidth=2, 
                label=f'{market} (Actual)', alpha=0.8)
        # Plot LSTM as dashed line
        ax2.plot(dates, lstm_pred, color=colors[idx], linewidth=1.5, 
                linestyle='--', label=f'{market} (LSTM)', alpha=0.6)
    
    ax2.set_title('Price Prediction Comparison - All Markets (LSTM)', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Price (Rp)', fontsize=12)
    ax2.legend(loc='upper left', framealpha=0.9, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_path2 = 'result/plots/prediction_comparison_all_markets.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Combined plot saved to: {output_path2}")
    
    # Create error comparison plot
    fig3, axes3 = plt.subplots(2, 1, figsize=(14, 8))
    
    # Calculate average errors across all markets
    lstm_errors = []
    arima_errors = []
    
    for idx, market in enumerate(market_columns):
        actual = lstm_results['actual'][:, idx]
        lstm_pred = lstm_results['predictions_with_holiday'][:, idx]
        arima_pred = arima_results[market]['arimax_forecast'].values
        
        min_len = min(len(actual), len(lstm_pred), len(arima_pred))
        actual = actual[:min_len]
        lstm_pred = lstm_pred[:min_len]
        arima_pred = arima_pred[:min_len]
        
        lstm_errors.append(np.abs(actual - lstm_pred))
        arima_errors.append(np.abs(actual - arima_pred))
    
    avg_lstm_error = np.mean(lstm_errors, axis=0)
    avg_arima_error = np.mean(arima_errors, axis=0)
    dates_aligned = test_dates[:len(avg_lstm_error)]
    
    # Plot 1: Absolute errors over time
    axes3[0].plot(dates_aligned, avg_lstm_error, 'b-', linewidth=2, label='LSTM MAE', alpha=0.7)
    axes3[0].plot(dates_aligned, avg_arima_error, 'r-', linewidth=2, label='ARIMA MAE', alpha=0.7)
    axes3[0].set_title('Average Absolute Error Over Time (All Markets)', fontsize=12, fontweight='bold')
    axes3[0].set_xlabel('Date', fontsize=10)
    axes3[0].set_ylabel('Absolute Error (Rp)', fontsize=10)
    axes3[0].legend(loc='upper right')
    axes3[0].grid(True, alpha=0.3)
    axes3[0].tick_params(axis='x', rotation=45)
    axes3[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Plot 2: Error distribution (box plot)
    error_data = [
        avg_lstm_error,
        avg_arima_error
    ]
    
    bp = axes3[1].boxplot(error_data, labels=['LSTM', 'ARIMA'], 
                          patch_artist=True, showmeans=True)
    
    # Color the boxes
    colors_box = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
    
    axes3[1].set_title('Error Distribution Comparison (All Markets)', fontsize=12, fontweight='bold')
    axes3[1].set_ylabel('Absolute Error (Rp)', fontsize=10)
    axes3[1].grid(True, alpha=0.3, axis='y')
    axes3[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    
    output_path3 = 'result/plots/error_comparison_lstm_arima.png'
    plt.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"✓ Error comparison plot saved to: {output_path3}")
    
    print("\n" + "="*80)
    print("✓ All visualization plots generated successfully!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. {output_path}")
    print(f"  2. {output_path2}")
    print(f"  3. {output_path3}")

if __name__ == "__main__":
    main()
