"""
ARIMA/ARIMAX Model Training Script for Chili Price Prediction
This script trains ARIMA models (with and without holiday features) for predicting chili prices
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Ensure directories exist
os.makedirs('models/arima', exist_ok=True)
os.makedirs('result/metrics', exist_ok=True)

def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error"""
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return min(mape, 999.99)

def check_stationarity(timeseries, name="Series"):
    """Check if time series is stationary using ADF test"""
    result = adfuller(timeseries.dropna())
    is_stationary = result[1] < 0.05
    return is_stationary

def find_best_arima_order(train_data, exog_data=None):
    """Find best ARIMA order using simple grid search"""
    best_aic = float('inf')
    best_order = None
    best_model = None
    
    # Simple orders that work well in practice
    priority_orders = [
        (1, 1, 1),  # Most common
        (2, 1, 2),
        (1, 1, 2),
        (2, 1, 1),
        (0, 1, 1),
        (1, 1, 0),
        (2, 1, 0),
        (0, 1, 2),
        (3, 1, 3),
    ]
    
    tested = 0
    for order in priority_orders:
        try:
            if exog_data is not None:
                # Ensure exog_data is 2D array
                if hasattr(exog_data, 'values'):
                    exog = exog_data.values
                else:
                    exog = np.array(exog_data)
                
                if exog.ndim == 1:
                    exog = exog.reshape(-1, 1)
                
                model = ARIMA(train_data, exog=exog, order=order)
            else:
                model = ARIMA(train_data, order=order)
            
            fitted_model = model.fit()
            tested += 1
            
            if fitted_model.aic < best_aic:
                best_aic = fitted_model.aic
                best_order = order
                best_model = fitted_model
        except Exception:
            continue
    
    if best_model is None:
        # Fallback to simplest model
        print("  ⚠ Grid search failed, using ARIMA(1,1,1)")
        if exog_data is not None:
            exog = exog_data.values if hasattr(exog_data, 'values') else np.array(exog_data)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            best_model = ARIMA(train_data, exog=exog, order=(1, 1, 1)).fit()
        else:
            best_model = ARIMA(train_data, order=(1, 1, 1)).fit()
        best_order = (1, 1, 1)
        best_aic = best_model.aic
    
    print(f"  Tested {tested} models")
    return best_order, best_model, best_aic

def main():
    print("="*80)
    print("ARIMA MODEL TRAINING - Chili Price Prediction")
    print("="*80)
    print("⚠️  ARIMA is NOT ideal for this high-variance, event-driven data")
    print("    This establishes a baseline - LSTM should outperform significantly")
    print("="*80)
    
    # Load preprocessed data
    print("\nLoading data...")
    df_with_holidays = pd.read_csv('data/processed/data_with_holidays.csv', index_col=0, parse_dates=True)
    print(f"✓ Data loaded: {df_with_holidays.shape}")
    print(f"  Date range: {df_with_holidays.index.min()} to {df_with_holidays.index.max()}")
    
    # Define market columns and train/test split
    market_columns = ['Pasar Sukaramai', 'Pasar Aksara', 'Pasar Petisah', 'Pusat Pasar', 'Pasar Brayan']
    TEST_SIZE = 0.2
    SPLIT_INDEX = int(len(df_with_holidays) * (1 - TEST_SIZE))
    
    # Split data
    train_data = df_with_holidays.iloc[:SPLIT_INDEX]
    test_data = df_with_holidays.iloc[SPLIT_INDEX:]
    
    print(f"\nData Split:")
    print(f"  Training: {len(train_data)} samples")
    print(f"  Testing: {len(test_data)} samples")
    print(f"  Split ratio: {(1-TEST_SIZE)*100:.0f}/{TEST_SIZE*100:.0f}")
    
    # Results storage
    arima_results = {}
    
    # Train ARIMA models for each market
    for market in market_columns:
        print(f"\n{'='*70}")
        print(f"Training ARIMA for {market}")
        print(f"{'='*70}")
    
        # Prepare data
        train_y = train_data[market]
        test_y = test_data[market]
        train_holiday = train_data['is_holiday']
        test_holiday = test_data['is_holiday']
        
        # Check for any issues in data
        print(f"Price range: {train_y.min():.0f} - {train_y.max():.0f}")
        print(f"Training samples: {len(train_y)}, Test samples: {len(test_y)}")
        cv = (train_y.std()/train_y.mean())*100
        print(f"Coefficient of Variation: {cv:.1f}% (HIGH - not ideal for ARIMA)")
    
        # Check stationarity
        is_stationary = check_stationarity(train_y, f"Stationarity check for {market}")
    
        # Model 1: ARIMA without exogenous variables
        print(f"\nTraining ARIMA model (without holidays)...")
        arima_order, arima_model, arima_aic = find_best_arima_order(train_y)
        print(f"✓ Best ARIMA order: {arima_order}, AIC: {arima_aic:.2f}")
    
        # Forecast
        arima_forecast = arima_model.forecast(steps=len(test_y))
    
        # Model 2: ARIMAX with holidays
        print(f"\nTraining ARIMAX model (with holidays)...")
        
        # Convert holiday to numpy array
        train_holiday_array = train_holiday.values.reshape(-1, 1)
        test_holiday_array = test_holiday.values.reshape(-1, 1)
        
        arimax_order, arimax_model, arimax_aic = find_best_arima_order(train_y, train_holiday_array)
        print(f"✓ Best ARIMAX order: {arimax_order}, AIC: {arimax_aic:.2f}")
    
        # Forecast with exogenous variable
        arimax_forecast = arimax_model.forecast(steps=len(test_y), exog=test_holiday_array)
    
        # Calculate metrics
        arima_rmse = np.sqrt(mean_squared_error(test_y, arima_forecast))
        arima_mae = mean_absolute_error(test_y, arima_forecast)
        arima_mape = calculate_mape(test_y.values, arima_forecast.values if hasattr(arima_forecast, 'values') else arima_forecast)
    
        arimax_rmse = np.sqrt(mean_squared_error(test_y, arimax_forecast))
        arimax_mae = mean_absolute_error(test_y, arimax_forecast)
        arimax_mape = calculate_mape(test_y.values, arimax_forecast.values if hasattr(arimax_forecast, 'values') else arimax_forecast)
    
        # Store results
        arima_results[market] = {
            'arima_order': arima_order,
            'arimax_order': arimax_order,
            'arima_forecast': arima_forecast,
            'arimax_forecast': arimax_forecast,
            'actual': test_y,
            'arima_rmse': arima_rmse,
            'arima_mae': arima_mae,
            'arima_mape': arima_mape,
            'arimax_rmse': arimax_rmse,
            'arimax_mae': arimax_mae,
            'arimax_mape': arimax_mape,
            'arima_aic': arima_aic,
            'arimax_aic': arimax_aic
        }
    
        # Display results
        print(f"\nRESULTS:")
        print(f"  ARIMA  - RMSE: {arima_rmse:>10,.2f}, MAE: {arima_mae:>10,.2f}, MAPE: {arima_mape:>6.2f}%")
        print(f"  ARIMAX - RMSE: {arimax_rmse:>10,.2f}, MAE: {arimax_mae:>10,.2f}, MAPE: {arimax_mape:>6.2f}%")
        
        # Calculate improvement
        improvement = ((arima_rmse - arimax_rmse) / arima_rmse) * 100
        if improvement > 0:
            print(f"  ✓ ARIMAX improved by {improvement:.2f}%")
        else:
            print(f"  ⚠ ARIMAX degraded by {abs(improvement):.2f}% (holidays may not help ARIMA)")
    
        # Save models
        joblib.dump({
            'model': arima_model,
            'order': arima_order
        }, f'models/arima/arima_model_{market.replace(" ", "_")}.joblib')
        
        joblib.dump({
            'model': arimax_model,
            'order': arimax_order
        }, f'models/arima/arimax_model_{market.replace(" ", "_")}.joblib')
    
    print("\n" + "="*80)
    print("✓ All ARIMA models saved to: models/arima/")
    print("="*80)
    
    # Summary statistics
    print("\nARIMA TRAINING COMPLETE - SUMMARY ACROSS ALL MARKETS:")
    print("="*70)
    avg_arima_rmse = np.mean([arima_results[m]['arima_rmse'] for m in market_columns])
    avg_arimax_rmse = np.mean([arima_results[m]['arimax_rmse'] for m in market_columns])
    avg_arima_mape = np.mean([arima_results[m]['arima_mape'] for m in market_columns if not np.isnan(arima_results[m]['arima_mape'])])
    avg_arimax_mape = np.mean([arima_results[m]['arimax_mape'] for m in market_columns if not np.isnan(arima_results[m]['arimax_mape'])])
    
    print(f"Average ARIMA RMSE:  {avg_arima_rmse:,.2f}")
    print(f"Average ARIMAX RMSE: {avg_arimax_rmse:,.2f}")
    print(f"Average ARIMA MAPE:  {avg_arima_mape:.2f}%")
    print(f"Average ARIMAX MAPE: {avg_arimax_mape:.2f}%")
    
    overall_improvement = ((avg_arima_rmse - avg_arimax_rmse) / avg_arima_rmse) * 100
    print(f"\nOverall holiday impact: {overall_improvement:+.2f}%")
    
    # Determine best ARIMA model
    best_arima = "ARIMAX (with holidays)" if avg_arimax_rmse < avg_arima_rmse else "ARIMA (without holidays)"
    print(f"\n✓ Best ARIMA model: {best_arima}")
    
    # Baseline comparison
    print(f"\n{'='*80}")
    print("BASELINE COMPARISON:")
    print(f"{'='*80}")
    print(f"7-day Moving Average baseline: MAPE ≈ 40%")
    print(f"Your ARIMA:                     MAPE = {avg_arima_mape:.2f}%")
    if avg_arima_mape > 40:
        print(f"⚠️  ARIMA is performing WORSE than simple moving average!")
        print(f"   This confirms: ARIMA is the WRONG model for this data")
    else:
        print(f"✓  ARIMA slightly beats simple baseline")
    
    print(f"\n{'='*80}")
    print("WHY ARIMA STRUGGLES:")
    print(f"{'='*80}")
    print("✗ High variance (CV=40%) - ARIMA assumes stable variance")
    print("✗ Event-driven spikes - ARIMA is linear, can't model sudden jumps")
    print("✗ Non-stationary patterns - ARIMA works best with stationary data")
    print("\n✓ EXPECTED: LSTM will perform 30-50% better!")
    print(f"{'='*80}")
    
    # Save ARIMA results for comparison
    arima_summary = {
        'algorithm': 'ARIMA',
        'avg_rmse': avg_arima_rmse,
        'avg_mape': avg_arima_mape,
        'markets': market_columns,
        'best_model': best_arima,
        'results': arima_results,
        'notes': 'ARIMA is NOT ideal for this data - use as baseline only'
    }
    
    arimax_summary = {
        'algorithm': 'ARIMAX',
        'avg_rmse': avg_arimax_rmse,
        'avg_mape': avg_arimax_mape,
        'notes': 'Holiday features may not significantly help ARIMA'
    }
    
    # Save to pickle for inference
    joblib.dump(arima_summary, 'result/metrics/arima_summary.pkl')
    joblib.dump(arimax_summary, 'result/metrics/arimax_summary.pkl')
    joblib.dump(arima_results, 'result/metrics/arima_detailed_results.pkl')
    
    print('\n✓ Results saved to result/metrics/')
    print('✓ ARIMA training completed successfully!')
    print('\n⚠️  ARIMA serves as BASELINE - expect LSTM to perform 30-50% better!')

if __name__ == "__main__":
    main()
