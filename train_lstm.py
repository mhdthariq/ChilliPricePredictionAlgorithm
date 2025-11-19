"""
LSTM Model Training Script for Chili Price Prediction
This script trains LSTM models (with and without holiday features) for predicting chili prices
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import sys

# Ensure directories exist
os.makedirs('models/lstm', exist_ok=True)
os.makedirs('result/metrics', exist_ok=True)

def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error"""
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return min(mape, 999.99)

def main():
    print("="*80)
    print("LSTM MODEL TRAINING - Chili Price Prediction")
    print("="*80)
    
    # Load preprocessed data
    print("\nLoading data...")
    df_with_holidays = pd.read_csv('data/processed/data_with_holidays.csv', index_col=0, parse_dates=True)
    print(f"✓ Data loaded: {df_with_holidays.shape}")
    print(f"  Date range: {df_with_holidays.index.min()} to {df_with_holidays.index.max()}")
    
    # Define parameters
    market_columns = ['Pasar Sukaramai', 'Pasar Aksara', 'Pasar Petisah', 'Pusat Pasar', 'Pasar Brayan']
    TEST_SIZE = 0.2
    LOOK_BACK = 30  # Use 30 days of history
    SPLIT_INDEX = int(len(df_with_holidays) * (1 - TEST_SIZE))
    EPOCHS = 50
    BATCH_SIZE = 16
    
    print(f"\nModel Parameters:")
    print(f"  Markets: {len(market_columns)}")
    print(f"  Look-back window: {LOOK_BACK} days")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # Split data chronologically
    train_data = df_with_holidays.iloc[:SPLIT_INDEX]
    test_data = df_with_holidays.iloc[SPLIT_INDEX:]
    
    print(f"\nData Split:")
    print(f"  Training: {train_data.shape[0]} days ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"  Testing: {test_data.shape[0]} days ({test_data.index[0]} to {test_data.index[-1]})")
    
    # Load the scalers created in data preprocessing
    print("\nLoading scalers...")
    scaler_markets = joblib.load('data/scalers/scaler_markets.joblib')
    scaler_with_features = joblib.load('data/scalers/scaler_with_features.joblib')
    print("✓ Scalers loaded successfully")
    
    # Scale the data
    print("\nScaling data...")
    train_markets_scaled = scaler_markets.transform(train_data[market_columns])
    test_markets_scaled = scaler_markets.transform(test_data[market_columns])
    
    feature_columns = market_columns + ['is_holiday']
    train_features_scaled = scaler_with_features.transform(train_data[feature_columns])
    test_features_scaled = scaler_with_features.transform(test_data[feature_columns])
    print("✓ Data scaled successfully")
    
    # ===========================
    # Model 1: LSTM without holidays
    # ===========================
    print("\n" + "="*70)
    print("Training Model 1: LSTM without holidays (Markets only)")
    print("="*70)
    
    data_no_holiday = train_markets_scaled
    test_data_nh = test_markets_scaled
    n_features_nh = data_no_holiday.shape[1]
    
    # Create time series generators
    train_generator_nh = TimeseriesGenerator(
        data_no_holiday, 
        data_no_holiday,
        length=LOOK_BACK,
        batch_size=BATCH_SIZE
    )
    
    test_generator_nh = TimeseriesGenerator(
        test_data_nh,
        test_data_nh,
        length=LOOK_BACK,
        batch_size=BATCH_SIZE
    )
    
    # Build LSTM model
    lstm_model = Sequential([
        LSTM(64, activation='relu', input_shape=(LOOK_BACK, n_features_nh), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(n_features_nh)
    ])
    
    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("\nModel architecture created")
    
    # Train the model
    print("Training model...")
    history = lstm_model.fit(
        train_generator_nh,
        epochs=EPOCHS,
        verbose=2,
        validation_data=test_generator_nh
    )
    
    # Make predictions
    print("\nMaking predictions...")
    lstm_predictions = []
    for i in range(len(test_generator_nh)):
        X, _ = test_generator_nh[i]
        pred = lstm_model.predict(X, verbose=0)
        lstm_predictions.extend(pred)
    
    lstm_predictions = np.array(lstm_predictions)
    
    # Inverse transform
    lstm_pred = scaler_markets.inverse_transform(lstm_predictions)
    actual_test_nh = test_data_nh[LOOK_BACK:LOOK_BACK + len(lstm_pred)]
    y_test = scaler_markets.inverse_transform(actual_test_nh)
    
    # Calculate metrics
    print("\n" + "="*50)
    print("LSTM (no holidays) - Metrics by Market:")
    print("="*50)
    lstm_rmse_list = []
    lstm_mae_list = []
    lstm_mape_list = []
    
    for idx, market in enumerate(market_columns):
        rmse = np.sqrt(mean_squared_error(y_test[:, idx], lstm_pred[:, idx]))
        mae = mean_absolute_error(y_test[:, idx], lstm_pred[:, idx])
        mape = calculate_mape(y_test[:, idx], lstm_pred[:, idx])
        
        lstm_rmse_list.append(rmse)
        lstm_mae_list.append(mae)
        lstm_mape_list.append(mape)
        
        print(f"{market:25s}: RMSE={rmse:8.2f}, MAE={mae:8.2f}, MAPE={mape:6.2f}%")
    
    avg_lstm = np.mean(lstm_rmse_list)
    print(f"\nAverage RMSE: {avg_lstm:.2f}")
    
    # Save the model
    lstm_model.save('models/lstm/lstm_model_all_markets.h5')
    print("✓ Model saved to: models/lstm/lstm_model_all_markets.h5")
    
    # ===========================
    # Model 2: LSTM with holidays
    # ===========================
    print("\n" + "="*70)
    print("Training Model 2: LSTM with holidays (Markets + Holiday feature)")
    print("="*70)
    
    data_with_holiday = train_features_scaled
    test_data_wh = test_features_scaled
    n_features_wh = data_with_holiday.shape[1]
    
    # Create time series generators
    train_generator_wh = TimeseriesGenerator(
        data_with_holiday,
        data_with_holiday,
        length=LOOK_BACK,
        batch_size=BATCH_SIZE
    )
    
    test_generator_wh = TimeseriesGenerator(
        test_data_wh,
        test_data_wh,
        length=LOOK_BACK,
        batch_size=BATCH_SIZE
    )
    
    # Build LSTM model with holiday
    lstm_holiday_model = Sequential([
        LSTM(64, activation='relu', input_shape=(LOOK_BACK, n_features_wh), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(n_features_wh)
    ])
    
    lstm_holiday_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("\nModel architecture created")
    
    # Train the model
    print("Training model...")
    history_h = lstm_holiday_model.fit(
        train_generator_wh,
        epochs=EPOCHS,
        verbose=2,
        validation_data=test_generator_wh
    )
    
    # Make predictions
    print("\nMaking predictions...")
    lstm_holiday_predictions = []
    for i in range(len(test_generator_wh)):
        X, _ = test_generator_wh[i]
        pred = lstm_holiday_model.predict(X, verbose=0)
        lstm_holiday_predictions.extend(pred)
    
    lstm_holiday_predictions = np.array(lstm_holiday_predictions)
    
    # Inverse transform - only take the first 5 columns (markets)
    lstm_holiday_pred_all = scaler_with_features.inverse_transform(lstm_holiday_predictions)
    lstm_holiday_pred = lstm_holiday_pred_all[:, :5]
    
    actual_test_wh = test_data_wh[LOOK_BACK:LOOK_BACK + len(lstm_holiday_pred)]
    y_test_h_all = scaler_with_features.inverse_transform(actual_test_wh)
    y_test_h = y_test_h_all[:, :5]
    
    # Calculate metrics
    print("\n" + "="*50)
    print("LSTM (with holidays) - Metrics by Market:")
    print("="*50)
    lstm_h_rmse_list = []
    lstm_h_mae_list = []
    lstm_h_mape_list = []
    
    for idx, market in enumerate(market_columns):
        rmse = np.sqrt(mean_squared_error(y_test_h[:, idx], lstm_holiday_pred[:, idx]))
        mae = mean_absolute_error(y_test_h[:, idx], lstm_holiday_pred[:, idx])
        mape = calculate_mape(y_test_h[:, idx], lstm_holiday_pred[:, idx])
        
        lstm_h_rmse_list.append(rmse)
        lstm_h_mae_list.append(mae)
        lstm_h_mape_list.append(mape)
        
        print(f"{market:25s}: RMSE={rmse:8.2f}, MAE={mae:8.2f}, MAPE={mape:6.2f}%")
    
    avg_lstm_h = np.mean(lstm_h_rmse_list)
    print(f"\nAverage RMSE: {avg_lstm_h:.2f}")
    
    # Save the model
    lstm_holiday_model.save('models/lstm/lstm_holiday_model_all_markets.h5')
    print("✓ Model saved to: models/lstm/lstm_holiday_model_all_markets.h5")
    
    # Store results
    lstm_results = {
        'predictions_no_holiday': lstm_pred,
        'predictions_with_holiday': lstm_holiday_pred,
        'actual': y_test,
        'test_dates': test_data.index[LOOK_BACK:LOOK_BACK + len(lstm_pred)],
        'rmse_no_holiday': lstm_rmse_list,
        'mae_no_holiday': lstm_mae_list,
        'mape_no_holiday': lstm_mape_list,
        'rmse_with_holiday': lstm_h_rmse_list,
        'mae_with_holiday': lstm_h_mae_list,
        'mape_with_holiday': lstm_h_mape_list,
        'avg_rmse_no_holiday': avg_lstm,
        'avg_rmse_with_holiday': avg_lstm_h
    }
    
    # Final summary
    avg_mape_no_holiday = np.mean(lstm_results['mape_no_holiday'])
    avg_mape_with_holiday = np.mean(lstm_results['mape_with_holiday'])
    
    print("\n" + "="*70)
    print("LSTM TRAINING COMPLETE - FINAL SUMMARY")
    print("="*70)
    print(f"LSTM (no holidays):")
    print(f"  Average RMSE: {lstm_results['avg_rmse_no_holiday']:,.2f}")
    print(f"  Average MAPE: {avg_mape_no_holiday:.2f}%")
    print(f"\nLSTM (with holidays):")
    print(f"  Average RMSE: {lstm_results['avg_rmse_with_holiday']:,.2f}")
    print(f"  Average MAPE: {avg_mape_with_holiday:.2f}%")
    
    improvement = ((lstm_results['avg_rmse_no_holiday'] - lstm_results['avg_rmse_with_holiday']) / lstm_results['avg_rmse_no_holiday']) * 100
    print(f"\nHoliday feature improvement: {improvement:+.2f}%")
    
    # Determine best model
    best_model = "with holidays" if avg_lstm_h < avg_lstm else "without holidays"
    print(f"\n✓ Best LSTM model: {best_model}")
    print("="*70)
    
    # Save results
    lstm_summary = {
        'algorithm': 'LSTM',
        'avg_rmse_no_holiday': lstm_results['avg_rmse_no_holiday'],
        'avg_rmse_with_holiday': lstm_results['avg_rmse_with_holiday'],
        'avg_mape_no_holiday': avg_mape_no_holiday,
        'avg_mape_with_holiday': avg_mape_with_holiday,
        'markets': market_columns,
        'best_model': best_model,
        'results': lstm_results
    }
    
    joblib.dump(lstm_summary, 'result/metrics/lstm_summary.pkl')
    joblib.dump(lstm_results, 'result/metrics/lstm_detailed_results.pkl')
    
    print('\n✓ Results saved to result/metrics/')
    print('✓ LSTM training completed successfully!')

if __name__ == "__main__":
    main()
