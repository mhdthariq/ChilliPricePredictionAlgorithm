# LSTM and ARIMA Training & Inference Scripts

This directory contains standalone Python scripts for training and performing inference with LSTM and ARIMA models for chili price prediction.

## Overview

The scripts convert the Jupyter notebook workflows into production-ready Python code that can be run from the command line.

## Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- tensorflow (for LSTM)
- statsmodels (for ARIMA)
- scikit-learn
- matplotlib
- seaborn
- joblib

## Scripts

### 1. `train_lstm.py`
Trains LSTM models (with and without holiday features) for predicting chili prices.

**Usage:**
```bash
python3 train_lstm.py
```

**Output:**
- Model files: `models/lstm/lstm_model_all_markets.h5` and `models/lstm/lstm_holiday_model_all_markets.h5`
- Results: `result/metrics/lstm_summary.pkl` and `result/metrics/lstm_detailed_results.pkl`

**Features:**
- Trains two LSTM models: one without holidays, one with holiday features
- Uses 30-day look-back window
- 50 epochs with batch size 16
- Multivariate approach predicting all 5 markets simultaneously
- Reports RMSE, MAE, and MAPE for each market

### 2. `train_arima.py`
Trains ARIMA and ARIMAX models for predicting chili prices.

**Usage:**
```bash
python3 train_arima.py
```

**Output:**
- Model files: `models/arima/arima_model_*.joblib` and `models/arima/arimax_model_*.joblib`
- Results: `result/metrics/arima_summary.pkl` and `result/metrics/arima_detailed_results.pkl`

**Features:**
- Grid search for optimal (p,d,q) parameters
- Trains both ARIMA (baseline) and ARIMAX (with holidays)
- Separate models for each of 5 markets
- Reports RMSE, MAE, MAPE, and AIC for each model

### 3. `inference.py`
Loads trained models and generates prediction tables.

**Usage:**
```bash
python3 inference.py
```

**Output:**
- CSV files: `result/predictions_*.csv` for each market
- Console output: Comparison tables showing Actual Price | LSTM Price | ARIMA Price

**Table Format:**
```
Date        Actual Price  LSTM Price  ARIMA Price
2025-07-28  30000.0      23749.39    22082.55
2025-07-29  34000.0      23907.30    22082.56
...
```

**Features:**
- Loads pre-trained model results
- Generates comparison tables for all 5 markets
- Calculates error metrics (MAE, MAPE)
- Shows first 10 and last 10 predictions
- Provides overall summary comparing LSTM vs ARIMA

### 4. `visualize_predictions.py`
Generates comparison plots for LSTM and ARIMA predictions.

**Usage:**
```bash
python3 visualize_predictions.py
```

**Output:**
- `result/plots/prediction_comparison_lstm_arima.png` - Individual plots for each market
- `result/plots/prediction_comparison_all_markets.png` - Combined view of all markets
- `result/plots/error_comparison_lstm_arima.png` - Error distribution comparison

**Features:**
- Visualizes Actual vs LSTM vs ARIMA for each market
- Shows prediction trends over time
- Compares error distributions using box plots
- High-resolution PNG output (300 DPI)

## Workflow

The typical workflow is:

1. **Data Preprocessing** (already done in notebooks):
   ```bash
   # Run notebook 01_data_cleaning_and_eda.ipynb
   # This creates processed data and scalers
   ```

2. **Train Models**:
   ```bash
   python3 train_arima.py   # ~5 minutes
   python3 train_lstm.py    # ~45 minutes
   ```

3. **Run Inference**:
   ```bash
   python3 inference.py
   ```

4. **Generate Visualizations**:
   ```bash
   python3 visualize_predictions.py
   ```

## Results Summary

Based on the test set evaluation:

| Model | Avg RMSE | Avg MAPE | Training Time |
|-------|----------|----------|---------------|
| ARIMA | 35,197.02 | 41.21% | ~5 min |
| LSTM (with holidays) | 14,497.90 | 18.02% | ~45 min |

**LSTM Improvement over ARIMA:**
- RMSE: **+58.81%**
- MAPE: **+56.28%**

## File Structure

```
.
├── train_lstm.py              # LSTM training script
├── train_arima.py             # ARIMA training script
├── inference.py               # Inference script
├── visualize_predictions.py   # Visualization script
├── data/
│   ├── processed/
│   │   └── data_with_holidays.csv
│   └── scalers/
│       ├── scaler_markets.joblib
│       └── scaler_with_features.joblib
├── models/
│   ├── lstm/
│   │   ├── lstm_model_all_markets.h5
│   │   └── lstm_holiday_model_all_markets.h5
│   └── arima/
│       ├── arima_model_*.joblib
│       └── arimax_model_*.joblib
└── result/
    ├── metrics/
    │   ├── lstm_summary.pkl
    │   ├── arima_summary.pkl
    │   └── *_detailed_results.pkl
    ├── predictions_*.csv
    └── plots/
        ├── prediction_comparison_lstm_arima.png
        ├── prediction_comparison_all_markets.png
        └── error_comparison_lstm_arima.png
```

## Scientific Paper

See `paper.md` for the complete scientific paper following BITS 2023 template, including:
- Abstract (Indonesian and English)
- Comprehensive methodology
- Results and discussion
- 20 IEEE-formatted references

## Notes

- Models are already trained and results are available in `result/metrics/`
- Training scripts can be re-run to retrain with new data
- Inference and visualization scripts use pre-saved results for faster execution
- All scripts include proper error handling and informative output

## Reproducibility

All scripts are self-contained and use the same random seeds for reproducibility. The exact results can be reproduced by:
1. Using the same data files
2. Running scripts in the specified order
3. Using the same package versions (see requirements.txt)

## Citation

If you use this code, please cite:

```
M. T. Hernanda, "Perbandingan Metode LSTM dan ARIMA dalam Prediksi Harga Cabai Merah 
di Kota Medan," Universitas Sumatera Utara, 2024.
```
