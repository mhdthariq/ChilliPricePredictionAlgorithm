# 🌶️ Chili Price Forecasting - Medan Markets

**Advanced time series forecasting analysis** comparing ARIMA, LSTM, and Prophet algorithms for predicting chili prices across 5 major traditional markets in Medan, Indonesia.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Prophet](https://img.shields.io/badge/Prophet-Latest-green.svg)](https://facebook.github.io/prophet/)

---

## 📊 Executive Summary

### Best Model: **LSTM (Baseline)**

- ✅ **RMSE:** 11,933
- ✅ **MAPE:** 13.76% (Excellent category)
- ✅ **Performance:** 66% better than ARIMA, 80% better than Prophet
- ✅ **Consistency:** Best in all 5 markets

### Key Findings

1. **Deep learning essential** for volatile commodity forecasting (CV=40%)
2. **Holiday features NOT needed** for LSTM (already captures patterns)
3. **ARIMA struggles** with non-linear price movements (MAPE 41%)
4. **Prophet fails** on volatile commodities (MAPE 70% - designed for smooth business metrics)

---

## 🚀 Quick Start

### Run Complete Pipeline

```bash
# Navigate to notebooks and execute in order:
cd notebooks/
jupyter notebook 01_data_cleaning_and_eda.ipynb
jupyter notebook 02_arima_modeling.ipynb
jupyter notebook 03_lstm_modeling.ipynb
jupyter notebook 04_prophet_modeling.ipynb
jupyter notebook 05_model_comparison_and_inference.ipynb
```

### View Results

- **📄 Full Report:** [LAPORAN_FINAL.md](LAPORAN_FINAL.md) (8,500 words)
- **📊 Comparison:** `results/metrics/model_comparison.csv`
- **📈 Visualizations:** `results/plots/`

---

## 📈 Results

| Rank  | Model               | RMSE       | MAPE (%)  | Category         |
| ----- | ------------------- | ---------- | --------- | ---------------- |
| **1** | **LSTM (baseline)** | **11,933** | **13.76** | **Excellent ⭐** |
| 2     | LSTM + Holiday      | 14,498     | 18.02     | Good             |
| 3     | ARIMA               | 35,197     | 41.21     | Poor             |
| 4     | Prophet + Holiday   | 49,684     | 69.94     | Very Poor        |
| 5     | Prophet (baseline)  | 51,090     | 73.90     | Very Poor        |

**Winner:** LSTM (baseline) - 66% better than ARIMA, 80% better than Prophet

---

## 📚 Full Documentation

See **[LAPORAN_FINAL.md](LAPORAN_FINAL.md)** for complete:

- Methodology
- Data analysis
- Model architectures
- Statistical tests
- Business recommendations
