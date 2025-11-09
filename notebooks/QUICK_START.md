# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Data Preparation (Required)

```bash
# Open and run all cells in:
notebooks/01_data_cleaning_and_eda.ipynb
```

**Time:** ~2-3 minutes  
**Output:** Cleaned data in `data/processed/`

### Step 2: Train Models (Choose your algorithm)

**Option A: Train All Models**

```bash
# Run in order:
notebooks/02_arima_modeling.ipynb     # ~5-10 min (log transformation included!)
notebooks/03_lstm_modeling.ipynb      # ~10-15 min
notebooks/04_prophet_modeling.ipynb   # ~5-10 min
```

**Option B: Train Just ARIMA (Fastest)**

```bash
# Run only:
notebooks/02_arima_modeling.ipynb     # ~5-10 min
```

### Step 3: View Results

```bash
# Run:
notebooks/05_model_comparison_and_inference.ipynb
```

**Time:** ~1-2 minutes  
**Output:** Comparison charts, statistical tests, executive summary

---

## 📊 What to Expect

### Notebook 01 Output:

```
✓ Data loaded successfully
✓ Cleaned data saved to: data/processed/cleaned_data.csv
✓ Holiday features added
✓ EDA visualizations generated
```

### Notebook 02 Output (ARIMA):

```
✓ Using grid search for ARIMA order selection
✓ Added log transformation to stabilize variance

Training ARIMA for Pasar Sukaramai...
Original price range: 18000.00 - 89000.00
Log-transformed range: 9.7980 - 11.3959
✓ Best ARIMA order: (2, 1, 3), AIC: 7506.75

RESULTS (on original price scale):
  ARIMA  - RMSE:   12,587.17    ← Should be ~10-15k!
  ARIMAX - RMSE:   11,234.56    ← Not 32k anymore!
```

### Notebook 05 Output (Comparison):

```
MODEL PERFORMANCE COMPARISON
Rank  Algorithm           Avg RMSE    Avg MAPE (%)
1     LSTM+Holiday        9,234.56    14.23
2     ARIMA (log)         12,456.78   18.45
3     Prophet+Holiday     13,123.45   19.67

✓ Statistical significance testing complete
✓ Charts saved to results/plots/
```

---

## ⚡ Quick Checks

### Is ARIMA working correctly?

**Check 1:** Cell 19 output should show:

```
✓ ARIMA helper functions defined successfully
✓ Added log transformation to stabilize variance  ← This line is KEY!
```

**Check 2:** Cell 20 training output should show:

```
Original price range: 18000.00 - 89000.00
Log-transformed range: 9.7980 - 11.3959  ← Log transform applied!
```

**Check 3:** Results should show:

```
Average ARIMA RMSE:  10,000-15,000  ← Good!
NOT: 32,000  ← Bad, means log transform not applied
```

### Is LSTM working correctly?

**Check:** Should create only **2 model files**:

```
models/lstm/lstm_model_no_holiday.h5      ← 1 model for all markets
models/lstm/lstm_model_with_holiday.h5    ← 1 model for all markets
```

**NOT:** 10 files (that was the old bug)

---

## 🎯 Expected Performance

| Model            | RMSE Target | MAPE Target | Training Time |
| ---------------- | ----------- | ----------- | ------------- |
| ARIMA (with log) | 10-15k      | 15-25%      | 5-10 min      |
| LSTM             | 8-12k       | 12-20%      | 10-15 min     |
| Prophet          | 9-14k       | 14-22%      | 5-10 min      |

---

## 🐛 Common Issues

### Issue 1: "File not found: data/processed/..."

**Solution:** Run notebook 01 first!

### Issue 2: ARIMA RMSE still ~32k

**Solution:**

1. Check Cell 19 has log transformation functions
2. Check Cell 20 uses `transform_prices()`
3. Restart kernel and run both cells again

### Issue 3: "KeyError: 'transform_params'"

**Solution:** Retrain ARIMA models (Cell 20) - old models missing transform params

### Issue 4: Comparison notebook errors

**Solution:** Train at least one model (notebooks 02-04) before running notebook 05

---

## 📁 Where Are My Results?

After running all notebooks, find results here:

```
results/
├── metrics/
│   ├── model_comparison.csv          ← Performance table
│   ├── executive_summary.csv         ← Key findings
│   ├── arima_summary.pkl             ← ARIMA results
│   ├── lstm_summary.pkl              ← LSTM results
│   └── prophet_summary.pkl           ← Prophet results
│
└── plots/
    ├── model_comparison_bars.png     ← RMSE/MAPE comparison
    ├── holiday_impact.png            ← Holiday feature effect
    ├── correlation_matrix.png        ← Market correlations
    └── price_distribution.png        ← Price histograms
```

---

## 💡 Pro Tips

1. **Run notebooks in VS Code** - Better debugging and visualization
2. **Check outputs carefully** - Verify RMSE values before proceeding
3. **Save trained models** - No need to retrain for comparison
4. **Use log transformation** - Essential for ARIMA with price data
5. **Read the README** - Full documentation in `notebooks/README.md`

---

## ✅ Success Checklist

- [ ] Notebook 01: Data cleaned, holiday features added
- [ ] Notebook 02: ARIMA RMSE is 10-15k (with log transform)
- [ ] Notebook 03: LSTM created 2 model files
- [ ] Notebook 04: Prophet trained successfully
- [ ] Notebook 05: Comparison charts generated
- [ ] Results saved to `results/metrics/` and `results/plots/`

---

**Ready to start?** Open `notebooks/01_data_cleaning_and_eda.ipynb` and run all cells! 🚀
