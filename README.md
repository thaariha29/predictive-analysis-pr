# 📊 Task 3 — Predictive Analytics Using Historical Data

> **Thiranex Internship Portal | Data Analytics Domain**  
> Build a predictive model to forecast future trends using regression and time-series techniques.

---

## 📌 Overview

This project implements end-to-end **Predictive Analytics** on historical sales data. It covers the full data science workflow — from raw data cleaning to model evaluation and future forecasting — fulfilling all requirements of Task 3 in the Thiranex Internship Program.

---

## ✅ Key Features Implemented

| # | Requirement | Status |
|---|---|---|
| 1 | Use regression or time-series models for prediction | ✅ Done |
| 2 | Clean and preprocess historical datasets | ✅ Done |
| 3 | Evaluate model accuracy and visualize predictions | ✅ Done |

### Expected Outcomes Covered
- ✅ **Predictive modeling** — 4 models trained and compared
- ✅ **Trend analysis** — Moving averages, MoM/YoY growth, stationarity test, seasonal decomposition
- ✅ **Data-driven forecasting** — 12-month future forecast with confidence bands

---

## 🗂️ Project Structure

```
predictive-analytics-task3/
│
├── predictive_analytics.py          # Main Python script
├── predictive_analytics_dashboard.png  # Output visualization
├── README.md                        # Project documentation
└── LICENSE                          # MIT License
```

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core programming language |
| Pandas | Data manipulation & feature engineering |
| NumPy | Numerical computations |
| Scikit-learn | Regression models, metrics, cross-validation |
| Statsmodels | Holt-Winters, ADF test, seasonal decomposition |
| Matplotlib | Visualization dashboard |

---

## 📋 Workflow

### Step 1 — Data Cleaning & Preprocessing
- Detected and fixed **missing values** via linear interpolation
- Identified and removed **negative/inconsistent values**
- Capped **outliers** using the IQR method
- Engineered features: Month, Quarter, Year, TimeIndex, Sine/Cosine seasonality, Lag features, Rolling mean

### Step 2 — Trend Analysis
- 3-month, 6-month, and 12-month **Moving Averages**
- Month-on-Month and Year-on-Year **Growth Rate** analysis
- **Seasonal Decomposition** (Trend + Seasonality + Residual)
- **ADF Stationarity Test** on the time series
- Best and worst performing months identified

### Step 3 — Model Training & Evaluation
Four models trained and evaluated on the test set:

| Model | R² Score | MAE | RMSE | MAPE |
|---|---|---|---|---|
| **Linear Regression** ⭐ | **0.9598** | **₹499** | **₹749** | **2.04%** |
| Ridge Regression | 0.8563 | ₹883 | ₹1,416 | 3.81% |
| Polynomial Regression (deg 2) | 0.2418 | ₹2,714 | ₹3,253 | 9.84% |
| Holt-Winters Exp. Smoothing | 0.1048 | ₹3,473 | ₹3,534 | 13.26% |

- 5-fold **cross-validation** performed on Linear Regression (Mean R² = 0.80)
- Metrics used: **R², MAE, RMSE, MAPE**

### Step 4 — Data-Driven Forecasting
- Holt-Winters model retrained on full dataset
- **12-month future forecast** generated (Jul 2024 – Jun 2025)
- **Confidence intervals** (±7%) included for each forecast point
- **Projected Annual Revenue**: ₹3,73,832

---

## 📊 Visualization Dashboard

The output dashboard (`predictive_analytics_dashboard.png`) contains **9 panels**:

1. Historical Sales with 3M / 6M / 12M Moving Averages
2. Seasonal Decomposition — Trend Component
3. Seasonal Decomposition — Seasonal Component
4. Average Sales by Month (Seasonality Pattern)
5. Linear Regression — Actual vs Predicted
6. Ridge Regression — Actual vs Predicted
7. Holt-Winters — Actual vs Predicted
8. 12-Month Future Forecast with Confidence Band
9. Model Accuracy Comparison (R² Bar Chart)

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/predictive-analytics-task3.git
cd predictive-analytics-task3
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib statsmodels
```

### 3. Run the script
```bash
python predictive_analytics.py
```

The dashboard image will be saved in the same directory as `predictive_analytics_dashboard.png`.

---

## 📈 Sample Output

```
[STEP 1] DATA CLEANING & PREPROCESSING
  Missing values       : 2
  Negative values      : 1  (inconsistent data)
  Outliers (IQR)       : 1 row(s)
  ✅ Data cleaned successfully

[STEP 2] TREND ANALYSIS
  Avg Month-on-Month Growth : 3.47%
  Avg Year-on-Year Growth   : 35.09%
  Best seasonal month       : Apr  (avg ₹22,988)

[STEP 3] MODEL TRAINING & EVALUATION
  Linear Regression  →  R²: 0.9598 | MAE: ₹499 | MAPE: 2.04%

[STEP 4] FORECASTING (12M)
  Jul 2024: ₹34,125  |  Jun 2025: ₹34,866
  Projected Annual Revenue: ₹3,73,832
```

---

## 🏷️ Tags

`python` `machine-learning` `predictive-analytics` `time-series` `regression` `data-science` `forecasting` `thiranex` `internship`

---

## 👩‍💻 Author

**Thaariha**  
Data Analytics Intern — Thiranex  

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
