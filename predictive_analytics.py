"""
=============================================================
  Task 3: Predictive Analytics Using Historical Data
  Thiranex Internship | Data Analytics Domain
=============================================================
  Key Features:
    1. Use regression or time-series models for prediction
    2. Clean and preprocess historical datasets
    3. Evaluate model accuracy and visualize predictions

  Expected Outcome:
    - Predictive modeling
    - Trend analysis
    - Data-driven forecasting
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, mean_absolute_percentage_error)
from sklearn.model_selection import train_test_split, cross_val_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# ══════════════════════════════════════════════════════════════
# SECTION 1 — GENERATE REALISTIC HISTORICAL DATASET
# ══════════════════════════════════════════════════════════════
np.random.seed(42)
dates = pd.date_range(start="2021-01-01", end="2024-06-01", freq="MS")
n = len(dates)

trend      = np.linspace(10000, 28000, n)
seasonal   = 3500 * np.sin(2 * np.pi * np.arange(n) / 12)
noise      = np.random.normal(0, 700, n)
sales_raw  = trend + seasonal + noise

# --- inject missing values, duplicates, inconsistencies ---
sales_raw[5]  = np.nan          # missing
sales_raw[18] = np.nan          # missing
sales_raw[30] = sales_raw[30] * 4.2   # extreme outlier
sales_raw[11] = -500            # negative (inconsistent)

df = pd.DataFrame({"Date": dates, "Sales": sales_raw})
df.set_index("Date", inplace=True)

print("=" * 65)
print("  TASK 3: PREDICTIVE ANALYTICS USING HISTORICAL DATA")
print("  Thiranex Internship Portal")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
# SECTION 2 — DATA CLEANING & PREPROCESSING
# ══════════════════════════════════════════════════════════════
print("\n──────────────────────────────────────────")
print("  [STEP 1] DATA CLEANING & PREPROCESSING")
print("──────────────────────────────────────────")

raw_shape      = df.shape
missing_before = df["Sales"].isna().sum()
negatives      = (df["Sales"] < 0).sum()

print(f"  Dataset shape        : {raw_shape[0]} rows × {raw_shape[1]} col")
print(f"  Missing values       : {missing_before}")
print(f"  Negative values      : {negatives}  (inconsistent data)")
print(f"  Sales range (raw)    : ₹{df['Sales'].min():,.0f} – ₹{df['Sales'].max():,.0f}")

# Fix negatives → NaN
df.loc[df["Sales"] < 0, "Sales"] = np.nan

# Outlier detection via IQR
Q1  = df["Sales"].quantile(0.25)
Q3  = df["Sales"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers    = df[(df["Sales"] > upper_bound) | (df["Sales"] < lower_bound)]
print(f"  Outliers (IQR)       : {len(outliers)} row(s)")

# Cap outliers
df["Sales"] = df["Sales"].clip(lower_bound, upper_bound)

# Fill missing with linear interpolation
df["Sales"] = df["Sales"].interpolate(method="linear").bfill().ffill()

missing_after = df["Sales"].isna().sum()
print(f"  Missing after clean  : {missing_after}")
print(f"  Sales range (clean)  : ₹{df['Sales'].min():,.0f} – ₹{df['Sales'].max():,.0f}")
print(f"  ✅ Data cleaned successfully")

# ── Feature Engineering ──
df["Month"]      = df.index.month
df["Quarter"]    = df.index.quarter
df["Year"]       = df.index.year
df["TimeIndex"]  = np.arange(len(df))
df["Sin12"]      = np.sin(2 * np.pi * df["Month"] / 12)
df["Cos12"]      = np.cos(2 * np.pi * df["Month"] / 12)
df["Lag1"]       = df["Sales"].shift(1).bfill()
df["Lag3"]       = df["Sales"].shift(3).bfill()
df["RollingMean3"] = df["Sales"].rolling(3, min_periods=1).mean()

print(f"\n  Features engineered  : Month, Quarter, Year, TimeIndex,")
print(f"                         Sin/Cos seasonality, Lag1, Lag3, RollingMean3")

# ══════════════════════════════════════════════════════════════
# SECTION 3 — TREND ANALYSIS
# ══════════════════════════════════════════════════════════════
print("\n──────────────────────────────────────────")
print("  [STEP 2] TREND ANALYSIS")
print("──────────────────────────────────────────")

df["MA3"]  = df["Sales"].rolling(window=3,  min_periods=1).mean()
df["MA6"]  = df["Sales"].rolling(window=6,  min_periods=1).mean()
df["MA12"] = df["Sales"].rolling(window=12, min_periods=1).mean()

# Growth rate
df["MoM_Growth"] = df["Sales"].pct_change() * 100
df["YoY_Growth"] = df["Sales"].pct_change(12) * 100

avg_mom = df["MoM_Growth"].mean()
avg_yoy = df["YoY_Growth"].dropna().mean()

print(f"  Avg Month-on-Month Growth : {avg_mom:.2f}%")
print(f"  Avg Year-on-Year Growth   : {avg_yoy:.2f}%")
print(f"  Peak Sales Month          : {df['Sales'].idxmax().strftime('%b %Y')}  (₹{df['Sales'].max():,.0f})")
print(f"  Lowest Sales Month        : {df['Sales'].idxmin().strftime('%b %Y')}  (₹{df['Sales'].min():,.0f})")

# Seasonality by month
monthly_avg = df.groupby("Month")["Sales"].mean()
best_month  = monthly_avg.idxmax()
worst_month = monthly_avg.idxmin()
month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
print(f"  Best seasonal month       : {month_names[best_month]}  (avg ₹{monthly_avg[best_month]:,.0f})")
print(f"  Worst seasonal month      : {month_names[worst_month]}  (avg ₹{monthly_avg[worst_month]:,.0f})")

# Stationarity Test (ADF)
adf_result = adfuller(df["Sales"])
print(f"\n  Stationarity (ADF Test)")
print(f"    ADF Statistic : {adf_result[0]:.4f}")
print(f"    p-value       : {adf_result[1]:.4f}")
print(f"    Series is     : {'Stationary ✅' if adf_result[1] < 0.05 else 'Non-Stationary (trend present)'}")

# Seasonal Decomposition
decomp = seasonal_decompose(df["Sales"], model="additive", period=12, extrapolate_trend="freq")

# ══════════════════════════════════════════════════════════════
# SECTION 4 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════
print("\n──────────────────────────────────────────")
print("  [STEP 3] MODEL TRAINING & EVALUATION")
print("──────────────────────────────────────────")

features = ["TimeIndex","Sin12","Cos12","Month","Quarter","Year","Lag1","Lag3","RollingMean3"]
X = df[features]
y = df["Sales"]

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

results = {}

def evaluate(name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape,
                     "y_pred": y_pred, "y_true": y_true}
    print(f"\n  ── {name}")
    print(f"     R² Score  : {r2:.4f}")
    print(f"     MAE       : ₹{mae:,.0f}")
    print(f"     RMSE      : ₹{rmse:,.0f}")
    print(f"     MAPE      : {mape:.2f}%")
    return r2, mae, rmse, mape

# ── Model 1: Linear Regression ──
lr = LinearRegression()
lr.fit(X_train, y_train)
evaluate("Linear Regression", y_test, lr.predict(X_test))

# ── Model 2: Ridge Regression ──
ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10))])
ridge.fit(X_train, y_train)
evaluate("Ridge Regression", y_test, ridge.predict(X_test))

# ── Model 3: Polynomial Regression (degree=2) ──
poly_pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])
X_time_train = X_train[["TimeIndex","Sin12","Cos12"]]
X_time_test  = X_test[["TimeIndex","Sin12","Cos12"]]
poly_pipe.fit(X_time_train, y_train)
evaluate("Polynomial Regression (deg 2)", y_test, poly_pipe.predict(X_time_test))

# ── Model 4: Holt-Winters (Time Series) ──
train_ts = df["Sales"].iloc[:split]
test_ts  = df["Sales"].iloc[split:]
hw = ExponentialSmoothing(train_ts, trend="add", seasonal="add", seasonal_periods=12).fit()
y_pred_hw = hw.forecast(len(test_ts))
y_pred_hw.index = test_ts.index
evaluate("Holt-Winters Exp. Smoothing", test_ts, y_pred_hw.values)

# Cross-validation for Linear Regression
cv_scores = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")
print(f"\n  Cross-Validation (Linear Regression, 5-fold)")
print(f"     CV R² scores : {[round(s,4) for s in cv_scores]}")
print(f"     Mean R²      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ══════════════════════════════════════════════════════════════
# SECTION 5 — FUTURE FORECAST (12 months)
# ══════════════════════════════════════════════════════════════
print("\n──────────────────────────────────────────")
print("  [STEP 4] DATA-DRIVEN FORECASTING (12M)")
print("──────────────────────────────────────────")

# Retrain Holt-Winters on full data for forecasting
hw_full = ExponentialSmoothing(df["Sales"], trend="add", seasonal="add", seasonal_periods=12).fit()
forecast_vals  = hw_full.forecast(12)
forecast_dates = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=12, freq="MS")
forecast_series = pd.Series(forecast_vals.values, index=forecast_dates)

# Confidence interval (±7% band)
ci_upper = forecast_series * 1.07
ci_lower = forecast_series * 0.93

print(f"\n  {'Month':<12} {'Forecast':>12} {'Lower (−7%)':>14} {'Upper (+7%)':>14}")
print(f"  {'-'*54}")
for d, v, lo, hi in zip(forecast_dates, forecast_series, ci_lower, ci_upper):
    print(f"  {d.strftime('%b %Y'):<12} ₹{v:>10,.0f} ₹{lo:>12,.0f} ₹{hi:>12,.0f}")

total_forecast = forecast_series.sum()
print(f"\n  Projected Annual Revenue (next 12M) : ₹{total_forecast:,.0f}")

# Best model
best_name = max(results, key=lambda k: results[k]["R2"])
print(f"\n  ✅ Best performing model : {best_name} (R² = {results[best_name]['R2']:.4f})")

# ══════════════════════════════════════════════════════════════
# SECTION 6 — VISUALIZATION DASHBOARD
# ══════════════════════════════════════════════════════════════
BG      = "#0f1117"
PANEL   = "#1a1d27"
GRID_C  = "#2a2d3e"
TEXT    = "#e2e8f0"
BLUE    = "#4f9cf9"
GREEN   = "#22c55e"
ORANGE  = "#f97316"
PURPLE  = "#a855f7"
YELLOW  = "#eab308"
RED     = "#ef4444"
CYAN    = "#06b6d4"
PINK    = "#ec4899"

def style_ax(ax, title, ylabel=""):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors=TEXT, labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color(GRID_C)
    ax.grid(True, color=GRID_C, linewidth=0.5, linestyle="--", alpha=0.6)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=8)
    ax.xaxis.label.set_color(TEXT)

fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.32)

# ── Plot 1: Historical Sales + Moving Averages (Trend Analysis) ──
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df.index, df["Sales"], color=BLUE, linewidth=1.6, label="Actual Sales", alpha=0.9)
ax1.plot(df.index, df["MA3"],  color=ORANGE, linewidth=1.3, linestyle="--", label="3M MA")
ax1.plot(df.index, df["MA6"],  color=GREEN,  linewidth=1.3, linestyle="--", label="6M MA")
ax1.plot(df.index, df["MA12"], color=PURPLE, linewidth=1.5, linestyle="-",  label="12M MA (Trend)")
ax1.fill_between(df.index, df["Sales"], alpha=0.07, color=BLUE)
style_ax(ax1, "📈 Historical Sales with Moving Averages — Trend Analysis", "Sales (₹)")
ax1.legend(fontsize=8.5, facecolor=PANEL, labelcolor=TEXT, framealpha=0.8)

# ── Plot 2: Seasonal Decomposition — Trend ──
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(decomp.trend.index, decomp.trend, color=CYAN, linewidth=1.5)
ax2.fill_between(decomp.trend.index, decomp.trend, alpha=0.1, color=CYAN)
style_ax(ax2, "📉 Decomposition: Trend Component", "Value (₹)")

# ── Plot 3: Seasonal Component ──
ax3 = fig.add_subplot(gs[1, 1])
ax3.bar(decomp.seasonal.index, decomp.seasonal, color=PURPLE, alpha=0.7, width=20)
style_ax(ax3, "🌀 Decomposition: Seasonal Component", "Value (₹)")

# ── Plot 4: Monthly Average (Seasonality Pattern) ──
ax4 = fig.add_subplot(gs[1, 2])
colors_bar = [GREEN if v == monthly_avg.max() else (RED if v == monthly_avg.min() else BLUE)
              for v in monthly_avg]
ax4.bar([month_names[m] for m in monthly_avg.index], monthly_avg.values,
        color=colors_bar, alpha=0.85, edgecolor=BG, linewidth=0.5)
style_ax(ax4, "📅 Average Sales by Month (Seasonality)", "Avg Sales (₹)")
ax4.tick_params(axis="x", labelsize=7)

# ── Plot 5: Linear Regression — Actual vs Predicted ──
ax5 = fig.add_subplot(gs[2, 0])
lr_res = results["Linear Regression"]
ax5.plot(y_test.index, lr_res["y_true"], color=BLUE, linewidth=1.5, label="Actual")
ax5.plot(y_test.index, lr_res["y_pred"], color=ORANGE, linewidth=1.5,
         linestyle="--", label=f"Predicted\nR²={lr_res['R2']:.3f}")
style_ax(ax5, "🔵 Linear Regression — Actual vs Predicted", "Sales (₹)")
ax5.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TEXT)

# ── Plot 6: Ridge Regression — Actual vs Predicted ──
ax6 = fig.add_subplot(gs[2, 1])
rr = results["Ridge Regression"]
ax6.plot(y_test.index, rr["y_true"], color=BLUE, linewidth=1.5, label="Actual")
ax6.plot(y_test.index, rr["y_pred"], color=CYAN, linewidth=1.5,
         linestyle="--", label=f"Predicted\nR²={rr['R2']:.3f}")
style_ax(ax6, "🔷 Ridge Regression — Actual vs Predicted", "Sales (₹)")
ax6.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TEXT)

# ── Plot 7: Holt-Winters — Actual vs Predicted ──
ax7 = fig.add_subplot(gs[2, 2])
hw_res = results["Holt-Winters Exp. Smoothing"]
ax7.plot(test_ts.index, hw_res["y_true"], color=BLUE, linewidth=1.5, label="Actual")
ax7.plot(test_ts.index, hw_res["y_pred"], color=GREEN, linewidth=1.5,
         linestyle="--", label=f"Predicted\nR²={hw_res['R2']:.3f}")
style_ax(ax7, "🟢 Holt-Winters — Actual vs Predicted", "Sales (₹)")
ax7.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TEXT)

# ── Plot 8: 12-Month Future Forecast ──
ax8 = fig.add_subplot(gs[3, :2])
ax8.plot(df.index, df["Sales"], color=BLUE, linewidth=1.5, label="Historical Sales")
ax8.plot(forecast_dates, forecast_series, color=YELLOW, linewidth=2.2,
         linestyle="-.", marker="o", markersize=4, label="Forecast (12M)")
ax8.fill_between(forecast_dates, ci_lower, ci_upper, alpha=0.18, color=YELLOW, label="±7% Confidence Band")
ax8.axvline(x=df.index[-1], color=RED, linewidth=1.2, linestyle=":", alpha=0.8, label="Forecast Start")
style_ax(ax8, "🔮 Data-Driven Forecasting — Next 12 Months (Jul 2024 – Jun 2025)", "Sales (₹)")
ax8.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT)

# ── Plot 9: Model Comparison (R², MAPE) ──
ax9 = fig.add_subplot(gs[3, 2])
model_names_short = ["Linear\nRegr.", "Ridge\nRegr.", "Poly\nRegr.", "Holt-\nWinters"]
r2_vals   = [results[k]["R2"]   for k in results]
mape_vals = [results[k]["MAPE"] for k in results]
bar_colors = [ORANGE, CYAN, PURPLE, GREEN]

x = np.arange(len(model_names_short))
bars = ax9.bar(x, r2_vals, color=bar_colors, edgecolor=BG, width=0.55, alpha=0.88)
for bar, val in zip(bars, r2_vals):
    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", color=TEXT, fontsize=7.5, fontweight="bold")
ax9.set_xticks(x)
ax9.set_xticklabels(model_names_short)
ax9.set_ylim(0, 1.15)
style_ax(ax9, "📊 Model Accuracy Comparison (R² Score)", "R² Score")

# Highlight best model
best_idx = list(results.keys()).index(best_name)
bars[best_idx].set_edgecolor(YELLOW)
bars[best_idx].set_linewidth(2.5)

# Main title
fig.suptitle(
    "TASK 3 — PREDICTIVE ANALYTICS USING HISTORICAL DATA\nThiranex Internship | Data Analytics Domain",
    color=TEXT, fontsize=14, fontweight="bold", y=0.995
)

plt.savefig("/mnt/user-data/outputs/predictive_analytics_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\n✅ Dashboard saved: predictive_analytics_dashboard.png")
print("=" * 65)
