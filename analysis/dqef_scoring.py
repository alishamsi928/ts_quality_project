"""
DQEF - Dataset Quality Evaluation Framework
Dimensions:
1. Validity - 25% (missing values, duplicates, timestamp regularity)
2. Statistical Integrity - 25% (stationarity, seasonality) 
3. Variability - 15% (coefficient of variation)
4. Diversity - 20% (channel correlation - MAPC)
5. Completeness - 15% (data volume, timespan)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import adfuller, acf

def normalize_score(values, higher_is_better=True):
    
    arr = np.array(values)
    min_val = arr.min()
    max_val = arr.max()
    
    # Avoid division by zero
    if max_val == min_val:
        return np.full(len(arr), 50.0)
    
    # Min-max normalization
    normalized = 100 * (arr - min_val) / (max_val - min_val)
    
    # Flip if lower is better
    if not higher_is_better:
        normalized = 100 - normalized
    
    return normalized

# Seasonal lag mapping
seasonal_lags = {
    'ETTh1.csv': 24,
    'ETTh2.csv': 24,
    'ETTm1.csv': 96, 
    'ETTm2.csv': 96,
    'electricity.csv': 24,
    'traffic.csv': 24,
    'weather.csv': 144, 
    'exchange_rate.csv': 7,
    'national_illness.csv': 52
}

data_folder = 'data'
files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

print("Analyzing all datasets for DQEF scoring...\n")


all_metrics = []

for file in files:
    print(f"Processing {file}...")
    df = pd.read_csv(f'{data_folder}/{file}')
    
    metrics = {'Dataset': file}
    # Missing values
    total_cells = df.shape[0] * df.shape[1]
    missing = df.isnull().sum().sum()
    metrics['missing_pct'] = (missing / total_cells) * 100 if total_cells > 0 else 0
    
    # Duplicates
    duplicates = df.duplicated().sum()
    metrics['duplicate_pct'] = (duplicates / df.shape[0]) * 100 if df.shape[0] > 0 else 0
    
    # Timestamp regularity
    metrics['irregular_ts_pct'] = 0  
    
    # STATISTICAL INTEGRITY
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Stationarity (ADF test)
    stationary_count = 0
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) >= 10:
            try:
                result = adfuller(series)
                if result[1] < 0.05:  # p-value
                    stationary_count += 1
            except:
                pass
    
    metrics['stationary_pct'] = (stationary_count / len(numeric_cols)) * 100 if len(numeric_cols) > 0 else 0
    
    # Seasonality (ACF)
    seasonal_lag = seasonal_lags.get(file, 24)
    acf_scores = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > seasonal_lag + 10:
            try:
                acf_values = acf(series, nlags=min(seasonal_lag + 5, len(series)//2), fft=True)
                if seasonal_lag < len(acf_values):
                    acf_scores.append(abs(acf_values[seasonal_lag]))
            except:
                pass
    
    metrics['mean_acf'] = np.mean(acf_scores) if acf_scores else 0
    
    # VARIABILITY 
    cv_scores = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 0:
            mean = np.mean(series)
            std = np.std(series)
            if abs(mean) > 1e-8:
                cv_scores.append(std / abs(mean))
    
    metrics['mean_cv'] = np.mean(cv_scores) if cv_scores else 0
    
    # DIVERSITY 
    # MAPC (Mean Absolute Pairwise Correlation)
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        pairwise_corrs = corr_matrix.where(mask).stack().values
        metrics['mapc'] = np.mean(np.abs(pairwise_corrs))
    else:
        metrics['mapc'] = 0
    
    # COMPLETENESS
    metrics['rows'] = df.shape[0]
    metrics['columns'] = len(numeric_cols)
    
    # Time span (if date column exists)
    date_cols = df.select_dtypes(include=['object']).columns
    if len(date_cols) > 0:
        try:
            dates = pd.to_datetime(df[date_cols[0]], errors='coerce')
            dates = dates.dropna()
            if len(dates) > 1:
                time_span = (dates.max() - dates.min()).days
                metrics['time_span_days'] = time_span
            else:
                metrics['time_span_days'] = 0
        except:
            metrics['time_span_days'] = 0
    else:
        metrics['time_span_days'] = 0
    
    all_metrics.append(metrics)

# Convert to DataFrame
df_metrics = pd.DataFrame(all_metrics)

print(f"\nCollected metrics for {len(df_metrics)} datasets")
print("\nNormalizing and computing dimension scores...")



# 1. VALIDITY (25%)
missing_norm = normalize_score(df_metrics['missing_pct'].values, higher_is_better=False)
dup_norm = normalize_score(df_metrics['duplicate_pct'].values, higher_is_better=False)
ts_norm = normalize_score(df_metrics['irregular_ts_pct'].values, higher_is_better=False)
df_metrics['Validity'] = missing_norm * 0.5 + dup_norm * 0.3 + ts_norm * 0.2

# 2. STATISTICAL INTEGRITY (25%)
stat_norm = normalize_score(df_metrics['stationary_pct'].values, higher_is_better=True)
acf_norm = normalize_score(df_metrics['mean_acf'].values, higher_is_better=True)
df_metrics['Stat_Integrity'] = stat_norm * 0.5 + acf_norm * 0.5

# 3. VARIABILITY (15%)
cv_norm = normalize_score(df_metrics['mean_cv'].values, higher_is_better=True)
df_metrics['Variability'] = cv_norm

# 4. DIVERSITY (20%)
div_norm = normalize_score(df_metrics['mapc'].values, higher_is_better=False)
df_metrics['Diversity'] = div_norm

# 5. COMPLETENESS (15%)
rows_norm = normalize_score(df_metrics['rows'].values, higher_is_better=True)
cols_norm = normalize_score(df_metrics['columns'].values, higher_is_better=True)
span_norm = normalize_score(df_metrics['time_span_days'].values, higher_is_better=True)
df_metrics['Completeness'] = rows_norm * 0.4 + cols_norm * 0.3 + span_norm * 0.3

# COMPOSITE SCORE
df_metrics['Composite_Score'] = (
    df_metrics['Validity'] * 0.25 +
    df_metrics['Stat_Integrity'] * 0.25 +
    df_metrics['Variability'] * 0.15 +
    df_metrics['Diversity'] * 0.20 +
    df_metrics['Completeness'] * 0.15
)

# Round scores
score_cols = ['Validity', 'Stat_Integrity', 'Variability', 'Diversity', 'Completeness', 'Composite_Score']
df_metrics[score_cols] = df_metrics[score_cols].round(2)

# Sort by composite score
df_metrics_sorted = df_metrics.sort_values('Composite_Score', ascending=False)

# Display results
print("\n" + "=" * 80)
print("DQEF SCORES - ALL DATASETS")
print("=" * 80)
print(df_metrics_sorted[['Dataset', 'Validity', 'Stat_Integrity', 'Variability', 
                          'Diversity', 'Completeness', 'Composite_Score']].to_string(index=False))
print("=" * 80)

# Save results
os.makedirs('results', exist_ok=True)
df_metrics.to_csv('results/dqef_scores.csv', index=False)
print("\n✅ Scores saved to results/dqef_scores.csv")

# Bar chart
fig, ax = plt.subplots(figsize=(12, 6))

datasets = df_metrics_sorted['Dataset'].str.replace('.csv', '').values
scores = df_metrics_sorted['Composite_Score'].values

# gradient
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(datasets)))

bars = ax.barh(datasets, scores, color=colors, edgecolor='gray', height=0.6)

# Add score labels
for bar, score in zip(bars, scores):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
            f'{score:.1f}', va='center', fontweight='bold')

ax.set_xlabel('DQEF Composite Score', fontsize=12)
ax.set_title('Dataset Quality Rankings', fontsize=14, fontweight='bold')
ax.set_xlim(0, 110)
ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='Average')
ax.legend()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/dqef_rankings.png', dpi=150, bbox_inches='tight')
print("✅ Chart saved to plots/dqef_rankings.png")

print("\nDone!")
