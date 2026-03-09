
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr

try:
    dqef = pd.read_csv('results/dqef_scores.csv')
    forecast = pd.read_csv('results/forecasting_results.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Merge on dataset name
dqef['Dataset'] = dqef['Dataset'].str.replace('.csv', '')
merged = pd.merge(dqef, forecast, on='Dataset', how='inner')

if len(merged) == 0:
    print("No matching datasets found between DQEF and forecasting results")
    exit(1)

print("=" * 70)
print("QUALITY vs PERFORMANCE ANALYSIS")
print("=" * 70)
print(f"\nAnalyzing {len(merged)} datasets\n")

# DQEF vs MAE
print("1. DQEF Composite Score vs Model Performance (MAE)")
print("-" * 70)

for model in ['ARIMA', 'DLinear', 'LSTM']:
    mae_col = f'{model}_MAE'
    if mae_col in merged.columns:
        valid = merged[['Composite_Score', mae_col]].dropna()
        if len(valid) >= 3:
            r, p = pearsonr(valid['Composite_Score'], valid[mae_col])
            print(f"  {model:10s}: r = {r:7.4f}, p = {p:.4f}", end="")
            if p < 0.05:
                if r < -0.5:
                    print("  → Strong negative ")
                elif r > 0.5:
                    print("  → Strong positive ")
                else:
                    print("  → Weak correlation ")
            else:
                print("  → Not significant")

# Diversity vs Performance
print("\n2. Channel Diversity (lower MAPC = more diverse) vs MAE")
print("-" * 70)

if 'Diversity' in merged.columns:
    for model in ['ARIMA', 'DLinear', 'LSTM']:
        mae_col = f'{model}_MAE'
        if mae_col in merged.columns:
            valid = merged[['Diversity', mae_col]].dropna()
            if len(valid) >= 3:
                r, p = pearsonr(valid['Diversity'], valid[mae_col])
                print(f"  {model:10s}: r = {r:7.4f}, p = {p:.4f}")

# Statistical Integrity vs R²
print("\n3. Statistical Integrity vs R² (higher = better)")
print("-" * 70)

if 'Stat_Integrity' in merged.columns:
    for model in ['ARIMA', 'DLinear', 'LSTM']:
        r2_col = f'{model}_R2'
        if r2_col in merged.columns:
            valid = merged[['Stat_Integrity', r2_col]].dropna()
            if len(valid) >= 3:
                r, p = pearsonr(valid['Stat_Integrity'], valid[r2_col])
                print(f"  {model:10s}: r = {r:7.4f}, p = {p:.4f}")

# Create visualizations
os.makedirs('plots', exist_ok=True)

# DQEF vs DLinear MAE
if 'DLinear_MAE' in merged.columns:
    valid = merged[['Dataset', 'Composite_Score', 'DLinear_MAE']].dropna()
    
    if len(valid) >= 3:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = valid['Composite_Score'].values
        y = valid['DLinear_MAE'].values
        
        ax.scatter(x, y, s=100, alpha=0.6, color='steelblue')
        
        for i, name in enumerate(valid['Dataset']):
            ax.annotate(name, (x[i], y[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.7, label="Trend line")
        
        r, p_val = pearsonr(x, y)
        
        ax.set_xlabel('DQEF Composite Score (higher = better quality)', fontsize=11)
        ax.set_ylabel('DLinear MAE (lower = better performance)', fontsize=11)
        ax.set_title(f'Dataset Quality vs Forecasting Performance\n(r = {r:.3f}, p = {p_val:.3f})', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('plots/quality_vs_performance.png', dpi=150, bbox_inches='tight')
        print("\n Plot saved: plots/quality_vs_performance.png")

# Save merged results
merged.to_csv('results/final_analysis.csv', index=False)
print(" Full results saved: results/final_analysis.csv")


