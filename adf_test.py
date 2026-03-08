

import pandas as pd
from statsmodels.tsa.stattools import adfuller
import os

data_folder = 'data'
files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

print("Running ADF tests...\n")

for file in files:
    print(f"=== {file} ===")
    df = pd.read_csv(f'{data_folder}/{file}')
    
    # numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    stationary_count = 0
    
    for col in numeric_cols:
        series = df[col].dropna()
        
        # Skip if too short
        if len(series) < 10:
            print(f"  {col}: Skipped (insufficient data)")
            continue
        
        try:
            #ADF test
            result = adfuller(series)
            p_value = result[1]
            
            if p_value < 0.05:
                verdict = "Stationary"
                stationary_count += 1
            else:
                verdict = "Non-stationary"
            
            print(f"  {col}: p={p_value:.4f} - {verdict}")
        except Exception as e:
            print(f"  {col}: Error - {str(e)}")

    
    
    pct = (stationary_count / len(numeric_cols)) * 100 if len(numeric_cols) > 0 else 0
    print(f"  Stationary: {stationary_count}/{len(numeric_cols)} ({pct:.1f}%)\n")
