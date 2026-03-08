
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
import os

data_folder = 'data'


seasonal_lags = {
    'ETTh1.csv': 24,    # hourly data, daily seasonality
    'ETTh2.csv': 24,
    'ETTm1.csv': 96,    # 15-min data, daily seasonality
    'ETTm2.csv': 96,
    'electricity.csv': 24,
    'traffic.csv': 24,
    'weather.csv': 144,  # 10-min data
    'exchange_rate.csv': 7,  # daily data, weekly seasonality
    'national_illness.csv': 52  # weekly data, yearly seasonality
}

def check_seasonality(series, lag):
    """Check ACF at a specific lag"""
    
    acf_values = acf(series, nlags=min(lag + 5, len(series)//2), fft=True)
    
    if lag < len(acf_values):
        return abs(acf_values[lag])
    else:
        return 0.0

# Process each dataset
for file in seasonal_lags.keys():
    filepath = f'{data_folder}/{file}'
    
    if not os.path.exists(filepath):
        print(f"Skipping {file} - not found")
        continue
    
    print(f"\n=== {file} ===")
    df = pd.read_csv(filepath)
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    lag = seasonal_lags[file]
    print(f"Seasonal lag: {lag}")
    
    acf_scores = []
    
    for col in numeric_cols:
        series = df[col].dropna()
        
        if len(series) < lag + 10:
            continue
        
        acf_value = check_seasonality(series, lag)
        acf_scores.append(acf_value)
        
        
        if acf_value >= 0.7:
            strength = "Strong"
        elif acf_value >= 0.3:
            strength = "Moderate"  
        else:
            strength = "Weak"
        
        print(f"  {col}: ACF={acf_value:.3f} ({strength})")
    
    if acf_scores:
        mean_acf = np.mean(acf_scores)
        print(f"Mean ACF: {mean_acf:.3f}")
