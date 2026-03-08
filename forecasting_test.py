"""
testing on one dataset
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load ETTh1
df = pd.read_csv('data/ETTh1.csv')
series = df['OT'].values

# Simple train
train_size = int(len(series) * 0.8)
train = series[:train_size]
test = series[train_size:]

print(f"Training on {len(train)} points")
print(f"Testing on {len(test)} points")

# Fit ARIMA
print("\nFitting ARIMA(5,1,0)...")
model = ARIMA(train, order=(5, 1, 0))
fitted = model.fit()

# Forecast
forecast = fitted.forecast(steps=len(test))

# Simple MAE
mae = np.mean(np.abs(test - forecast))
print(f"MAE: {mae:.4f}")


