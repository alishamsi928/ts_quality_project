

import pandas as pd
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    PYTORCH_AVAILABLE = True
    
    print("\n" + "="*60)
    print("PyTorch GPU Setup")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*60 + "\n")
    
except ImportError:
    PYTORCH_AVAILABLE = False
    print("\n  PyTorch not available - using NumPy fallbacks\n")


if PYTORCH_AVAILABLE:
    
    class TimeSeriesLSTM(nn.Module):
        
        def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.2, horizon=96):
            super(TimeSeriesLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.horizon = horizon
            
            # Smaller network to fit in memory
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, horizon)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            last_output = out[:, -1, :]
            last_output = self.dropout(last_output)
            predictions = self.fc(last_output)
            return predictions
    
    
    class DLinearPyTorch(nn.Module):
        
        def __init__(self, lookback=96, horizon=96, kernel_size=25):
            super(DLinearPyTorch, self).__init__()
            self.lookback = lookback
            self.horizon = horizon
            self.kernel_size = kernel_size
            self.trend_layer = nn.Linear(lookback, horizon)
            self.residual_layer = nn.Linear(lookback, horizon)
        
        def moving_average(self, x):
            batch_size = x.size(0)
            pad = self.kernel_size // 2
            x_padded = torch.nn.functional.pad(x, (pad, pad), mode='replicate')
            x_padded = x_padded.unfold(dimension=1, size=self.kernel_size, step=1)
            trend = x_padded.mean(dim=2)
            return trend[:, :self.lookback]
        
        def forward(self, x):
            trend = self.moving_average(x)
            residual = x - trend
            trend_output = self.trend_layer(trend)
            residual_output = self.residual_layer(residual)
            return trend_output + residual_output
    
    
    class LSTMForecaster:
        
        def __init__(self, lookback=96, horizon=96, hidden_size=32, num_layers=2):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = TimeSeriesLSTM(1, hidden_size, num_layers, 0.2, horizon)
            self.model.to(self.device)
            print(f"    LSTM using {self.device.upper()}: " + 
                  (torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU'))
            print(f"    Config: {hidden_size} hidden units, {num_layers} layers (memory optimized)")
        
        def fit(self, X_train, y_train, epochs=50, batch_size=16):
            # Split validation
            val_size = max(1, int(len(X_train) * 0.1))
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            X_train, y_train = X_train[:-val_size], y_train[:-val_size]
            
            # Training with smaller batches
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            self.model.train()
            
            # Process in small batches to save memory
            for epoch in range(epochs):
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    # Convert to tensors
                    batch_X = torch.FloatTensor(batch_X).unsqueeze(-1).to(self.device)
                    batch_y = torch.FloatTensor(batch_y).to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    # Clear cache
                    del batch_X, batch_y, outputs, loss
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
        
        def predict(self, X_test):
            self.model.eval()
            predictions = []
            
            # Predict in small batches
            batch_size = 16
            with torch.no_grad():
                for i in range(0, len(X_test), batch_size):
                    batch_X = X_test[i:i+batch_size]
                    batch_X = torch.FloatTensor(batch_X).unsqueeze(-1).to(self.device)
                    batch_pred = self.model(batch_X)
                    predictions.append(batch_pred.cpu().numpy())
                    
                    del batch_X, batch_pred
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
            
            return np.vstack(predictions)
    
    
    class DLinearForecaster:
        
        def __init__(self, lookback=96, horizon=96, kernel_size=25):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = DLinearPyTorch(lookback, horizon, kernel_size)
            self.model.to(self.device)
            print(f"    DLinear using {self.device.upper()}: " + 
                  (torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU'))
        
        def fit(self, X_train, y_train, epochs=30):
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            self.model.train()
            batch_size = 32  
            
            for epoch in range(epochs):
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    batch_X = torch.FloatTensor(batch_X).to(self.device)
                    batch_y = torch.FloatTensor(batch_y).to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    del batch_X, batch_y, outputs, loss
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
        
        def predict(self, X_test):
            self.model.eval()
            predictions = []
            batch_size = 32
            
            with torch.no_grad():
                for i in range(0, len(X_test), batch_size):
                    batch_X = X_test[i:i+batch_size]
                    batch_X = torch.FloatTensor(batch_X).to(self.device)
                    batch_pred = self.model(batch_X)
                    predictions.append(batch_pred.cpu().numpy())
                    
                    del batch_X, batch_pred
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
            
            return np.vstack(predictions)

else:
    class DLinearForecaster:
        def __init__(self, lookback=96, horizon=96, kernel_size=25):
            self.lookback = lookback
            self.horizon = horizon
            self.kernel_size = kernel_size
            print("    DLinear using NumPy (CPU)")
        
        def moving_average(self, X):
            pad = self.kernel_size // 2
            result = []
            for row in X:
                padded = np.pad(row, (pad, pad), mode='edge')
                kernel = np.ones(self.kernel_size) / self.kernel_size
                smoothed = np.convolve(padded, kernel, mode='valid')
                result.append(smoothed[:self.lookback])
            return np.array(result)
        
        def fit(self, X_train, y_train, epochs=30):
            trend = self.moving_average(X_train)
            residual = X_train - trend
            self.W_trend = np.linalg.lstsq(trend, y_train, rcond=None)[0]
            self.W_residual = np.linalg.lstsq(residual, y_train, rcond=None)[0]
        
        def predict(self, X_test):
            trend = self.moving_average(X_test)
            residual = X_test - trend
            return trend @ self.W_trend + residual @ self.W_residual
    
    class LSTMForecaster:
        def __init__(self, lookback=96, horizon=96, hidden_size=32, num_layers=2):
            self.horizon = horizon
            print("    LSTM using Simple Baseline (CPU)")
        
        def fit(self, X_train, y_train, epochs=50, batch_size=16):
            pass
        
        def predict(self, X_test):
            return np.repeat(X_test[:, -1:], self.horizon, axis=1)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-6
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


data_folder = 'data'
results_folder = 'results'
os.makedirs(results_folder, exist_ok=True)

files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])
print(f"Found {len(files)} datasets")
print(f"Using {'PyTorch GPU' if PYTORCH_AVAILABLE else 'fallback'}\n")

all_results = []

for file in files:
    print(f"{'='*60}")
    print(f"Processing: {file}")
    print('='*60)
    
    try:
        df = pd.read_csv(f'{data_folder}/{file}')
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            print("  No numeric columns - skipping\n")
            continue
        
        target_col = numeric_cols[0]
        series = df[target_col].values.astype(float)
        series = series[~np.isnan(series)]
        
        if len(series) < 500:
            print(f"  Insufficient data - skipping\n")
            continue
        
        print(f"  Target column: {target_col}")
        print(f"  Total points: {len(series)}")
        
        train_size = int(len(series) * 0.7)
        train = series[:train_size]
        test = series[train_size:]
        
        print(f"  Train: {len(train)}, Test: {len(test)}")
        
        mean, std = np.mean(train), np.std(train) + 1e-8
        train_n, test_n = (train - mean) / std, (test - mean) / std
        
        results = {'Dataset': file.replace('.csv', '')}
        
        # ARIMA
        print("\n  [1/3] ARIMA...")
        try:
            horizon = min(96, len(test))
            model = ARIMA(train_n, order=(5, 1, 0))
            fitted = model.fit()
            pred = fitted.forecast(steps=horizon)
            actual = test_n[:horizon]
            
            results['ARIMA_MAE'] = mae(actual, pred)
            results['ARIMA_RMSE'] = rmse(actual, pred)
            results['ARIMA_MAPE'] = mape(actual, pred)
            results['ARIMA_R2'] = r2_score(actual, pred)
            print(f"    MAE: {results['ARIMA_MAE']:.4f}, R²: {results['ARIMA_R2']:.4f}")
        except Exception as e:
            print(f"    Failed: {str(e)[:40]}")
            results.update({'ARIMA_MAE': np.nan, 'ARIMA_RMSE': np.nan, 
                          'ARIMA_MAPE': np.nan, 'ARIMA_R2': np.nan})
        
        lookback = min(96, len(train_n) // 10)
        horizon = min(96, len(test_n) // 2)
        
        X_train, y_train = [], []
        for i in range(len(train_n) - lookback - horizon + 1):
            X_train.append(train_n[i:i+lookback])
            y_train.append(train_n[i+lookback:i+lookback+horizon])
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        X_test, y_test = [], []
        for i in range(len(test_n) - lookback - horizon + 1):
            X_test.append(test_n[i:i+lookback])
            y_test.append(test_n[i+lookback:i+lookback+horizon])
        X_test, y_test = np.array(X_test), np.array(y_test)
        
        print(f"  Sequences: Train {len(X_train)}, Test {len(X_test)}")
        
        if len(X_train) < 10 or len(X_test) < 5:
            print("  Insufficient sequences\n")
            results.update({'DLinear_MAE': np.nan, 'DLinear_RMSE': np.nan,
                          'DLinear_MAPE': np.nan, 'DLinear_R2': np.nan,
                          'LSTM_MAE': np.nan, 'LSTM_RMSE': np.nan,
                          'LSTM_MAPE': np.nan, 'LSTM_R2': np.nan})
            all_results.append(results)
            continue
        
        # DLinear
        print("\n  [2/3] DLinear...")
        try:
            model_dl = DLinearForecaster(lookback, horizon, 25)
            model_dl.fit(X_train, y_train, epochs=30)
            pred_dl = model_dl.predict(X_test)
            
            results['DLinear_MAE'] = mae(y_test.flatten(), pred_dl.flatten())
            results['DLinear_RMSE'] = rmse(y_test.flatten(), pred_dl.flatten())
            results['DLinear_MAPE'] = mape(y_test.flatten(), pred_dl.flatten())
            results['DLinear_R2'] = r2_score(y_test.flatten(), pred_dl.flatten())
            print(f"    MAE: {results['DLinear_MAE']:.4f}, R²: {results['DLinear_R2']:.4f}")
        except Exception as e:
            print(f"    Failed: {str(e)[:40]}")
            results.update({'DLinear_MAE': np.nan, 'DLinear_RMSE': np.nan,
                          'DLinear_MAPE': np.nan, 'DLinear_R2': np.nan})
        
        # LSTM
        print("\n  [3/3] LSTM...")
        try:
            model_lstm = LSTMForecaster(lookback, horizon, 32, 2)  # Smaller: 32 units, 2 layers
            model_lstm.fit(X_train, y_train, epochs=50, batch_size=16)  # Smaller batch
            pred_lstm = model_lstm.predict(X_test)
            
            results['LSTM_MAE'] = mae(y_test.flatten(), pred_lstm.flatten())
            results['LSTM_RMSE'] = rmse(y_test.flatten(), pred_lstm.flatten())
            results['LSTM_MAPE'] = mape(y_test.flatten(), pred_lstm.flatten())
            results['LSTM_R2'] = r2_score(y_test.flatten(), pred_lstm.flatten())
            print(f"    MAE: {results['LSTM_MAE']:.4f}, R²: {results['LSTM_R2']:.4f}")
        except Exception as e:
            print(f"    Failed: {str(e)[:40]}")
            results.update({'LSTM_MAE': np.nan, 'LSTM_RMSE': np.nan,
                          'LSTM_MAPE': np.nan, 'LSTM_R2': np.nan})
        
        all_results.append(results)
        print()
        
    except Exception as e:
        print(f"  ERROR: {str(e)}\n")
        continue

if all_results:
    df_results = pd.DataFrame(all_results).round(4)
    output_file = f'{results_folder}/forecasting_results_final.csv'
    df_results.to_csv(output_file, index=False)
    
    print("="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(df_results.to_string(index=False))
    
    print("\n" + "="*60)
    print("AVERAGE PERFORMANCE")
    print("="*60)
    for model in ['ARIMA', 'DLinear', 'LSTM']:
        mae_col = f'{model}_MAE'
        r2_col = f'{model}_R2'
        if mae_col in df_results.columns:
            print(f"{model:8} - Avg MAE: {df_results[mae_col].mean():.4f}, Avg R²: {df_results[r2_col].mean():.4f}")
    
    print(f"\n Saved to: {output_file}")
    print("="*60)
else:
    print(" No results generated")
