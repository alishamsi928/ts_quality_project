
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class DLinearPyTorch(nn.Module):
   
    
    def __init__(self, lookback=96, horizon=96, kernel_size=25):
        super(DLinearPyTorch, self).__init__()
        
        self.lookback = lookback
        self.horizon = horizon
        self.kernel_size = kernel_size
        
        # Linear layers for each component
        self.trend_layer = nn.Linear(lookback, horizon, bias=True)
        self.residual_layer = nn.Linear(lookback, horizon, bias=True)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.trend_layer.weight)
        nn.init.xavier_uniform_(self.residual_layer.weight)
        
    def moving_average(self, x):
       
        batch_size = x.size(0)
        
        # Padding
        pad = self.kernel_size // 2
        x_padded = torch.nn.functional.pad(x, (pad, pad), mode='replicate')
        
        # Moving average using unfold
        x_padded = x_padded.unfold(dimension=1, size=self.kernel_size, step=1)
        trend = x_padded.mean(dim=2)
        
        # Trim to exact lookback length
        return trend[:, :self.lookback]
    
    def forward(self, x):
       
        # Step 1: Decompose into trend and residual
        trend = self.moving_average(x)
        residual = x - trend
        
        # Step 2: Apply separate linear transformations
        trend_output = self.trend_layer(trend)
        residual_output = self.residual_layer(residual)
        
        # Step 3: Combine predictions
        output = trend_output + residual_output
        
        return output
    
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=30, batch_size=32, learning_rate=0.001,
            device='cuda', verbose=True):
      
        # Move model to device
        self.to(device)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data if provided
        if X_val is not None and y_val is not None:
            X_val = torch.FloatTensor(X_val)
            y_val = torch.FloatTensor(y_val)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_loader = None
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                predictions = self(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        
                        predictions = self(batch_X)
                        loss = criterion(predictions, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f'    Epoch [{epoch+1}/{epochs}], '
                          f'Train Loss: {train_loss:.4f}, '
                          f'Val Loss: {val_loss:.4f}')
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f'    Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}')
    
    def predict(self, X_test, device='cuda', batch_size=32):
       
        self.eval()
        self.to(device)
        
        # Convert to tensor
        X_test = torch.FloatTensor(X_test)
        test_dataset = TensorDataset(X_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        predictions = []
        
        with torch.no_grad():
            for (batch_X,) in test_loader:
                batch_X = batch_X.to(device)
                outputs = self(batch_X)
                predictions.append(outputs.cpu().numpy())
        
        return np.vstack(predictions)


class DLinearForecaster:
   
    
    def __init__(self, lookback=96, horizon=96, kernel_size=25):
        self.lookback = lookback
        self.horizon = horizon
        self.kernel_size = kernel_size
        
        # Detect device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create model
        self.model = DLinearPyTorch(
            lookback=lookback,
            horizon=horizon,
            kernel_size=kernel_size
        )
        
        if self.device == 'cuda':
            print(f"    DLinear using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"    DLinear using CPU")
    
    def fit(self, X_train, y_train, epochs=30, verbose=False):
       
        # Split validation set (10%)
        val_size = max(1, int(len(X_train) * 0.1))
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        # Train
        self.model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=32,
            learning_rate=0.001,
            device=self.device,
            verbose=verbose
        )
    
    def predict(self, X_test):
        
        return self.model.predict(X_test, device=self.device)


# Comparison with NumPy version
class DLinearNumPy:
    
    
    def __init__(self, lookback=96, horizon=96, kernel_size=25):
        self.lookback = lookback
        self.horizon = horizon
        self.kernel_size = kernel_size
        self.W_trend = None
        self.W_residual = None
    
    def moving_average(self, X):
        pad = self.kernel_size // 2
        result = []
        for row in X:
            padded = np.pad(row, (pad, pad), mode='edge')
            kernel = np.ones(self.kernel_size) / self.kernel_size
            smoothed = np.convolve(padded, kernel, mode='valid')
            result.append(smoothed[:self.lookback])
        return np.array(result)
    
    def fit(self, X_train, y_train):
        trend = self.moving_average(X_train)
        residual = X_train - trend
        self.W_trend = np.linalg.lstsq(trend, y_train, rcond=None)[0]
        self.W_residual = np.linalg.lstsq(residual, y_train, rcond=None)[0]
    
    def predict(self, X_test):
        trend = self.moving_average(X_test)
        residual = X_test - trend
        return trend @ self.W_trend + residual @ self.W_residual


if __name__ == "__main__":
    print("="*70)
    print("DLinear Implementation Comparison")
    print("="*70)
    
    # Generate test data
    np.random.seed(42)
    n_samples = 500
    X_train = np.random.randn(n_samples, 96)
    y_train = np.random.randn(n_samples, 96)
    X_test = np.random.randn(50, 96)
    
    print("\n1. NumPy DLinear (Your Current Implementation)")
    print("-" * 70)
    model_numpy = DLinearNumPy(lookback=96, horizon=96)
    
    import time
    start = time.time()
    model_numpy.fit(X_train, y_train)
    pred_numpy = model_numpy.predict(X_test)
    time_numpy = time.time() - start
    
    print(f"Training + Prediction time: {time_numpy:.3f}s")
    print(f"   Predictions shape: {pred_numpy.shape}")
    
    print("\n2. PyTorch DLinear (GPU-Accelerated)")
    print("-" * 70)
    model_pytorch = DLinearForecaster(lookback=96, horizon=96)
    
    start = time.time()
    model_pytorch.fit(X_train, y_train, epochs=30, verbose=False)
    pred_pytorch = model_pytorch.predict(X_test)
    time_pytorch = time.time() - start
    
    print(f"Training + Prediction time: {time_pytorch:.3f}s")
    print(f"   Predictions shape: {pred_pytorch.shape}")
    
    print("\n" + "="*70)
    print("Comparison:")
    print(f"  NumPy:   {time_numpy:.3f}s")
    print(f"  PyTorch: {time_pytorch:.3f}s")
    
    if time_pytorch < time_numpy:
        speedup = time_numpy / time_pytorch
        print(f"  PyTorch is {speedup:.1f}x faster")
    else:
        slowdown = time_pytorch / time_numpy
        print(f"  NumPy is {slowdown:.1f}x faster (on small data, NumPy can be faster!)")
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("="*70)
    print("\n Both implementations are scientifically valid!")
    print("\n Use NumPy version if:")
    print("   - You want simplicity and transparency")
    print("   - Your datasets are small-medium (< 100k samples)")
    print("   - You want to save time")
    print("\n Use PyTorch version if:")
    print("   - You want all models in same framework")
    print("   - You have very large datasets")
    print("   - You want GPU acceleration")
    print("\n HONEST TRUTH: For your project, NumPy is perfectly fine!")
    print("="*70)
