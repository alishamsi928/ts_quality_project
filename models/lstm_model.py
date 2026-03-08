
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class TimeSeriesLSTM(nn.Module):
    
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, 
                 dropout=0.2, horizon=96):
        super(TimeSeriesLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, horizon)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        # LSTM forward pass
        out, (hidden, cell) = self.lstm(x)
        
        # Take the last output from the sequence
        last_output = out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Pass through fully connected layer
        predictions = self.fc(last_output)
        
        return predictions
    
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=50, batch_size=32, learning_rate=0.001,
            device='cuda', patience=10, verbose=True):
        
        # Move model to device
        self.to(device)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)
        y_train_tensor = torch.FloatTensor(y_train)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data if provided
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).unsqueeze(-1)
            y_val_tensor = torch.FloatTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_loader = None
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        
                        outputs = self(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_state = self.state_dict()
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f'    Epoch [{epoch+1}/{epochs}], '
                          f'Train Loss: {train_loss:.4f}, '
                          f'Val Loss: {val_loss:.4f}')
                
                # Early stopping
                if patience_counter >= patience:
                    if verbose:
                        print(f'    Early stopping at epoch {epoch+1}')
                    if hasattr(self, 'best_state'):
                        self.load_state_dict(self.best_state)
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f'    Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}')
    
    def predict(self, X_test, device='cuda', batch_size=32):
       
        self.eval()
        self.to(device)
        
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(-1)
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        predictions = []
        
        with torch.no_grad():
            for (batch_X,) in test_loader:
                batch_X = batch_X.to(device)
                outputs = self(batch_X)
                predictions.append(outputs.cpu().numpy())
        
        return np.vstack(predictions)


class LSTMForecaster:
   
    
    def __init__(self, lookback=96, horizon=96, hidden_size=128, num_layers=3):
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Detect device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create model
        self.model = TimeSeriesLSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            horizon=horizon
        )
        
        if self.device == 'cuda':
            print(f"    Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"    Using CPU (GPU not available)")
    
    def fit(self, X_train, y_train, epochs=50, batch_size=64, verbose=True):
        
        # Split validation set
        val_size = max(1, int(len(X_train) * 0.1))
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        if verbose:
            print(f"    Training LSTM on {self.device.upper()}...")
            print(f"    Train: {len(X_train)}, Val: {len(X_val)} sequences")
        
        # Train
        self.model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.001,
            device=self.device,
            patience=10
        )
    
    def predict(self, X_test):
       
        return self.model.predict(X_test, device=self.device, batch_size=64)


# Test GPU availability
if __name__ == "__main__":
    print("="*60)
    print("PyTorch LSTM Setup Check")
    print("="*60)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # GPU test
        x = torch.randn(10, 10).cuda()
        print(f"\n✅ GPU is working! Test tensor device: {x.device}")
    else:
        print("\n⚠️  GPU not detected - will use CPU")
        print("To use GPU, install PyTorch with CUDA:")
        print("pip install torch --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n" + "="*60)
    print("Ready to use! Import with:")
    print("from lstm_pytorch import LSTMForecaster")
    print("="*60)
