
import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# DLinear model
class DLinear:
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

# Metrics
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)

# Data loading
def load_and_prepare_data(filepath, lookback=96, horizon=96):
    
    try:
        df = pd.read_csv(filepath)
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            return None, None, None, None, None, None
        
        # Use first numeric column
        series = df[numeric_cols[0]].values.astype(float)
        series = series[~np.isnan(series)]
        
        if len(series) < 500:
            return None, None, None, None, None, None
        
        # Split
        train_size = int(len(series) * 0.7)
        val_size = int(len(series) * 0.1)
        
        train = series[:train_size]
        val = series[train_size:train_size+val_size]
        test = series[train_size+val_size:]
        
        # Normalize
        mean = np.mean(train)
        std = np.std(train) + 1e-8
        train_n = (train - mean) / std
        val_n = (val - mean) / std
        test_n = (test - mean) / std
        
        # Make sequences
        X_train, y_train = [], []
        for i in range(len(train_n) - lookback - horizon + 1):
            X_train.append(train_n[i:i+lookback])
            y_train.append(train_n[i+lookback:i+lookback+horizon])
        
        X_test, y_test = [], []
        for i in range(len(test_n) - lookback - horizon + 1):
            X_test.append(test_n[i:i+lookback])
            y_test.append(test_n[i+lookback:i+lookback+horizon])
        
        if len(X_train) < 10 or len(X_test) < 5:
            return None, None, None, None, None, None
        
        return (np.array(X_train), np.array(y_train), 
                np.array(X_test), np.array(y_test), mean, std)
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None, None, None, None, None

# experiment
data_folder = 'data'
results_folder = 'results'
os.makedirs(results_folder, exist_ok=True)

# Datasets to use
datasets = {
    'ETTh1': 'ETTh1.csv',
    'ETTh2': 'ETTh2.csv',
    'ETTm1': 'ETTm1.csv',
    'weather': 'weather.csv',
    'electricity': 'electricity.csv',
    'traffic': 'traffic.csv'
}

# Only use datasets that exist
available_datasets = {}
for name, filename in datasets.items():
    filepath = os.path.join(data_folder, filename)
    if os.path.exists(filepath):
        available_datasets[name] = filepath

if len(available_datasets) < 2:
    print("Available datasets:", list(available_datasets.keys()))
    exit()

print("="*70)
print("CROSS-DATASET GENERALIZATION EXPERIMENT")
print("="*70)
print(f"\nTesting generalization across {len(available_datasets)} datasets")
print("\nDatasets:", list(available_datasets.keys()))
print("="*70)

# Load all datasets
print("\nLoading datasets...")
dataset_cache = {}
for name, filepath in available_datasets.items():
    X_train, y_train, X_test, y_test, mean, std = load_and_prepare_data(filepath)
    if X_train is not None:
        dataset_cache[name] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'mean': mean,
            'std': std
        }
        print(f"  ✓ Loaded {name}: {len(X_train)} train sequences, {len(X_test)} test sequences")
    else:
        print(f"  ✗ Failed to load {name}")

if len(dataset_cache) < 2:
    print("\nNeed at least 2 valid datasets. Exiting.")
    exit()

# Run cross-dataset experiments
results = []

print("\n" + "="*70)
print("RUNNING EXPERIMENTS")
print("="*70)

for train_name in dataset_cache.keys():
    print(f"\n>>> Training on: {train_name}")
    
    # Train model
    train_data = dataset_cache[train_name]
    model = DLinear(lookback=96, horizon=96)
    model.fit(train_data['X_train'], train_data['y_train'])
    
    # Test on same dataset (baseline)
    pred_same = model.predict(train_data['X_test'])
    mae_same = mae(train_data['y_test'].flatten(), pred_same.flatten())
    r2_same = r2_score(train_data['y_test'].flatten(), pred_same.flatten())
    
    print(f"  Same-dataset (baseline): MAE={mae_same:.4f}, R²={r2_same:.4f}")
    
    results.append({
        'Train_Dataset': train_name,
        'Test_Dataset': train_name,
        'Type': 'Same-dataset (baseline)',
        'MAE': mae_same,
        'R2': r2_same
    })
    
    # Test on other datasets
    for test_name in dataset_cache.keys():
        if test_name == train_name:
            continue
        
        test_data = dataset_cache[test_name]
        pred_cross = model.predict(test_data['X_test'])
        mae_cross = mae(test_data['y_test'].flatten(), pred_cross.flatten())
        r2_cross = r2_score(test_data['y_test'].flatten(), pred_cross.flatten())
        
        # Performance drop
        mae_drop = ((mae_cross - mae_same) / mae_same) * 100 if mae_same > 0 else 0
        r2_drop = ((r2_same - r2_cross) / abs(r2_same)) * 100 if r2_same != 0 else 0
        
        print(f"  → Test on {test_name}: MAE={mae_cross:.4f} ({mae_drop:+.1f}%), R²={r2_cross:.4f} ({r2_drop:+.1f}%)")
        
        results.append({
            'Train_Dataset': train_name,
            'Test_Dataset': test_name,
            'Type': 'Cross-dataset',
            'MAE': mae_cross,
            'R2': r2_cross,
            'MAE_Change_%': mae_drop,
            'R2_Change_%': r2_drop
        })

# Save results
df_results = pd.DataFrame(results)
output_path = os.path.join(results_folder, 'cross_dataset_generalization.csv')
df_results.to_csv(output_path, index=False)

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

# Summary statistics
same_dataset = df_results[df_results['Type'] == 'Same-dataset (baseline)']
cross_dataset = df_results[df_results['Type'] == 'Cross-dataset']

if len(cross_dataset) > 0:
    print("\nPerformance Summary:")
    print(f"  Same-dataset average MAE: {same_dataset['MAE'].mean():.4f}")
    print(f"  Cross-dataset average MAE: {cross_dataset['MAE'].mean():.4f}")
    print(f"  Average MAE increase: {cross_dataset['MAE_Change_%'].mean():.1f}%")
    
    print(f"\n  Same-dataset average R²: {same_dataset['R2'].mean():.4f}")
    print(f"  Cross-dataset average R²: {cross_dataset['R2'].mean():.4f}")
    print(f"  Average R² drop: {cross_dataset['R2_Change_%'].mean():.1f}%")
    
    # Best generalizers
    print("\nBest Generalizing Models (lowest MAE drop in cross-dataset):")
    cross_avg = cross_dataset.groupby('Train_Dataset')['MAE_Change_%'].mean().sort_values()
    for dataset, drop in cross_avg.head(3).items():
        print(f"  {dataset}: {drop:.1f}% average MAE increase")
    
    # Dataset similarity
    print("\nMost Similar Dataset Pairs (best cross-dataset performance):")
    cross_sorted = cross_dataset.nsmallest(5, 'MAE')
    for _, row in cross_sorted.iterrows():
        print(f"  {row['Train_Dataset']} → {row['Test_Dataset']}: MAE={row['MAE']:.4f}")

print(f"\ Results saved to: {output_path}")

# Key findings
print("\n" + "="*70)
print("KEY FINDINGS ON GENERALIZATION")
print("="*70)

avg_mae_increase = cross_dataset['MAE_Change_%'].mean()

if avg_mae_increase < 20:
    finding = "GOOD: Models generalize well across datasets (<20% error increase)"
elif avg_mae_increase < 50:
    finding = "MODERATE: Some generalization, but performance drops significantly"
else:
    finding = "POOR: Models struggle to generalize (>50% error increase)"

print(f"\n{finding}")
print(f"Average performance drop: {avg_mae_increase:.1f}% MAE increase")

# Domain transfer insights
if len(cross_dataset) > 0:
    print("\nDomain Transfer Insights:")
    
    # Energy to energy
    energy_datasets = ['ETTh1', 'ETTh2', 'ETTm1']
    energy_transfers = cross_dataset[
        (cross_dataset['Train_Dataset'].isin(energy_datasets)) & 
        (cross_dataset['Test_Dataset'].isin(energy_datasets))
    ]
    
    if len(energy_transfers) > 0:
        print(f"  Within-domain (Energy→Energy): {energy_transfers['MAE_Change_%'].mean():.1f}% drop")
    
    # Cross-domain
    cross_domain = cross_dataset[
        ~((cross_dataset['Train_Dataset'].isin(energy_datasets)) & 
          (cross_dataset['Test_Dataset'].isin(energy_datasets)))
    ]
    
    if len(cross_domain) > 0:
        print(f"  Cross-domain transfers: {cross_domain['MAE_Change_%'].mean():.1f}% drop")

