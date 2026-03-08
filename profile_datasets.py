
import pandas as pd
import os

data_folder = 'data'

# Get all files
files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
print(f"Found {len(files)} datasets\n")

results = []

for file in files:
    print(f"Profiling {file}...")
    df = pd.read_csv(f'{data_folder}/{file}')
    
    # Basic stats
    rows = df.shape[0]
    cols = df.shape[1]
    
    # Missing values
    total_cells = rows * cols
    missing = df.isnull().sum().sum()
    missing_pct = (missing / total_cells) * 100
    
    # Duplicates
    duplicates = df.duplicated().sum()
    dup_pct = (duplicates / rows) * 100 if rows > 0 else 0
    
   
    results.append({
        'Dataset': file,
        'Rows': rows,
        'Columns': cols,
        'Missing (%)': round(missing_pct, 2),
        'Duplicates (%)': round(dup_pct, 2)
    })
    
    print(f"  Rows: {rows}, Cols: {cols}, Missing: {missing_pct:.2f}%\n")

# Save results
results_df = pd.DataFrame(results)
print(results_df)

# Save to CSV
os.makedirs('results', exist_ok=True)
results_df.to_csv('results/basic_profile.csv', index=False)
print("\nSaved to results/basic_profile.csv")
