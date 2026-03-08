
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

data_folder = 'data'
output_folder = 'plots'
os.makedirs(output_folder, exist_ok=True)

files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

for file in files:
    print(f"\n=== {file} ===")
    df = pd.read_csv(f'{data_folder}/{file}')
    
    # Get numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.shape[1] < 2:
        print("  Only 1 numeric column - skipping correlation analysis")
        continue
    
    # Correlation matrix
    corr_matrix = numeric_df.corr(method='pearson')
    print(f"  Channels: {numeric_df.shape[1]}")
    
    # Get upper triangle values (avoid diagonal and duplicates)
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_triangle = corr_matrix.where(mask)
    pairwise_corrs = upper_triangle.stack().values
    
    # MAPC - Mean Absolute Pairwise Correlation
    mapc = np.mean(np.abs(pairwise_corrs))
    print(f"  MAPC: {mapc:.4f}")
    
    
    if mapc < 0.3:
        diversity = "High diversity"
    elif mapc < 0.6:
        diversity = "Medium diversity"  
    else:
        diversity = "Low diversity (redundant channels)"
    
    print(f"  {diversity}")
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, 
                cbar_kws={'label': 'Correlation'})
    plt.title(f'{file} - Channel Correlation Heatmap')
    plt.tight_layout()
    
    # Save
    plot_name = file.replace('.csv', '_correlation.png')
    plt.savefig(f'{output_folder}/{plot_name}')
    plt.close()
    print(f"  Saved heatmap: {plot_name}")
