# Time Series Benchmark Dataset Quality Analysis
 
 **project**: Evaluating the quality and representativeness of benchmark datasets used in time series forecasting.

**Supervisor:** Michael Stenger  
**Institution:** University of Würzburg - Chair of Software Engineering  
**Year:** 2026

---

## Overview

This project critically evaluates 11 widely-used deep learning benchmark datasets for time series forecasting. The goal is to determine which datasets are actually good quality and representative, and whether dataset quality impacts forecasting model performance.

## Datasets Analyzed

- **ETTh1, ETTh2, ETTm1, ETTm2** - Electricity Transformer Temperature (hourly and 15-min)
- **Electricity** - Electricity consumption
- **Exchange Rate** - Currency exchange rates
- **Traffic** - Road traffic data
- **Weather** - Weather measurements
- **ILI** - Influenza-Like Illness surveillance

## DQEF Framework

The Dataset Quality Evaluation Framework (DQEF) scores datasets across 5 dimensions:

1. **Validity** (25%) - Missing values, duplicates, timestamp regularity
2. **Statistical Integrity** (25%) - Stationarity (ADF), Seasonality (ACF)
3. **Variability** (15%) - Coefficient of variation
4. **Diversity** (20%) - Channel correlation (MAPC metric)
5. **Completeness** (15%) - Data volume, timespan

## Models Tested

- **ARIMA** - Classical statistical baseline
- **DLinear** - Simple linear decomposition model (2023)
- **LSTM** - Recurrent neural network

Metrics: MAE, RMSE, MAPE, R²

## Requirements

```
pandas
numpy
matplotlib
seaborn
statsmodels
scipy
```

## Results

All results are saved in the `results/` folder:

Visualizations in `plots/`:
- Correlation heatmaps
- Quality vs performance scatter plots

---

**Contact:** ali.anwar@stud-mail.uni-wuerzburg.de
