import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import anderson

# Load the dataset containing the specified variables
# Assuming the dataset is loaded into a DataFrame named 'bulls_data'
# Replace 'file_path.csv' with the actual path to your dataset
bulls_data = pd.read_csv('Data on Bulls.csv')

# Selecting specified variables
selected_vars = ['YrHgt', 'FtFrBody', 'PrctFFB', 'BkFat', 'SaleHt', 'SaleWt']
selected_data = bulls_data[selected_vars]

# Plot histograms for each variable
selected_data.hist(bins=10, figsize=(12, 8))
plt.tight_layout()
plt.show()

# Perform Anderson-Darling test for multivariate normality
for var in selected_vars:
    result = anderson(selected_data[var])
    print(f"Anderson-Darling test for {var}: Statistic: {result.statistic}, Critical Values: {result.critical_values}, Significance Levels: {result.significance_level}")
