import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot

# Load the dataset containing the specified variables
# Assuming the dataset is loaded into a DataFrame named 'psych_data'
# Replace 'file_path.csv' with the actual path to your dataset
psych_data = pd.read_csv('Psychological Profile Data.csv')

# Select specified variables
selected_vars = ['Indep', 'Supp', 'Benav', 'Conform', 'Leader']
selected_data = psych_data[selected_vars]

# Plot histograms and Q-Q plots for each variable
for var in selected_vars:
    plt.figure(figsize=(10, 4))
    
    # Histogram
    plt.subplot(1, 2, 1)
    selected_data[var].hist(bins=10)
    plt.title(f"Histogram for {var}")
    
    # Q-Q plot
    plt.subplot(1, 2, 2)
    probplot(selected_data[var], dist="norm", plot=plt)
    plt.title(f"Q-Q plot for {var}")
    
    plt.tight_layout()
    plt.show()
