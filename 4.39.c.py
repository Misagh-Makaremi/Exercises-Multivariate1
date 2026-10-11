import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot
from scipy.stats import boxcox

# Load the dataset containing the specified variables
# Assuming the dataset is loaded into a DataFrame named 'psych_data'
# Replace 'file_path.csv' with the actual path to your dataset
psych_data = pd.read_csv('Psychological Profile Data.csv')

# Select specified variables
selected_vars = ['Indep', 'Supp', 'Benav', 'Conform', 'Leader']
selected_data = psych_data[selected_vars]

# Apply Box-Cox transformation and check for improved normality
for var in selected_vars:
    transformed_data, lambda_value = boxcox(selected_data[var] + 1 - min(selected_data[var]))  # Adding 1 to handle zero values
    
    # Plot histograms and Q-Q plots for transformed data
    plt.figure(figsize=(10, 4))
    
    # Histogram of transformed data
    plt.subplot(1, 2, 1)
    plt.hist(transformed_data, bins=10)
    plt.title(f"Transformed Histogram for {var}")
    
    # Q-Q plot of transformed data
    plt.subplot(1, 2, 2)
    probplot(transformed_data, dist="norm", plot=plt)
    plt.title(f"Transformed Q-Q plot for {var} (lambda={lambda_value:.3f})")
    
    plt.tight_layout()
    plt.show()
