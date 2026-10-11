import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import probplot

# Assuming you have a DataFrame named 'data' with a column 'x2'
# Replace 'file_path.csv' with the actual path to your dataset
data = pd.read_csv('Snow Data.csv')

# Extracting 'x2' column
x2_values = data['x2']

# Finding the optimal lambda value for the Box-Cox transformation
transformed_x2, lambda_value = stats.boxcox(x2_values)

# Display the estimated lambda value
print(f"Estimated lambda value for transformation: {lambda_value:.4f}")

# Create a Q-Q plot for the transformed 'x2' values
plt.figure(figsize=(6, 6))
probplot(transformed_x2, dist="norm", plot=plt)
plt.title("Q-Q Plot of Transformed 'x2' Values")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)
plt.show()
