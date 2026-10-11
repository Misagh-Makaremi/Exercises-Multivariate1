import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import probplot

# Assuming you have a DataFrame named 'data' with a column 'x1'
# Replace 'file_path.csv' with the actual path to your dataset
data = pd.read_csv('Snow Data.csv')

# Extracting 'x1' column
x1_values = data['x1']

# Finding the optimal lambda value for the Box-Cox transformation
transformed_x1, lambda_value = stats.boxcox(x1_values)

# Display the estimated lambda value
print(f"Estimated lambda value for transformation: {lambda_value:.4f}")

# Create a Q-Q plot for the transformed 'x1' values
plt.figure(figsize=(6, 6))
probplot(transformed_x1, dist="norm", plot=plt)
plt.title("Q-Q Plot of Transformed 'x1' Values")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)
plt.show()
