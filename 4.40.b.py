import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import probplot

# Assuming you have a DataFrame named 'data' with a column 'Size'
# Replace 'file_path.csv' with the actual path to your dataset
data = pd.read_csv('Attendance and Size of National Parks.csv')

# Extracting 'Size' column
Size_values = data['Size']

# Finding the optimal lambda value for the Box-Cox transformation
transformed_Size, lambda_value = stats.boxcox(Size_values)

# Display the estimated lambda value
print(f"Estimated lambda value for transformation: {lambda_value:.4f}")

# Create a Q-Q plot for the transformed 'Size' values
plt.figure(figsize=(6, 6))
probplot(transformed_Size, dist="norm", plot=plt)
plt.title("Q-Q Plot of Transformed 'Size' Values")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)
plt.show()
