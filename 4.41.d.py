import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Assuming you have a DataFrame named 'data' with columns 'x1' and 'x2'
# Replace 'file_path.csv' with the actual path to your dataset
data = pd.read_csv('Snow Data.csv')

# Extracting 'x1' and 'x2' columns
x1_values = data['x1']
x2_values = data['x2']

# Applying Box-Cox transformations separately to 'x1' and 'x2'
transformed_x1, lambda_value_x1 = stats.boxcox(x1_values)
transformed_x2, lambda_value_x2 = stats.boxcox(x2_values)

# Display the estimated lambda values for 'x1' and 'x2'
print(f"Estimated lambda value for 'x1' transformation: {lambda_value_x1:.4f}")
print(f"Estimated lambda value for 'x2' transformation: {lambda_value_x2:.4f}")

# Visualize joint distribution of transformed 'x1' and 'x2'
plt.figure(figsize=(8, 6))
plt.scatter(transformed_x1, transformed_x2, alpha=0.5)
plt.xlabel("Transformed 'x1'")
plt.ylabel("Transformed 'x2'")
plt.title("Scatter Plot of Transformed 'x1' and 'x2'")
plt.grid(True)
plt.show()
