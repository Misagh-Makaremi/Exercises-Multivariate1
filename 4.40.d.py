import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Assuming you have a DataFrame named 'data' with columns 'Size' and 'Visitors'
# Replace 'file_path.csv' with the actual path to your dataset
data = pd.read_csv('Attendance and Size of National Parks.csv')

# Extracting 'Size' and 'Visitors' columns
Size_values = data['Size']
Visitors_values = data['Visitors']

# Applying Box-Cox transformations separately to 'Size' and 'Visitors'
transformed_Size, lambda_value_Size = stats.boxcox(Size_values)
transformed_Visitors, lambda_value_Visitors = stats.boxcox(Visitors_values)

# Display the estimated lambda values for 'Size' and 'Visitors'
print(f"Estimated lambda value for 'Size' transformation: {lambda_value_Size:.4f}")
print(f"Estimated lambda value for 'Visitors' transformation: {lambda_value_Visitors:.4f}")

# Visualize joint distribution of transformed 'Size' and 'Visitors'
plt.figure(figsize=(8, 6))
plt.scatter(transformed_Size, transformed_Visitors, alpha=0.5)
plt.xlabel("Transformed 'Size'")
plt.ylabel("Transformed 'Visitors'")
plt.title("Scatter Plot of Transformed 'Size' and 'Visitors'")
plt.grid(True)
plt.show()
