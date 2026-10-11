import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'data' with columns 'x1' and 'x2'
# Replace 'file_path.csv' with the actual path to your dataset
data = pd.read_csv('Snow Data.csv')

# Extracting 'x1' and 'x2' columns
x1 = data['x1']
x2 = data['x2']

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x1, x2, alpha=0.5)
plt.xlabel('Duration of the Storm (x1)')
plt.ylabel('Time Spent on Snow Removal (x2)')
plt.title('Scatter Plot of x1 vs x2')
plt.grid(True)
plt.show()
