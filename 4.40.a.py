import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'data' with columns 'attendance' and 'size'
# Replace 'file_path.csv' with the actual path to your dataset
data = pd.read_csv('Attendance and Size of National Parks.csv')

# Extracting 'attendance' and 'size' columns
attendance = data['Visitors']
size = data['Size']

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(size, attendance, alpha=0.5)
plt.xlabel('Size (acres)')
plt.ylabel('Visitors (millions)')
plt.title('Scatter Plot of Visitors vs Size')
plt.grid(True)
plt.show()
