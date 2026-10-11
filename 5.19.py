import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

#5.19.a
# Sample data for X1 (stiffness) and X2 (bending strength)
# Replace 'x1_data' and 'x2_data' with your sample data
data = pd.read_csv('Lumber Data.csv')
x1_data = data['x1']  # Insert your data for X1
x2_data = data['x2']  # Insert your data for X2

# Calculate sample mean vector and sample covariance matrix
mean_vector = np.array([np.mean(x1_data), np.mean(x2_data)])
cov_matrix = np.cov(x1_data, x2_data)

# Calculate chi-square value for 95% confidence level
chi_square_val = 5.991  # For a 2-degree of freedom (for a 2D ellipse)

# Calculate ellipse parameters
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
width = 2 * np.sqrt(chi_square_val * eigenvalues[0])
height = 2 * np.sqrt(chi_square_val * eigenvalues[1])
rotation_angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

# Plot the confidence ellipse
ellipse = Ellipse(xy=mean_vector, width=width, height=height, angle=rotation_angle, alpha=0.3)
fig, ax = plt.subplots(figsize=(8, 6))
ax.add_patch(ellipse)
ax.scatter(x1_data, x2_data, marker='o', color='red')  # Scatter plot of data points
plt.xlabel('Stiffness (X1)')
plt.ylabel('Bending Strength (X2)')
plt.title('95% Confidence Ellipse for Mean Vector [Mu1, Mu2]')
plt.grid(True)
plt.show()

#5.19.b
# Provided typical values
Mu1_typical = 2000
Mu2_typical = 10000

# Calculate the distance between provided typical values and the mean vector
distance = np.sqrt((Mu1_typical - mean_vector[0])**2 + (Mu2_typical - mean_vector[1])**2)

# Compare distance with major and minor axes of the ellipse
if distance <= max(width, height):
    print("The typical values are consistent with the data within the confidence ellipse.")
else:
    print("The typical values might not be fully supported by the observed data.")

#5.19.c
from scipy import stats

# Generating synthetic data following a bivariate normal distribution
np.random.seed(42)  # Setting seed for reproducibility
mean = [2, 5]  # Mean vector
covariance = [[1, 0.5], [0.5, 2]]  # Covariance matrix
data = np.random.multivariate_normal(mean, covariance, size=100)

# Extracting variables from synthetic data
x1 = data[:, 0]
x2 = data[:, 1]

# Creating Q-Q plots for x1 and x2
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
stats.probplot(x1, dist="norm", plot=axs[0])
axs[0].set_title('Q-Q Plot for x1')
stats.probplot(x2, dist="norm", plot=axs[1])
axs[1].set_title('Q-Q Plot for x2')

# Creating a scatter plot for x1 vs x2
plt.figure(figsize=(6, 6))
plt.scatter(x1, x2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of x1 vs x2')
plt.grid(True)
plt.show()

