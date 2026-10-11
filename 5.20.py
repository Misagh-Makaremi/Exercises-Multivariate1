import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

#5.20.a
# Sample data for x1 (tail length) and x2 (wing length) of female hook-billed kites
# Replace 'x1_data' and 'x2_data' with your sample data
data = pd.read_csv('Bird Data.csv')
x1_data = data['x1']  # Insert your data for tail length (x1)
x2_data = data['x2']  # Insert your data for wing length (x2)

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

# Plot the confidence ellipse for female hook-billed kites
ellipse = Ellipse(xy=mean_vector, width=width, height=height, angle=rotation_angle, edgecolor='blue', fill=False)
fig, ax = plt.subplots(figsize=(8, 6))
ax.add_patch(ellipse)
ax.scatter(x1_data, x2_data, marker='o', color='red')  # Scatter plot of data points
plt.xlabel('Tail Length (x1)')
plt.ylabel('Wing Length (x2)')
plt.title('95% Confidence Ellipse for Female Hook-billed Kites')
plt.grid(True)

# Plausible values for male hook-billed kites
plt.axvline(x=190, color='green', linestyle='--', label='Mu1 for Male')
plt.axhline(y=275, color='purple', linestyle='--', label='Mu2 for Male')
plt.legend()
plt.show()


#5.20.b
from scipy.stats import t

# Sample sizes and sample mean vector
n = len(x1_data)
mean_vector = np.array([np.mean(x1_data), np.mean(x2_data)])

# Sample covariance matrix
cov_matrix = np.cov(x1_data, x2_data)

# Degrees of freedom for T² distribution
df = n - 1

# Critical value for T² distribution
alpha = 0.05
critical_value_T2 = np.sqrt((df * (n - 1)) / (df * (n - 1) - 2)) * t.ppf(1 - alpha, df=df)

# T²-intervals for Mu1 and Mu2
T2_interval_mu1 = np.sqrt(critical_value_T2 * (cov_matrix[0, 0] / n))
T2_interval_mu2 = np.sqrt(critical_value_T2 * (cov_matrix[1, 1] / n))

# Bonferroni intervals for Mu1 and Mu2
t_critical = t.ppf(1 - alpha / 2, df=n - 1)
Bonferroni_interval_mu1 = t_critical * np.sqrt(cov_matrix[0, 0] / n)
Bonferroni_interval_mu2 = t_critical * np.sqrt(cov_matrix[1, 1] / n)

print("Simultaneous 95% T²-intervals for Mu1 and Mu2:")
print(f"Mu1 interval: ({mean_vector[0] - T2_interval_mu1}, {mean_vector[0] + T2_interval_mu1})")
print(f"Mu2 interval: ({mean_vector[1] - T2_interval_mu2}, {mean_vector[1] + T2_interval_mu2})")

print("\n95% Bonferroni intervals for Mu1 and Mu2:")
print(f"Mu1 interval: ({mean_vector[0] - Bonferroni_interval_mu1}, {mean_vector[0] + Bonferroni_interval_mu1})")
print(f"Mu2 interval: ({mean_vector[1] - Bonferroni_interval_mu2}, {mean_vector[1] + Bonferroni_interval_mu2})")

#5.20.c
from scipy import stats

# Generate synthetic data for demonstration (Replace with your own data)
np.random.seed(42)
mean = [5, 10]  # Mean vector
covariance = [[2, 1.5], [1.5, 3]]  # Covariance matrix
data = np.random.multivariate_normal(mean, covariance, size=100)

# Extract variables from synthetic data
x1 = data[:, 0]
x2 = data[:, 1]

# Q-Q plots for x1 and x2
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
stats.probplot(x1, dist="norm", plot=axs[0])
axs[0].set_title('Q-Q Plot for x1')
stats.probplot(x2, dist="norm", plot=axs[1])
axs[1].set_title('Q-Q Plot for x2')

# Scatter diagram
plt.figure(figsize=(6, 6))
plt.scatter(x1, x2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of x1 vs x2')
plt.grid(True)
plt.show()
