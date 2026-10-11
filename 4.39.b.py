import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.stats import chi2

# Load the dataset containing the specified variables
# Assuming the dataset is loaded into a DataFrame named 'psych_data'
# Replace 'file_path.csv' with the actual path to your dataset
psych_data = pd.read_csv('Psychological Profile Data.csv')

# Select specified variables
selected_vars = ['Indep', 'Supp', 'Benav', 'Conform', 'Leader']
selected_data = psych_data[selected_vars]

# Calculate skewness and kurtosis for the selected data
skewness = skew(selected_data)
kurt = kurtosis(selected_data)

# Calculate Mardia's test statistic
n = len(selected_data)
mardia_test_statistic = (n * (n + 1) * (n + 1)) / 6.0 * (skewness.dot(skewness) + kurt.dot(kurt) / 4.0)

# Degrees of freedom for chi-square distribution
df = len(selected_vars) * (len(selected_vars) + 2) * (len(selected_vars) + 1) / 6.0

# Calculate critical value for Mardia's test statistic using chi-square distribution
alpha = 0.05  # Significance level
critical_value = chi2.ppf(1 - alpha, df)

# Determine if the data deviates significantly from multivariate normality
if mardia_test_statistic > critical_value:
    print("The data deviates significantly from multivariate normality.")
else:
    print("The data shows no significant deviation from multivariate normality.")
