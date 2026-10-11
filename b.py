import pandas as pd
import numpy as np
import statsmodels.stats.multicomp as mc

# Load your CSV data into a DataFrame
# Assuming your CSV file is named 'data.csv'
data = pd.read_csv('Oxygen Data.csv')

# Separate the data into male and female groups
male_data = data[data['gender'] == 'Male']
female_data = data[data['gender'] == 'Female']

# Specify the columns corresponding to the variables x1, x2, x3, x4
columns = ['x1', 'x2', 'x3', 'x4']

# Combine the male and female data for each variable
groups = [male_data[col].values for col in columns] + [female_data[col].values for col in columns]

# Create labels for the groups (0 for males, 1 for females)
labels = [0] * len(male_data) + [1] * len(female_data)

# Perform the Tukey-Kramer test
result = mc.MultiComparison(np.concatenate(groups), labels)
tukey_results = result.tukeyhsd(alpha=0.05)

# Print the Tukey-Kramer intervals
print("Tukey-Kramer Intervals:")
print(tukey_results.summary())

# Calculate Bonferroni intervals
bonferroni_intervals = result.tconfintall(alpha=0.05, method='bonf')

# Print the Bonferroni intervals
print("\nBonferroni Intervals:")
print(pd.DataFrame(bonferroni_intervals, columns=columns))
