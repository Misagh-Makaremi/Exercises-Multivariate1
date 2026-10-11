import numpy as np
import pandas as pd
from scipy.stats import shapiro

# Assuming you have the data in a DataFrame named 'df'
# Replace 'your_data.csv' with the actual data file or use your own data loading method
df = pd.read_csv('Number Parity Data.csv')

# Assuming you have columns x1, x2, x3, x4 in your DataFrame
# Replace them with the actual column names if different

# Calculate difference scores
df['NumberFormatContrast'] = df['x1'] - df['x2']
df['ParityTypeContrast'] = df['x3'] - df['x4']
df['InteractionContrast'] = df['x1'] - df['x2'] - (df['x3'] - df['x4'])

# Combine the difference scores into a matrix
difference_scores_matrix = df[['NumberFormatContrast', 'ParityTypeContrast', 'InteractionContrast']].to_numpy()

# Perform Shapiro-Wilk test for multivariate normality
stat, p_value = shapiro(difference_scores_matrix)

# Display the results
print(f"Shapiro-Wilk test statistic: {stat}")
print(f"P-value: {p_value}")

# Interpretation
alpha = 0.05
if p_value > alpha:
    print("The difference scores likely follow a multivariate normal distribution.")
else:
    print("The difference scores do not follow a multivariate normal distribution.")
