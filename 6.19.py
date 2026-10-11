import pandas as pd
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Assuming you have the data in a DataFrame named 'df'
# Replace 'your_data.csv' with the actual data file or use your own data loading method
df = pd.read_csv('Milk Transportation-Cost Data.csv')

# Assuming you have columns x1, x2, x3, and type in your DataFrame
# Replace them with the actual column names if different

# Filter data for gasoline trucks (type 1)
gasoline_trucks = df[df['type'] == 1][['x1', 'x2', 'x3']]

# Filter data for diesel trucks (type 2)
diesel_trucks = df[df['type'] == 2][['x1', 'x2', 'x3']]

# Perform MANOVA
manova_results = MANOVA.from_formula('x1 + x2 + x3 ~ type', data=df)

# Display the results
print(manova_results)

# Extract the p-value from the MANOVA results
p_value = manova_results.mv_test().results['type']['stat'].values[0]

# Interpretation
alpha = 0.01
print(p_value)
if all(p_value < alpha):
    print("There are significant differences in the mean cost vectors between gasoline and diesel trucks.")
else:
    print("There are no significant differences in the mean cost vectors between gasoline and diesel trucks.\n")

# Perform t-tests for each dependent variable
results = []
for col in ['x1', 'x2', 'x3']:
    t_stat, p_value = ttest_ind(gasoline_trucks[col], diesel_trucks[col])
    results.append({'Variable': col, 't-statistic': t_stat, 'p-value': p_value})
    print(f'Test for {col}: t-statistic = {t_stat}, p-value = {p_value}')
print('\n')

# Adjust p-values for multiple comparisons using Bonferroni correction
adjusted_p_values = multipletests([result['p-value'] for result in results], alpha=0.01, method='bonferroni')[1]

# Display the adjusted p-values
for i, result in enumerate(results):
    result['Adjusted p-value'] = adjusted_p_values[i]
    print(result)
print('\n')

# Drop multivariate outliers (observations 9 and 21 for gasoline trucks)
df_cleaned = df.drop(index=[8, 20])

# Perform MANOVA on the cleaned dataset
manova_results_cleaned = MANOVA.from_formula('x1 + x2 + x3 ~ type', data=df_cleaned)

# Display the MANOVA results for the cleaned dataset
print(manova_results_cleaned)

# Extract the p-value from the MANOVA results
p_value_cleaned = manova_results_cleaned.mv_test().results['type']['stat'].values[0]

# Interpretation
alpha = 0.01
print(p_value_cleaned)
if all(p_value_cleaned < alpha):
    print("There are significant differences in the mean cost vectors between gasoline and diesel trucks.")
else:
    print("There are no significant differences in the mean cost vectors between gasoline and diesel trucks.")