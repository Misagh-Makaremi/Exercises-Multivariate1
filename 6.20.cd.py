import pandas as pd
import numpy as np
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind

# Example data (replace this with your actual dataset)
data = pd.read_csv('Hook Billed Kite Data.csv')

df = pd.DataFrame(data)

# Separate the data into male and female groups
male_data = df[df['gender'] == 'male']
female_data = df[df['gender'] == 'female']

# Calculate means and standard deviations for male and female groups
mean_male = male_data[['x1', 'x2']].mean()
mean_female = female_data[['x1', 'x2']].mean()
std_male = male_data[['x1', 'x2']].std()
std_female = female_data[['x1', 'x2']].std()

# Calculate the difference of means and standard errors
diff_means = mean_male - mean_female
std_errors = np.sqrt((std_male**2) / len(male_data) + (std_female**2) / len(female_data))

# Degrees of freedom for t-distribution
df = len(male_data) + len(female_data) - 2

# Confidence level (e.g., 95%)
confidence_level = 0.95

# Calculate critical value for t-distribution
t_critical = t.ppf(confidence_level + (1 - confidence_level) / 2, df)

# Calculate confidence intervals
lower_bound = diff_means - t_critical * std_errors
upper_bound = diff_means + t_critical * std_errors

# Display results
print("Difference of means (Mu1 - Mu2):", diff_means)
print("95% Confidence Region for Mu1 - Mu2:")
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)


#6.20.d
# Perform independent samples t-test for each size variable
for column in ['x1', 'x2']:
    t_statistic, p_value = ttest_ind(male_data[column], female_data[column])
    
    print(f"T-test for {column}:")
    print(f"   T-statistic: {t_statistic}")
    print(f"   P-value: {p_value}")
    print()

    # Check the p-value to determine significance
    if p_value < 0.05:
        print(f"The difference in {column} between male and female birds is statistically significant.")
    else:
        print(f"There is no statistically significant difference in {column} between male and female birds.")