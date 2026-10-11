import pandas as pd
from scipy.stats import f_oneway

# Load the dataset from the CSV file
df = pd.read_csv('Number Parity Data.csv')

# Perform repeated measures ANOVA
anova_result = f_oneway(df['x1'], df['x2'], df['x3'], df['x4'])

# Display the ANOVA results
print("ANOVA results:")
print(anova_result)

# Check if the p-value is less than the significance level (alpha)
alpha = 0.05
if anova_result.pvalue < alpha:
    print("Reject the null hypothesis. There are significant treatment effects.")
else:
    print("Fail to reject the null hypothesis. No significant treatment effects.")
