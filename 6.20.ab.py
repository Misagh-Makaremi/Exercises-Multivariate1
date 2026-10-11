import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import f

# Assuming you have a DataFrame named 'df' with columns 'x1', 'x2', and 'gender'
# Replace 'your_data.csv' with the actual file or data source
data = pd.read_csv('Hook Billed Kite Data.csv')

df = pd.DataFrame(data)

# Filter the data for males and females
male_data = df[df['gender'] == 'male']
female_data = df[df['gender'] == 'female']

#6.20.a
# Plotting scatter diagram
plt.scatter(male_data['x1'], male_data['x2'], label='Male Hook-billed Kites')

# Highlight the outlier (observation 31)
outlier = male_data.iloc[30]
plt.scatter(outlier['x1'], outlier['x2'], color='red', label='Outlier (Observation 31)')

# Add labels and title
plt.xlabel('Tail Length (mm)')
plt.ylabel('Wing Length (mm)')
plt.title('Scatter Diagram for Male Hook-billed Kites with Outlier Highlight')

# Add legend
plt.legend()

# Show the plot
plt.show()

#6.20.b
# Convert gender to numeric (0 for female, 1 for male)
male_data['gender'] = np.where(male_data['gender'] == 'male', 1, 0)
female_data['gender'] = np.where(female_data['gender'] == 'female',1, 0)

# Check for outliers in the male hook-billed kite data
# You can use methods like z-scores, Mahalanobis distance, or visualization techniques

# Perform Hotelling's T-squared test
def hotelling_t_squared_test(df_male, df_female, alpha=0.05):
    n1 = len(df_male)
    n2 = len(df_female)
    p = 2  # Number of variables (tail_length and wing_length)

    sample_mean_diff = df_male[['x1', 'x2']].mean() - df_female[['x1', 'x2']].mean()
    
    pooled_covariance = ((n1 - 1) * df_male[['x1', 'x2']].cov() + (n2 - 1) * df_female[['x1', 'x2']].cov()) / (n1 + n2 - 2)

    t_squared_statistic = n1 * n2 / (n1 + n2) * np.dot(np.dot(sample_mean_diff.values.T, np.linalg.inv(pooled_covariance)), sample_mean_diff.values)

    # Degrees of freedom for the F-distribution
    df1 = p
    df2 = n1 + n2 - p - 1

    # Critical value from F-distribution
    critical_value = f.ppf(1 - alpha, df1, df2)

    # Compare the test statistic with the critical value
    if t_squared_statistic > critical_value:
        print("Reject the null hypothesis. There is a significant difference between male and female hook-billed kites.")
        
        # Extract the linear combination most responsible for the rejection of H0
        linear_combination = np.dot(np.linalg.inv(pooled_covariance), sample_mean_diff.values)
        print("Linear Combination Coefficients:", linear_combination)

    else:
        print("Fail to reject the null hypothesis. There is no significant difference between male and female hook-billed kites.")

# Call the function with your data
hotelling_t_squared_test(male_data, female_data)