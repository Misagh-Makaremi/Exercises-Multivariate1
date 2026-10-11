import numpy as np
import pandas as pd
import statsmodels.api as sm

# Assuming you have the data in a DataFrame named 'df'
# Replace 'your_data.csv' with the actual data file or use your own data loading method
df = pd.read_csv('Number Parity Data.csv')

# Assuming you have columns x1, x2, x3, x4 in your DataFrame
# Replace them with the actual column names if different

# Design matrix
X = sm.add_constant(np.column_stack((np.ones(len(df)), df['x1'], df['x2'], df['x3'] * df['x4'])))

# Contrasts
contrasts = np.eye(4)

# Fit the model
for i in range(1,5):
    model = sm.OLS(df['x' + str(i)], X)
    results = model.fit()

    # Calculate simultaneous confidence intervals
    ci = results.conf_int(alpha=0.05, cols=None)

    # Display the results
    for i in range(4):
        print(f"Contrast {i + 1} CI: ({ci.iloc[i, 0]}, {ci.iloc[i, 1]})")

    interaction_ci = ci.iloc[3, :]
    if interaction_ci[0] <= 0 <= interaction_ci[1]:
        print("The confidence interval for the interaction effect includes zero, supporting the M model.")
    else:
        print("The confidence interval for the interaction effect does not include zero, supporting the C and C model.")