import pandas as pd
from statsmodels.multivariate.manova import MANOVA

# Load the data from the CSV file
data = pd.read_csv('Oxygen Data.csv')

# Specify the variables
variables = ['x1', 'x2', 'x3', 'x4']

# Set up the MANOVA model and perform the test
manova_model = MANOVA(data[variables], data['gender'])
manova_results = manova_model.mv_test()

# Extract Pillai's Trace from the results
pillai_trace = manova_results.pillai_stat

# Print the results
print(manova_results)

# Check if the overall MANOVA test is significant
alpha = 0.05
if pillai_trace < alpha:
    print("Reject H0: Mu1-Mu2=0 (There are gender differences)")
else:
    print("Fail to reject H0: Mu1-Mu2=0 (No significant gender differences)")

# If you reject the null hypothesis, find the linear combination most responsible
if pillai_trace < alpha:
    # Get the coefficients of the linear combination
    coefficients = manova_results.coefs

    # Print the coefficients for the linear combination most responsible
    print("Linear combination coefficients:")
    print(coefficients)