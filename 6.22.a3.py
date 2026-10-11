import pandas as pd
from statsmodels.multivariate.manova import MANOVA

# Load the CSV file into a DataFrame
data = pd.read_csv("Oxygen Data.csv")

# Perform MANOVA
variables = ['x1', 'x2', 'x3', 'x4']
males_data = data[data['gender'] == '1'][variables]
females_data = data[data['gender'] == '2'][variables]

manova = MANOVA.from_formula(f'{"+ ".join(variables)} ~ gender', data=data)
manova_results = manova.mv_test()
print(manova_results.summary())

# Check if H0 is rejected
p_value = manova_results.results['gender']['stat']
alpha = 0.05
if all(p_value < alpha):
    print("\nReject H0: There are gender differences.")
    
    # Find the linear combination most responsible
    coeffs = manova_results.results['gender']['stat']
    max_responsible_combination = coeffs.idxmax()
    print(f"\nThe linear combination most responsible is: {max_responsible_combination}")
else:
    print("\nFail to reject H0: No significant gender differences.")