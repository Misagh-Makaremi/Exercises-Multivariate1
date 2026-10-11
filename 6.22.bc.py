import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# Sample data
# Replace this with your actual dataset
data = pd.read_csv('Oxygen Data.csv')

df = pd.DataFrame(data)

# Separate data for males and females
male_data = df[df['gender'] == 1]
female_data = df[df['gender'] == 2]

# Number of comparisons
num_comparisons = 4

# Significance level
alpha = 0.05

# Construct simultaneous confidence intervals
conf_intervals = []
bonferroni_intervals = []

for i in range(1, num_comparisons + 1):
    x_male = male_data[f'x{i}']
    x_female = female_data[f'x{i}']

    # Using the t-test to compare means
    t_stat, p_value, _ = sm.stats.ttest_ind(x_male, x_female)

    # Bonferroni correction
    bonferroni_alpha = alpha / num_comparisons

    # Confidence interval
    conf_interval = sm.stats.CompareMeans(sm.stats.DescrStatsW(x_male), sm.stats.DescrStatsW(x_female)).tconfint_diff(alpha=bonferroni_alpha, alternative='two-sided')

    conf_intervals.append(conf_interval)
    bonferroni_intervals.append((conf_interval[0] - 1, conf_interval[1] + 1))

# Print results
for i in range(num_comparisons):
    print(f"Comparison {i + 1}:")
    print(f"Simultaneous CI: {conf_intervals[i]}")
    print(f"Bonferroni CI: {bonferroni_intervals[i]}")
    print()

'''
    To construct 95% simultaneous confidence intervals for each Mu1i - Mu2i where i=1,2,3,4, you can use the Bonferroni method. The Bonferroni method is a way to adjust the individual confidence intervals to control the familywise error rate.
    This script generates 95% simultaneous confidence intervals for the differences between the means (Mu1i - Mu2i) using the Bonferroni method. Adjustments are made to control the overall familywise error rate.
    '''

#6.22.c
'''
The fact that the data were collected from graduate-student volunteers rather than from a random sample can have several implications:

Selection Bias: The sample may not be representative of the broader population. Graduate students may have different characteristics, lifestyles, and health conditions compared to the general population. This can introduce selection bias, as the sample may not accurately reflect the diversity of the entire population.

Generalizability: Findings from the study may not be easily generalizable to the broader population. The characteristics and behaviors of graduate students might not be representative of other demographic groups. Any conclusions drawn from the data should be cautiously applied beyond the specific group studied.

External Validity: The external validity of the study may be limited. External validity refers to the extent to which the study's findings can be generalized to other populations, settings, or conditions. The results may not be applicable to individuals who are not graduate students.

Confounding Variables: Graduate students may share common attributes or experiences that could confound the analysis. For example, their age, education level, or lifestyle choices may differ from those of the general population, and these factors could influence the observed relationships in the data.

Volunteer Bias: Since the data were collected from volunteers, individuals who chose to participate may have unique characteristics compared to those who chose not to participate. Volunteer bias can affect the validity of the results, as volunteers may be more health-conscious or have specific motivations for participating.

Ethical Considerations: The use of graduate students as volunteers raises ethical considerations, as they may feel compelled to participate due to academic or institutional pressures. This could impact the voluntariness of their participation and potentially influence the study outcomes.

In summary, while the data collected from graduate students provide valuable information within that specific context, caution should be exercised when making broader inferences about the general population. Researchers and readers should be aware of the limitations associated with the non-random sampling method and consider these factors when interpreting and applying the study's findings.
'''