import pandas as pd
import numpy as np
import os

# Load the dataset
base_dir = r'C:\Users\KURRA ANUSHA\OneDrive - iitkgp.ac.in\Desktop\SummerIntern\Main project\pimaindians'
file_path = os.path.join(base_dir, 'diabetes.csv')
df = pd.read_csv(file_path)

# Define adjusted risk categories for Body Mass Index(BMI)
low_tsf = df[df['BMI'] < 18.5]
normal_tsf = df[(df['BMI'] >= 18.5) & (df['BMI'] < 25)]
high_tsf = df[df['BMI'] >= 25]

def calculate_rr_ci(exposure_group, reference_group):
    a = exposure_group[exposure_group['Outcome'] == 1].shape[0]  # Cases in exposure group
    b = exposure_group[exposure_group['Outcome'] == 0].shape[0]  # Non-cases in exposure group
    c = reference_group[reference_group['Outcome'] == 1].shape[0]  # Cases in reference group
    d = reference_group[reference_group['Outcome'] == 0].shape[0]  # Non-cases in reference group

    print(f"a = {a}, b = {b}, c = {c}, d = {d}")  # Print a, b, c, d for clarity
    
    # Relative Risk (RR)
    rr = (a / (a + b)) / (c / (c + d))
    
    # Log of RR for CI calculation
    log_rr = np.log(rr)/np.log(2.71828)

    
    
    # Standard Error (SE) of log(RR)
    se = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    
    # Confidence Interval (CI)
    ci_lower = np.exp(log_rr - 1.96 * se)
    ci_upper = np.exp(log_rr + 1.96 * se)
    
    return rr, (ci_lower, ci_upper)

# Calculate RR and CI for each category compared to 'Low' Body Mass Index(BMI)
rr_low_tsf, ci_low_tsf = calculate_rr_ci(normal_tsf, low_tsf)  # Normal vs Low
rr_high_tsf, ci_high_tsf = calculate_rr_ci(high_tsf, low_tsf)  # High vs Low
rr_high_tsf_normal, ci_high_tsf_normal = calculate_rr_ci(high_tsf, normal_tsf)  # High vs Normal

# Print the results
print("\nRelative Risk (RR) and Confidence Interval (CI) for Body Mass Index(BMI):")
print(f"Normal vs Low (18.5 vs <25): RR = {rr_low_tsf:.2f}, CI = {ci_low_tsf}")
print(f"High vs Low (>30 vs <15): RR = {rr_high_tsf:.2f}, CI = {ci_high_tsf}")
print(f"High vs Normal (>30 vs 15-30): RR = {rr_high_tsf_normal:.2f}, CI = {ci_high_tsf_normal}")
