import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.tools.tools import add_constant


# Define the full path to the CSV file
base_dir = r'C:\Users\KURRA ANUSHA\OneDrive - iitkgp.ac.in\Desktop\SummerIntern\Main project\pimaindians'

# Define the full path to the CSV file
file_path = os.path.join(base_dir, 'diabetes.csv')

# Load the dataset from a CSV file
data = pd.read_csv(file_path)

# Define the target variable
target = 'Outcome'  # replace 'class' with your target column name

# Define the features
features = ['Glucose','BMI','Age']  # replace with your feature column names

# Initialize a list to store the results
unadjusted_odds_ratios = []

for feature in features:
    # Add a constant to the model
    X = add_constant(data[[feature]])
    y = data[target]
   
    # Fit the logistic regression model
    model = sm.Logit(y, X).fit(disp=0)  # disp=0 suppresses the output
   
    # Get the odds ratio and confidence intervals
    OR = np.exp(model.params[feature])
    CI_lower = np.exp(model.conf_int().loc[feature][0])
    CI_upper = np.exp(model.conf_int().loc[feature][1])
   
    # Append the results to the list
    unadjusted_odds_ratios.append({
        'Feature': feature,
        'OR': OR,
        'Lower CI': CI_lower,
        'Upper CI': CI_upper
    })

# Convert the results to a DataFrame
unadjusted_odds_ratios_df = pd.DataFrame(unadjusted_odds_ratios)

print(unadjusted_odds_ratios_df)
