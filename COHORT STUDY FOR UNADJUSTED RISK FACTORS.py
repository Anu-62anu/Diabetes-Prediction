import os
import pandas as pd

# Read the dataset
base_dir = r'C:\Users\KURRA ANUSHA\OneDrive - iitkgp.ac.in\Desktop\SummerIntern\Main project\pimaindians'

# Define the full path to the CSV file
file_path = os.path.join(base_dir, 'diabetes.csv')

# Load the dataset from a CSV file
df = pd.read_csv(file_path)


df['Exposure'] = df['Insulin'].apply(lambda x: 'Exposure' if x >166 else 'No Exposure')

# Calculate the counts for each group
a = len(df[(df['Exposure'] == 'Exposure') & (df['Outcome'] == 1)])
b = len(df[(df['Exposure'] == 'Exposure') & (df['Outcome'] == 0)])
c = len(df[(df['Exposure'] == 'No Exposure') & (df['Outcome'] == 1)])
d = len(df[(df['Exposure'] == 'No Exposure') & (df['Outcome'] == 0)])

# Create the table
table = pd.DataFrame({
    'Outcome Present (A)': [a, c],
    'Outcome Absent (B)': [b, d],
    'Total (A + B)': [a + b, c + d]
}, index=['Exposure', 'No Exposure'])

# Print the table
print(table)
