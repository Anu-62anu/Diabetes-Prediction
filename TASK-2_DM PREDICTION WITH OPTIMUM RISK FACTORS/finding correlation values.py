import os
import pandas as pd
from scipy.stats import pointbiserialr

# Example file path (adjust as per your actual file location)
base_dir = r'C:\Users\KURRA ANUSHA\OneDrive - iitkgp.ac.in\Desktop\SummerIntern\Main project\pimaindians'
file_path = os.path.join(base_dir, 'diabetes.csv')

try:
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Ensure Outcome (target variable) matches your actual column name
    target_variable = 'Outcome'
    
    # Define continuous features for point-biserial correlation
    continuous_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    results = []
    
    # Compute point-biserial correlation for each continuous feature
    for feature in continuous_features:
        # Calculate point-biserial correlation
        correlation, _ = pointbiserialr(df[feature], df[target_variable])
        results.append((feature, target_variable, 'Point-Biserial Correlation', correlation))
    
    # Sort results by correlation value in descending order
    sorted_results = sorted(results, key=lambda x: abs(x[3]), reverse=True)
    
    # Print results
    for result in sorted_results:
        print(f"{result[2]} between {result[0]} and {result[1]}: {result[3]}")
    
except Exception as e:
    print(f"Error: {e}")


