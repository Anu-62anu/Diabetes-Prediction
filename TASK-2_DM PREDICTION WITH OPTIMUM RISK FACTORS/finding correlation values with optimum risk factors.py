import os
import pandas as pd
import numpy as np

# load the dataset
base_dir = r'C:\Users\KURRA ANUSHA\OneDrive - iitkgp.ac.in\Desktop\SummerIntern\Main project\pimaindians'

# Define the full path to the CSV file
file_path = os.path.join(base_dir, 'diabetes.csv')

df = pd.read_csv(file_path) 
df.head(3)

label = 'Outcome'
features = df.columns.tolist()
features.remove(label)

from scipy.stats import pointbiserialr
from math import sqrt

def getMerit(subset, label):
    k = len(subset)

    # average feature-class correlation
    rcf_all = []
    for feature in subset:
        coeff = pointbiserialr( df[label], df[feature] )
        rcf_all.append( abs( coeff.correlation ) )
    rcf = np.mean( rcf_all )

    # average feature-feature correlation
    corr = df[subset].corr()
    corr.values[np.tril_indices_from(corr.values)] = np.nan
    corr = abs(corr)
    rff = corr.unstack().mean()

    return (k * rcf) / sqrt(k + k * (k-1) * rff)

subset = ['Glucose','BMI','Pregnancies','Age','DiabetesPedigreeFunction','Insulin','SkinThickness','BloodPressure']

l=len(subset)
riskset=[]
for i in range(2,l+1):
    for j in range(0,i):
        riskset.append(subset[j])
    print(riskset," ",getMerit(riskset,label))
    riskset=[]
    

