import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def compute_losses(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    if hasattr(clf, 'predict_proba'):
        y_train_pred_proba = clf.predict_proba(X_train)
        y_test_pred_proba = clf.predict_proba(X_test)
    else:
        y_train_pred_proba = clf.predict(X_train)
        y_test_pred_proba = clf.predict(X_test)
    train_loss = log_loss(y_train, y_train_pred_proba)
    test_loss = log_loss(y_test, y_test_pred_proba)
    return train_loss, test_loss

def train_and_evaluate_model(classifier, X_train, y_train, X_test, y_test, k_values):
    loss_results = []
    train_losses_history = []
    test_losses_history = []
    
    for k in k_values:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        train_losses = []
        test_losses = []
        
        for train_index, test_index in kf.split(X_train):
            X_train_k, X_valid_k = X_train[train_index], X_train[test_index]
            y_train_k, y_valid_k = y_train[train_index], y_train[test_index]
            
            train_loss, test_loss = compute_losses(classifier, X_train_k, y_train_k, X_valid_k, y_valid_k)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
        
        loss_results.append({
            'k': k,
            'Train Loss': np.mean(train_losses),
            'Test Loss': np.mean(test_losses)
        })
        
        train_losses_history.append(train_losses)
        test_losses_history.append(test_losses)
    
    return loss_results, train_losses_history, test_losses_history

# Define the path to the CSV file
file_path = r'C:\Users\KURRA ANUSHA\OneDrive - iitkgp.ac.in\Desktop\SummerIntern\Main project\pimaindians\diabetes.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Correct column names for replacement
columns_with_zero_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace zero values with NaN in specific columns
df[columns_with_zero_values] = df[columns_with_zero_values].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

# Features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Define k values
k_values = [3, 5, 10]

# Train and evaluate each classifier
for clf_name, clf in classifiers.items():
    print(f"Training and evaluating {clf_name}...")
    loss_results, train_losses_history, test_losses_history = train_and_evaluate_model(clf, X_resampled, y_resampled, X_resampled, y_resampled, k_values)
    print("Loss Results:")
    print(pd.DataFrame(loss_results))
    
    # Plot loss vs epoch graph
    plt.figure(figsize=(10, 6))
    for i, k in enumerate(k_values):
        plt.plot(range(1, k+1), train_losses_history[i], marker='o', linestyle='-', label=f'Train Loss (K={k})')
        plt.plot(range(1, k+1), test_losses_history[i], marker='o', linestyle='-', label=f'Test Loss (K={k})')
    plt.title(f'{clf_name} Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(1, max(k_values) + 1))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

