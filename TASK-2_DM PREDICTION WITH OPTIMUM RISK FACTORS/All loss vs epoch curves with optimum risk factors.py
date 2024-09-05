import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

file_path = r'C:\Users\KURRA ANUSHA\OneDrive - iitkgp.ac.in\Desktop\SummerIntern\Main project\pimaindians\Task-2\Important\testingfile.xlsx'
# Load the data
df = pd.read_excel(file_path)

# Features and target variable
X = df.drop('Outcome', axis=1)  # Assuming the target variable is named 'Outcome'
y = df['Outcome']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Define classifiers
classifiers = {
    'Support Vector Machine': SVC(probability=True),
    'k-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression()
}

# Function to compute losses during training
def compute_losses(clf, X_train, y_train, X_valid, y_valid):
    clf.fit(X_train, y_train)
    if hasattr(clf, 'predict_proba'):
        y_train_pred_proba = clf.predict_proba(X_train)[:, 1]
        y_valid_pred_proba = clf.predict_proba(X_valid)[:, 1]
    else:
        y_train_pred_proba = clf.predict(X_train)
        y_valid_pred_proba = clf.predict(X_valid)
    
    train_loss = log_loss(y_train, y_train_pred_proba)
    valid_loss = log_loss(y_valid, y_valid_pred_proba)
    return train_loss, valid_loss

# Train and evaluate each classifier
for clf_name, clf in classifiers.items():
    print(f"Training and evaluating {clf_name}...")
    k_values = [3, 5, 10]
    for k in k_values:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        train_losses = []
        valid_losses = []
        for train_index, valid_index in kf.split(X_resampled):
            X_train_k, X_valid_k = X_resampled[train_index], X_resampled[valid_index]
            y_train_k, y_valid_k = y_resampled[train_index], y_resampled[valid_index]
            
            train_loss, valid_loss = compute_losses(clf, X_train_k, y_train_k, X_valid_k, y_valid_k)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
        
        # Plot loss vs epoch graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, k+1), train_losses, marker='o', linestyle='-', color='b', label='Train Loss')
        plt.plot(range(1, k+1), valid_losses, marker='o', linestyle='-', color='r', label='Validation Loss')
        plt.title(f'{clf_name} Loss vs Epoch (k={k})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(range(1, k+1))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
