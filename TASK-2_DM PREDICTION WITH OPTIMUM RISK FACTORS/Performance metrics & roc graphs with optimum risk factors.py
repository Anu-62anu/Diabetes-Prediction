import os
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
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

# Apply feature selection
selector = SelectKBest(f_classif, k='all')
X_new = selector.fit_transform(X, y)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_new, y)

# Define classifiers with hyperparameter grids
classifiers = {
    'Support Vector Machine': SVC(probability=True),
    'k-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression()
}

param_grids = {
    'Support Vector Machine': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    },
    'k-Nearest Neighbors': {
        'n_neighbors': [3, 5, 10, 20],
        'weights': ['uniform', 'distance']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30]
    },
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'criterion': ['gini', 'entropy']
    },
    'Logistic Regression': {
        'C': [0.1, 1, 10, 100]
    }
}

# Initialize lists to store results and ROC AUC scores
tuned_results = []
roc_auc_scores = {3: [], 5: [], 10: []}  # Store ROC AUC scores for each k value

# Iterate over classifiers for hyperparameter tuning
for clf_name, clf in classifiers.items():
    if clf_name in param_grids and clf_name != 'k-Nearest Neighbors':
        grid_search = GridSearchCV(clf, param_grids[clf_name], cv=5, scoring='accuracy')
        grid_search.fit(X_resampled, y_resampled)
        best_clf = grid_search.best_estimator_

        # Cross-validation with best parameters
        k_values = [3, 5, 10]
        for k in k_values:
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            accuracy_scores = cross_val_score(best_clf, X_resampled, y_resampled, cv=kf, scoring='accuracy')
            precision_scores = cross_val_score(best_clf, X_resampled, y_resampled, cv=kf, scoring='precision')
            recall_scores = cross_val_score(best_clf, X_resampled, y_resampled, cv=kf, scoring='recall')
            f1_scores = cross_val_score(best_clf, X_resampled, y_resampled, cv=kf, scoring='f1')
            auc_scores = cross_val_score(best_clf, X_resampled, y_resampled, cv=kf, scoring='roc_auc')

            # Compute mean scores for each metric
            accuracy_mean = np.mean(accuracy_scores)
            precision_mean = np.mean(precision_scores)
            recall_mean = np.mean(recall_scores)
            f1_mean = np.mean(f1_scores)
            auc_mean = np.mean(auc_scores)

            # Append results to the list
            tuned_results.append({
                'Classifier': clf_name,
                'k': k,
                'Accuracy': accuracy_mean,
                'Precision': precision_mean,
                'Recall': recall_mean,
                'F1 Score': f1_mean,
                'AUC': auc_mean,
                'Best Parameters': grid_search.best_params_
            })

            if k in roc_auc_scores:
                # Get probabilities for ROC curve
                y_pred_proba = cross_val_predict(best_clf, X_resampled, y_resampled, cv=kf, method='predict_proba')[:, 1]
                fpr, tpr, _ = roc_curve(y_resampled, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                roc_auc_scores[k].append((clf_name, fpr, tpr, roc_auc))

    elif clf_name == 'k-Nearest Neighbors':
        for k in [3, 5, 10]:
            clf.set_params(n_neighbors=k)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            accuracy_scores = cross_val_score(clf, X_resampled, y_resampled, cv=kf, scoring='accuracy')
            precision_scores = cross_val_score(clf, X_resampled, y_resampled, cv=kf, scoring='precision')
            recall_scores = cross_val_score(clf, X_resampled, y_resampled, cv=kf, scoring='recall')
            f1_scores = cross_val_score(clf, X_resampled, y_resampled, cv=kf, scoring='f1')
            auc_scores = cross_val_score(clf, X_resampled, y_resampled, cv=kf, scoring='roc_auc')

            # Compute mean scores for each metric
            accuracy_mean = np.mean(accuracy_scores)
            precision_mean = np.mean(precision_scores)
            recall_mean = np.mean(recall_scores)
            f1_mean = np.mean(f1_scores)
            auc_mean = np.mean(auc_scores)

            # Append results to the list
            tuned_results.append({
                'Classifier': clf_name,
                'k': k,
                'Accuracy': accuracy_mean,
                'Precision': precision_mean,
                'Recall': recall_mean,
                'F1 Score': f1_mean,
                'AUC': auc_mean,
                'Best Parameters': f'n_neighbors: {k}'
            })

            if 5 in roc_auc_scores:
                # Get probabilities for ROC curve
                y_pred_proba = cross_val_predict(clf, X_resampled, y_resampled, cv=kf, method='predict_proba')[:, 1]
                fpr, tpr, _ = roc_curve(y_resampled, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                roc_auc_scores[5].append((clf_name, fpr, tpr, roc_auc))

    elif clf_name == 'Naive Bayes':
        k_values = [3, 5, 10]
        for k in k_values:
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            accuracy_scores = cross_val_score(clf, X_resampled, y_resampled, cv=kf, scoring='accuracy')
            precision_scores = cross_val_score(clf, X_resampled, y_resampled, cv=kf, scoring='precision')
            recall_scores = cross_val_score(clf, X_resampled, y_resampled, cv=kf, scoring='recall')
            f1_scores = cross_val_score(clf, X_resampled, y_resampled, cv=kf, scoring='f1')
            auc_scores = cross_val_score(clf, X_resampled, y_resampled, cv=kf, scoring='roc_auc')

            # Compute mean scores for each metric
            accuracy_mean = np.mean(accuracy_scores)
            precision_mean = np.mean(precision_scores)
            recall_mean = np.mean(recall_scores)
            f1_mean = np.mean(f1_scores)
            auc_mean = np.mean(auc_scores)

            # Append results to the list
            tuned_results.append({
                'Classifier': clf_name,
                'k': k,
                'Accuracy': accuracy_mean,
                'Precision': precision_mean,
                'Recall': recall_mean,
                'F1 Score': f1_mean,
                'AUC': auc_mean,
                'Best Parameters': 'N/A'
            })
            
            if k in roc_auc_scores:
                # Get probabilities for ROC curve
                y_pred_proba = cross_val_predict(clf, X_resampled, y_resampled, cv=kf, method='predict_proba')[:, 1]
                fpr, tpr, _ = roc_curve(y_resampled, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                roc_auc_scores[k].append((clf_name, fpr, tpr, roc_auc))

# Create a DataFrame to display the tuned results
tuned_results_df = pd.DataFrame(tuned_results)
print("Tuned Results:")
print(tuned_results_df)

# Plot ROC curve for each classifier and each k value
for k in [3, 5, 10]:
    plt.figure(figsize=(10, 8))
    for clf_name, fpr, tpr, roc_auc in roc_auc_scores[k]:
        plt.plot(fpr, tpr, label=f'{clf_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curve (k={k})', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    plt.show()



