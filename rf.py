import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import os

output_folder = 'RF_Plots'
os.makedirs(output_folder, exist_ok=True)

# Load the dataset
data = pd.read_csv('22-375minof40%.csv')

# Selecting features and target variable
feature_columns = ['blueRiftHeraldKill', 'blueTowerKill', 'blueTotalGold', 'blueAvgPlayerLevel', 'fullTimeMin', 'GoldDiff', 'ChampionKillsDiff', 'DragonKillsDiff', 'blueMinionsKilledTotal', 'MinionsKilledDiff']
X = data[feature_columns]
y = data['blueWin']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=70)

# Random Forest model
rf_model = RandomForestClassifier(random_state=70)

# Hyperparameter tuning using Grid Search
param_grid = {'n_estimators': np.arange(1, 100, 2)}
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Extract best model and parameters
best_rf_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Predict using the best model
y_pred = best_rf_model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# ROC and Precision-Recall Curves
y_scores = best_rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_scores)
precision, recall, _ = precision_recall_curve(y_test, y_scores)

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)
print(f"Best Parameters from Grid Search: {best_params}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(os.path.join(output_folder, 'ROC Curve.png'))

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig(os.path.join(output_folder, 'Precision-Recall Curve.png'))

# Plot Accuracy vs. Number of Estimators
estimator_range = param_grid['n_estimators']
accuracy_scores = grid_search.cv_results_['mean_test_score']
plt.figure(figsize=(8, 6))
plt.plot(estimator_range, accuracy_scores, marker='o')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Random Forest Accuracy for Different Values of n_estimators (Grid Search)')
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'RF_Accuracy_vs_n_estimators.png'))
