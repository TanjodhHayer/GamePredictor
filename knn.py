import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import os

output_folder = 'KNN_Plots'
os.makedirs(output_folder, exist_ok=True)
# Load the dataset
data = pd.read_csv('22-375minof40%.csv')

# Selecting features and target variable
feature_columns = ['blueRiftHeraldKill', 'blueTowerKill', 'blueTotalGold', 'blueAvgPlayerLevel', 'fullTimeMin', 'GoldDiff', 'ChampionKillsDiff', 'DragonKillsDiff', 'blueMinionsKilledTotal', 'MinionsKilledDiff']
X = data[feature_columns]
y = data['blueWin']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=70)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initial KNN model training
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
initial_accuracy = accuracy_score(y_test, y_pred)

# Cross-validation
cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()}")

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

# Grid Search for Hyperparameter Tuning
param_grid = {'n_neighbors': np.arange(1, 100, 2)}

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Extract results from grid search
grid_means = grid_search.cv_results_['mean_test_score']
grid_stds = grid_search.cv_results_['std_test_score']
grid_params = grid_search.cv_results_['params']

# Get the best parameters from the grid search
best_params = grid_search.best_params_

# Use the best parameters to train the optimized KNN model
best_knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'])
best_knn.fit(X_train_scaled, y_train)
y_pred_optimized = best_knn.predict(X_test_scaled)
optimized_accuracy = accuracy_score(y_test, y_pred_optimized)

print(f"Initial Accuracy: {initial_accuracy}")
print(f"Optimized Accuracy (Grid Search): {optimized_accuracy}")
print(f"Best Parameters from Grid Search: {best_params}")

# Data for Accuracy Comparison Graph
models = ['Initial KNN (''n_neighbours'':5)', f'Optimized KNN {best_params}']

accuracies = [initial_accuracy, optimized_accuracy]

# Creating the Accuracy Comparison Graph
plt.figure(figsize=(16, 16))
plt.bar(models, accuracies, color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: Initial vs Optimized KNN Model')
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.005, f'{acc:.2f}', ha='center')
plt.savefig(os.path.join(output_folder, 'accuracy_comparison.png'))

# Simplified optimization for 'n_neighbors'
simplified_neighbors_range = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
simplified_scores = []

for n in simplified_neighbors_range:
    knn_simple = KNeighborsClassifier(n_neighbors=n)
    knn_simple.fit(X_train_scaled, y_train)
    score = knn_simple.score(X_test_scaled, y_test)
    simplified_scores.append(score)

# Data for Hyperparameter Tuning Graph
plt.figure(figsize=(10, 6))
plt.plot(simplified_neighbors_range, simplified_scores, marker='o', label='Simplified Optimization')
#plt.errorbar(param_grid['n_neighbors'], grid_means, yerr=grid_stds, marker='x', label='Grid Search')
plt.plot([params['n_neighbors'] for params in grid_params], grid_means, marker='x', linestyle='-', label='Grid Search')
plt.xlabel('Number of Neighbors (n)')
plt.ylabel('Accuracy')
plt.title('KNN Model Accuracy for Different Values of n_neighbors')
plt.xticks(simplified_neighbors_range)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'knn_across_diff_values.png'))


# Initial KNN model
y_scores_initial = knn.predict_proba(X_test_scaled)[:, 1]
fpr_initial, tpr_initial, _ = roc_curve(y_test, y_scores_initial)
precision_initial, recall_initial, _ = precision_recall_curve(y_test, y_scores_initial)

# Optimized KNN model (Grid Search)
y_scores_optimized = best_knn.predict_proba(X_test_scaled)[:, 1]
fpr_optimized, tpr_optimized, _ = roc_curve(y_test, y_scores_optimized)
precision_optimized, recall_optimized, _ = precision_recall_curve(y_test, y_scores_optimized)

# Plotting ROC curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr_initial, tpr_initial, color='blue', label='Initial KNN')
plt.plot(fpr_optimized, tpr_optimized, color='green', label='Optimized KNN (Grid Search)')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Plotting precision-recall curves
plt.subplot(1, 2, 2)
plt.plot(recall_initial, precision_initial, color='blue', label='Initial KNN')
plt.plot(recall_optimized, precision_optimized, color='green', label='Optimized KNN (Grid Search)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'knn_roc_curves&precision-recall_curves.png'))


# Save the best model
import joblib
joblib.dump(best_knn, 'best_knn_model_grid_search.pkl')