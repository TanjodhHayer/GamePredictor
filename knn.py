import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv('22-375minof40%.csv')

# Selecting features and target variable
feature_columns = [col for col in data.columns if 'red' not in col and col not in ['matchID', 'blueWin']]
X = data[feature_columns]
y = data['blueWin']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initial KNN model training
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
initial_accuracy = accuracy_score(y_test, y_pred)

# Simplified optimization for 'n_neighbors'
simplified_neighbors_range = [1, 5, 10, 15, 20, 25, 30]
simplified_scores = []
for n in simplified_neighbors_range:
    knn_simple = KNeighborsClassifier(n_neighbors=n)
    knn_simple.fit(X_train_scaled, y_train)
    score = knn_simple.score(X_test_scaled, y_test)
    simplified_scores.append(score)

# Finding the optimal 'n_neighbors'
optimal_n_simple = simplified_neighbors_range[simplified_scores.index(max(simplified_scores))]

# Retraining KNN model with optimal 'n_neighbors'
knn_optimized = KNeighborsClassifier(n_neighbors=optimal_n_simple)
knn_optimized.fit(X_train_scaled, y_train)
y_pred_optimized = knn_optimized.predict(X_test_scaled)
optimized_accuracy = accuracy_score(y_test, y_pred_optimized)

print(f"Initial Accuracy: {initial_accuracy}")
print(f"Optimized Accuracy: {optimized_accuracy}")

import matplotlib.pyplot as plt

# Data for Accuracy Comparison Graph
models = ['Initial KNN (n=5)', 'Optimized KNN (n=30)']
accuracies = [initial_accuracy, optimized_accuracy]

# Creating the Accuracy Comparison Graph
plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: Initial vs Optimized KNN Model')
plt.ylim(0.65, 0.75)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.005, f'{acc:.2f}', ha = 'center')
plt.savefig('knn accuracy')

# Data for Hyperparameter Tuning Graph
plt.figure(figsize=(10, 6))
plt.plot(simplified_neighbors_range, simplified_scores, marker='o')
plt.xlabel('Number of Neighbors (n)')
plt.ylabel('Accuracy')
plt.title('KNN Model Accuracy for Different Values of n_neighbors')
plt.xticks(simplified_neighbors_range)
plt.grid(True)
plt.savefig('knn across diff values')
