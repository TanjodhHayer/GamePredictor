import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns

output_folder = 'LR_Plots'
os.makedirs(output_folder, exist_ok=True)
# Load the dataset
data = pd.read_csv('22-375minof40%.csv')

# Selecting features and target variable
feature_columns = ['blueRiftHeraldKill', 'blueTowerKill', 'blueTotalGold', 'blueAvgPlayerLevel', 'fullTimeMin', 'GoldDiff', 'ChampionKillsDiff', 'DragonKillsDiff', 'blueMinionsKilledTotal', 'MinionsKilledDiff']
X = data[feature_columns]
y = data['blueWin']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=70)

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns)  # Use all columns as numeric
    ])

# Create Logistic Regression pipeline
logistic_regression_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=2000, random_state=70))
])

# Fit the preprocessor on the entire dataset
preprocessor.fit(X)

# Fit the model
logistic_regression_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logistic_regression_model.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_folder, 'Confusion Matrix.png'))

# Plot ROC curve
y_probs = logistic_regression_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)

# Plot Accuracy vs. Threshold
thresholds = np.linspace(0, 1, 100)
accuracies = [accuracy_score(y_test, y_probs >= thr) for thr in thresholds]

plt.figure(figsize=(18, 6))

# Plot ROC Curve
plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# Plot Precision-Recall Curve
plt.subplot(1, 3, 2)
plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")

# Plot Accuracy vs. Threshold
plt.subplot(1, 3, 3)
plt.plot(thresholds, accuracies, color='purple', lw=2, label='Accuracy')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Threshold')
plt.legend(loc="lower right")

# Save the combined plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(output_folder, 'Combined_Plots.png'))