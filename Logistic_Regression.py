# EXPERIMENT 2: LOGISTIC REGRESSION
#
# Aim:
# To implement Logistic Regression for classification using Python.
#
# Theory:
# Logistic Regression is a supervised learning algorithm used for classification.
# It uses the sigmoid function to estimate class probabilities between 0 and 1.
# Accuracy and the confusion matrix are used to evaluate the model.
#
# Algorithm:
# 1. Import the required libraries.
# 2. Create the dataset.
# 3. Separate features and target labels.
# 4. Split the dataset.
# 5. Train the Logistic Regression model.
# 6. Predict class labels.
# 7. Calculate accuracy and confusion matrix.
#
# Output:
# Displays predicted classes, accuracy, classification report and confusion matrix.
#
# Result:
# Logistic Regression is implemented successfully for classification.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Create sample dataset
data = {
    "StudyHours": [
        1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,
        6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 11
    ],
    "Passed": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

# Select features and target
X = df[["StudyHours"]]
y = df["Passed"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict classes
y_pred = model.predict(X_test)

# Display evaluation
print("Predicted Classes:", y_pred)
print("Actual Classes:", y_test.to_numpy())
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n",
      classification_report(y_test, y_pred, zero_division=0))

# Display confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Result: Logistic Regression classifies observations and evaluates performance.