# EXPERIMENT 5: SUPPORT VECTOR MACHINE
#
# Aim:
# To implement SVM for classification using Python.
#
# Theory:
# SVM finds an optimal hyperplane that separates classes with the maximum margin.
# The closest data points are called support vectors. Kernel functions allow SVM
# to handle linear and nonlinear classification problems.
#
# Algorithm:
# 1. Import libraries and load the Iris dataset.
# 2. Split the dataset into training and testing sets.
# 3. Scale the features.
# 4. Create an SVM classifier.
# 5. Train the model.
# 6. Predict test labels.
# 7. Calculate accuracy and display a confusion matrix.
#
# Output:
# Displays accuracy, classification report and confusion matrix.
#
# Result:
# SVM is implemented successfully for classification.

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create pipeline: scale features, then apply SVM
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear")
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n",
      classification_report(y_test, y_pred, target_names=iris.target_names))

# Display confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=iris.target_names
)
plt.title("SVM Confusion Matrix")
plt.show()

# Result: SVM classifies Iris flowers and evaluates its performance.