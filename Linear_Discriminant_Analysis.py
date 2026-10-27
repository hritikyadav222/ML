# EXPERIMENT 10: LINEAR DISCRIMINANT ANALYSIS
#
# Aim:
# To implement LDA for dimensionality reduction and classification.
#
# Theory:
# LDA is a supervised dimensionality reduction technique. It uses class labels
# to maximize separation between classes relative to variation within each class.
# Unlike PCA, LDA uses target labels during dimensionality reduction.
#
# Algorithm:
# 1. Import libraries and load the Iris dataset.
# 2. Split the data into training and testing sets.
# 3. Scale the features.
# 4. Apply LDA to reduce the feature dimensions.
# 5. Train a classifier using the reduced features.
# 6. Predict test labels.
# 7. Calculate accuracy and display a confusion matrix.
# 8. Plot the reduced data.
#
# Output:
# Displays the original and reduced data shapes, accuracy, classification report,
# confusion matrix and a two-dimensional LDA plot.
#
# Result:
# LDA is used for dimensionality reduction and classification.

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
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

# Scale features using training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

print("Original Feature Shape:", X_train.shape)
print("Reduced Training Shape:", X_train_lda.shape)
print("Reduced Testing Shape:", X_test_lda.shape)

# Train classifier on reduced features
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train_lda, y_train)

# Predict
y_pred = classifier.predict(X_test_lda)

# Evaluate model
print("Accuracy after LDA:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(
    "Classification Report:\n",
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

# Display confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=iris.target_names
)
plt.title("LDA Classification Confusion Matrix")
plt.show()

# Plot reduced training data
plt.scatter(
    X_train_lda[:, 0],
    X_train_lda[:, 1],
    c=y_train,
    cmap="viridis"
)
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.title("LDA Reduced Data")
plt.show()

# Result: LDA reduces feature dimensions and evaluates classification performance.