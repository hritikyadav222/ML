# EXPERIMENT 9: CART
#
# Aim:
# To implement CART for classification and regression using Python.
#
# Theory:
# CART builds a decision tree by repeatedly splitting data into smaller groups.
# Classification trees can use Gini impurity, while regression trees minimize
# squared prediction error. Limiting tree depth helps control overfitting.
#
# Algorithm:
# 1. Import the required libraries.
# 2. Load the Iris dataset.
# 3. Split data into training and testing sets.
# 4. Train a classification tree.
# 5. Predict classes and calculate accuracy.
# 6. Visualize the classification tree.
# 7. Train a regression tree using sample data.
# 8. Calculate regression MSE and R2 Score.
#
# Output:
# Displays classification accuracy, regression metrics and a decision tree.
#
# Result:
# CART is demonstrated for classification and regression.

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score
)

# ---------------- CLASSIFICATION ----------------

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train CART classifier
classifier = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

classifier.fit(X_train, y_train)

# Predict classes
y_pred = classifier.predict(X_test)

print("Classification Accuracy:", accuracy_score(y_test, y_pred))
print(
    "Classification Report:\n",
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

# Plot decision tree
plt.figure(figsize=(12, 7))
plot_tree(
    classifier,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.title("CART Classification Tree")
plt.show()

# ---------------- REGRESSION ----------------

# Create a small sample regression dataset
X_reg = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y_reg = [30000, 35000, 40000, 45000, 50000,
         55000, 60000, 65000, 70000, 75000]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Create and train CART regressor
regressor = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

regressor.fit(Xr_train, yr_train)

# Predict values
yr_pred = regressor.predict(Xr_test)

print("Actual Regression Values:", yr_test)
print("Predicted Regression Values:", yr_pred)
print("Regression MSE:", mean_squared_error(yr_test, yr_pred))
print("Regression R2 Score:", r2_score(yr_test, yr_pred))

# Result: CART classification and regression trees are implemented.