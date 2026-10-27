# EXPERIMENT 3: ENSEMBLE LEARNING
#
# Aim:
# To implement and compare Bagging and Boosting using Python.
#
# Theory:
# Ensemble Learning combines multiple models to improve predictions.
# Bagging trains models independently on different samples of the dataset.
# Boosting trains models sequentially to improve previous errors.
#
# Algorithm:
# 1. Import the libraries and load the Iris dataset.
# 2. Split the dataset into training and testing sets.
# 3. Train a Bagging classifier.
# 4. Predict test labels and calculate accuracy.
# 5. Train an AdaBoost classifier.
# 6. Predict test labels and calculate accuracy.
# 7. Compare both accuracies using a graph.
#
# Output:
# Displays Bagging accuracy, Boosting accuracy and a comparison graph.
#
# Result:
# Bagging and Boosting are implemented and compared successfully.

import inspect
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create Bagging classifier, supporting newer and older sklearn versions
bag_params = {"n_estimators": 50, "random_state": 42}
base_tree = DecisionTreeClassifier(random_state=42)

if "estimator" in inspect.signature(BaggingClassifier).parameters:
    bag_params["estimator"] = base_tree
else:
    bag_params["base_estimator"] = base_tree

bagging = BaggingClassifier(**bag_params)
bagging.fit(X_train, y_train)
bag_pred = bagging.predict(X_test)
bag_acc = accuracy_score(y_test, bag_pred)

# Create AdaBoost classifier
boost_tree = DecisionTreeClassifier(max_depth=1, random_state=42)
boost_params = {"n_estimators": 50, "random_state": 42}

if "estimator" in inspect.signature(AdaBoostClassifier).parameters:
    boost_params["estimator"] = boost_tree
else:
    boost_params["base_estimator"] = boost_tree

boosting = AdaBoostClassifier(**boost_params)
boosting.fit(X_train, y_train)
boost_pred = boosting.predict(X_test)
boost_acc = accuracy_score(y_test, boost_pred)

# Display accuracy
print("Bagging Accuracy:", bag_acc)
print("Boosting Accuracy:", boost_acc)

# Compare models
plt.bar(["Bagging", "AdaBoost"], [bag_acc, boost_acc])
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Bagging vs Boosting")
plt.show()

# Result: Both ensemble techniques are trained and their accuracy is compared.