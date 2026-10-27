# EXPERIMENT 4: MULTIPLE LINEAR REGRESSION
#
# Aim:
# To predict a dependent variable using multiple independent variables.
#
# Theory:
# Multiple Linear Regression predicts an output using two or more input features.
# The model estimates coefficients using the least-squares method.
# R2 Score and MSE evaluate model performance.
#
# Algorithm:
# 1. Import the libraries.
# 2. Create a dataset with multiple input features.
# 3. Select input and output variables.
# 4. Split the data into training and testing sets.
# 5. Train the model.
# 6. Predict test values.
# 7. Calculate R2 Score and MSE.
# 8. Compare actual and predicted values using a graph.
#
# Output:
# Displays actual and predicted salaries, R2 Score, MSE and a graph.
#
# Result:
# Multiple Linear Regression is implemented successfully.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "EducationYears": [12, 12, 14, 14, 16, 16, 16, 18, 18, 18],
    "Salary": [30000, 33000, 39000, 42000, 50000,
               54000, 58000, 65000, 70000, 76000]
}

df = pd.DataFrame(data)

# Select multiple independent variables and target
X = df[["Experience", "EducationYears"]]
y = df["Salary"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Display results
print("Actual Salaries:", y_test.to_numpy())
print("Predicted Salaries:", y_pred)
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Compare actual and predicted values
plt.plot(range(len(y_test)), y_test.to_numpy(),
         marker="o", label="Actual")
plt.plot(range(len(y_pred)), y_pred,
         marker="x", label="Predicted")
plt.xlabel("Test Sample")
plt.ylabel("Salary")
plt.title("Multiple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()

# Result: The model predicts salary using experience and education.