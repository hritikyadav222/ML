# EXPERIMENT 1: SIMPLE LINEAR REGRESSION
#
# Aim:
# To implement Simple Linear Regression using Python and predict salary
# based on years of experience.
#
# Theory:
# Linear Regression predicts a continuous value using the equation Y = mX + c.
# The model finds the best-fit line by minimizing squared prediction errors.
# R2 Score and Mean Squared Error (MSE) are used for evaluation.
#
# Algorithm:
# 1. Import the required libraries.
# 2. Create a sample dataset.
# 3. Select independent and dependent variables.
# 4. Split the data into training and testing sets.
# 5. Train the Linear Regression model.
# 6. Predict salary values.
# 7. Calculate R2 Score and MSE.
# 8. Plot the regression line.
#
# Output:
# Displays predicted salaries, R2 Score, MSE, and a regression graph.
#
# Result:
# Simple Linear Regression is implemented successfully.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create sample dataset
data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Salary": [30000, 35000, 40000, 45000, 50000,
              55000, 60000, 65000, 70000, 75000]
}

df = pd.DataFrame(data)

# Select input and output variables
X = df[["Experience"]]
y = df["Salary"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test values
y_pred = model.predict(X_test)

# Display results
print("Predicted Salaries:", y_pred)
print("Actual Salaries:", y_test.to_numpy())
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Plot actual data and regression line
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")
plt.grid(True)
plt.legend()
plt.show()

# Result: The model predicts salary from experience and evaluates its performance.