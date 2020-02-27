# To run code in PyCharm console, highlight and press ⌥⇧E (option shift E)

# Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Regression/Polynomial Regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting linear regression to the datatset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting a polynomial regression model to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(4)
# We'll use the same linear regression class but input instead a polynomial modeled input X. So this function will
# take all of X and outputs a new array with all combinations of the different features and their powers up to the
# specified degree
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualizing the linear regression results
plt.scatter(X, y, c='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualizing the polynomial regression results
plt.scatter(X, y, c='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), c='blue')
plt.title('Truth or Bluff (Polynomial)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# With a little higher resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, c='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), c='blue')
plt.title('Truth or Bluff (Polynomial 4)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Predicting a result with linear regression model
lin_reg.predict([[6.5]])

# Predicting a result with poly model
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
