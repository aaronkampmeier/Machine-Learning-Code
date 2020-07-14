# To run code in PyCharm console, highlight and press ⌥⇧E (option shift E)

# Data Pre-processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Regression/Model Selection/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting dataset into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Run the test suite for each model type:
# 2D array of the model and its pre-processing and post-processing code (pre is applied to X_test, post is applied to
# the predicted values from X_test)
regressors = []
# Multiple Linear Regression
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X_train, y_train)
regressors.append(["Multiple Linear", lin_regressor, None, None])

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(4)
# We'll use the same linear regression class but input instead a polynomial modeled input X. So this function will
# take all of X and outputs a new array with all combinations of the different features and their powers up to the
# specified degree
X_poly = poly_features.fit_transform(X_train)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y_train)
regressors.append(["Polynomial", poly_regressor, lambda xt: poly_features.fit_transform(xt), None])

# Support Vector Regression
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X_train)
y_scaled = sc_y.fit_transform(y_train.reshape(-1, 1))
y_scaled = y_scaled[:, 0]
# Create the SVR regressor
from sklearn.svm import SVR
svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(X_scaled, y_scaled)
regressors.append(["SVR", svr_regressor, lambda xt: sc_X.fit_transform(xt), lambda yp: sc_y.inverse_transform(yp)])

# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
decision_regressor = DecisionTreeRegressor(random_state=0)
decision_regressor.fit(X_train, y_train)
regressors.append(["Decision Tree", decision_regressor, None, None])

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
random_forest_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
random_forest_regressor.fit(X_train, y_train)
regressors.append(["Random Forest", random_forest_regressor, None, None])


# Now test all of them
# Get a y_pred for each model
regressor_results = []
for regressor_suite in regressors:
	regressor_name = regressor_suite[0]
	regressor = regressor_suite[1]
	test_transform = regressor_suite[2]
	pred_transform = regressor_suite[3]

	print("\n\nTesting regressor: " + regressor_name)

	test_data = X_test
	if (test_transform):
		test_data = test_transform(X_test)

	y_pred = regressor.predict(test_data)

	if(pred_transform):
		y_pred = pred_transform(y_pred)

	np.set_printoptions(precision=2)
	print(np.concatenate((y_test.reshape(-1, 1), y_pred.reshape(-1, 1)), 1))

	# Analyze with R^2
	from sklearn.metrics import r2_score
	score = r2_score(y_test, y_pred)

	print("R^2: " + str(score))
	regressor_results.append([regressor_name, score])


# Print out the best
print(regressor_results)

