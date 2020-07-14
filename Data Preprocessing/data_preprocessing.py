#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 23:15:42 2020

@author: aaron
"""

# To run code in PyCharm console, highlight and press ⌥⇧E (option shift E)

# Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('FILE')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking Care of missing data
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
X[:, 0] = labelencoder_x.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

# Enconding dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting dataset into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling, after splitting the train and test to prevent information leakage between them
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
