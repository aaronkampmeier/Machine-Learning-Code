# Artificial Neural Networks

# To run code in PyCharm console, highlight and press ⌥⇧E (option shift E)

# Data Preprocessing
import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

# Importing the dataset
dataset = pd.read_csv('Deep Learning/Artificial Neural Networks/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
gender_label_encoder_x = LabelEncoder()
X[:, 1] = labelencoder_x.fit_transform(X[:, 1])
X[:, 2] = gender_label_encoder_x.fit_transform(X[:, 2])
# One hot encode the geography column
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

# Enconding dependent variable
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)

# Splitting dataset into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling, after splitting the train and test to prevent information leakage between them
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


## Now train!!!
ann = tf.keras.models.Sequential()

# Add the input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Add the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Add the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Compile the ANN
# For binary classification, must use this loss function. For non-binary classifications must be something else
ann.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the ANN on training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# Try it with one customer
customer = np.array([[600, "France", "Male", 40, 3, 60000, 2, 1, 1, 50000]])
customer[:, 1] = labelencoder_x.transform(customer[:, 1])
customer[:, 2] = gender_label_encoder_x.transform(customer[:, 2])
customer = onehotencoder.transform(customer).toarray()
customer = sc_X.transform(customer)

prediction = ann.predict(customer, verbose=1)
print(prediction)

# Evaluate it
evaluation = ann.evaluate(X_test, y_test)
print(list(zip(ann.metrics_names, evaluation)))