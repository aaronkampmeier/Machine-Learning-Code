import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from DeepLearning import CustomANN
import pandas as pd
#
# X = np.array([
# 	[0, 1, 0.5],
# 	[0, 1, 0.5],
# 	[1, 0, -0.5],
# 	[1, 0, -0.5]
# ])

# Y = np.array([
# 	[0.7, 0],
# 	[0.7, 0],
# 	[0, 0.7],
# 	[0, 0.7]
# ])
#
# Y = np.array([
# 	[3, -1],
# 	[3, -1],
# 	[1.1, 2.4],
# 	[1.1, 2.4]
# ])


## California Housing Dataset

california_housing_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

california_housing_data.describe()
california_housing_data.head()

X = california_housing_data.iloc[:, 2:-1]
Y = california_housing_data.iloc[:, -1:]

## Churn Modeling
# dataset = pd.read_csv('DeepLearning/Artificial Neural Networks/Churn_Modelling.csv')
# X = dataset.iloc[:, 3:-1].values
# Y = dataset.iloc[:, -1].values
# Encoding categorical data
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_x = LabelEncoder()
# gender_label_encoder_x = LabelEncoder()
# X[:, 1] = labelencoder_x.fit_transform(X[:, 1])
# X[:, 2] = gender_label_encoder_x.fit_transform(X[:, 2])
# # One hot encode the geography column
# onehotencoder = OneHotEncoder(categorical_features=[1])
# X = onehotencoder.fit_transform(X).toarray()


try:
	Y[:, 1]
except:
	Y = np.reshape(Y, (-1, 1))

# Scale my data
scaler_X = StandardScaler()
scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

neural_net = CustomANN.MyNeuralNetwork()
# neural_net.fit(X_scaled, Y_scaled, learning_rate=1, num_of_hidden_layers=2, hidden_neurons_in_layer=6,
			   # epochs=10000, batch_size=4)
neural_net.fit(X_scaled, Y_scaled, learning_rate=0.05, num_of_hidden_layers=3, hidden_neurons_in_layer=6,
			   epochs=100000, batch_size=100)
# neural_net.fit(X_scaled, Y_scaled, learning_rate=1, num_of_hidden_layers=2, hidden_neurons_in_layer=6,
# 			   epochs=10000, batch_size=32)

y_pred = neural_net.forward_propagate(X_scaled)[-1]
y_pred = scaler_Y.inverse_transform(y_pred)

print(y_pred)
