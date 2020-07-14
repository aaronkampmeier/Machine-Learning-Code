# K-means Clustering

# To run code in PyCharm console, highlight and press ⌥⇧E (option shift E)

# Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Clustering/Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# # Encoding categorical data
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_x = LabelEncoder()
# X[:, 1] = labelencoder_x.fit_transform(X[:, 1])
# # onehotencoder = OneHotEncoder(categorical_features=[1])
# # X = onehotencoder.fit_transform(X).toarray()


# Train the clusters and choose optimal one with elbow method
from sklearn.cluster import KMeans
wccs_vals = []
for i in range(1, 11):
	trained_model = KMeans(n_clusters=i, init='k-means++', random_state=42)
	trained_model.fit(X)

	# Compute the wcss values
	wccs_vals.append(trained_model.inertia_)

plt.plot(range(1,11), wccs_vals)
plt.title('The elbow method')
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# Elbow showed us that 5 clusters would be optimal
trained_model = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_pred = trained_model.fit_predict(X)

# Visualize

plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(trained_model.cluster_centers_[:,0], trained_model.cluster_centers_[:, 1], s=300, c='yellow',
            label="Centroids")

plt.title('Customer Clusters')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

