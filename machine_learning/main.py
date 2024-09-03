# Implementing K-Means Clustering Algorithm with customized kmeans function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/kmtest.csv', header=None)             # (19, 2)
# print(f'shape of the data: {df.shape}')
# print(f'description of the data: {df.describe()}')
print(f'head of the data: \n{df.head()}')
# df = df.drop(['species'], axis=1)
print(f'x column: {df.iloc[0]}')
print(f'y column: {df.iloc[:, 1]}')

# use first column as x-axis and second column as y-axis
plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
plt.show()

k = 4                   # Number of clusters
n = df.shape[0]         # Number of training data
c = df.shape[1]         # Number of features in the data


# Implementation of a custom k-means function
def kmeans(X, k, max_iters=100):
    # # Randomly initialize the centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    print(f'centroids: {centroids}')
    # Choose centroids that are far apart
    # centroids = np.array[(0,0), (100,0), (100,10), (160,0)]

    for i in range(max_iters):
        # Assign each data point to the nearest centroid
        c = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # Find the new centroids
        centroids = np.array([X[c == k].mean(axis=0) for k in range(k)])
    return centroids


# Run the k-means algorithm
X = df.values
centroids = kmeans(X, k)

# Plot the data and the centroids
plt.scatter(X[:, 0], X[:, 1], c='black', s=7)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='red', s=150)
plt.show()

