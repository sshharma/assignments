# Implementing K-Means Clustering Algorithm with customized kmeans function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




# Implementation of a custom k-means function
def plot_graph(centroids, c_, X):
    # Plot markers and colors
    mark = ['D', 'o', '1', '>', '*', 'p', 's']  # List of markers
    color = ['lime', 'blue', 'green', 'black', 'brown', 'pink']  # List of colors

    # Plot the clusters
    for i in range(len(centroids)):
        cluster_points = X[c_ == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], marker=mark[i], s=50, c=color[i])

    # Plot the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='+', c='red', s=400)
    plt.show()


def kmeans(points, k, max_iters=100):
    # initialize the centroids randomly
    # centroids = points[np.random.choice(range(n), k, replace=False)]

    # choose centroids manually, that are far apart
    centroids = [[0, 0], [100, 0], [100, 10], [160, 0]]
    last_centroids = centroids
    # Run the k-means algorithm for a maximum of max_iters iterations
    for i in range(max_iters):
        cluster_assignments = []                    # List to store the cluster assignments
        for x_i in points:                               # Iterate over each data point
            distances = []                          # List to store the distances to each centroid
            last_centroid_position = [0, 0]
            for y_k in centroids:                   # Iterate over each centroid
                # Calculate the Euclidean distance
                distance = np.linalg.norm(x_i - y_k)
                distances.append(distance)
            closest_centroid = np.argmin(distances) # Find the closest centroid to assign the data point to
            cluster_assignments.append(closest_centroid)

        c_ = np.array(cluster_assignments)

        # Find the new centroids
        centroids = np.array([points[c_ == k].mean(axis=0) for k in range(k)])
        # compare the new centroids with the last centroids upto 2 decimal places
        if np.all(np.round(centroids, 2) == np.round(last_centroids, 2)):
            print(f'Converged after {i} iterations.')
            break
        last_centroids = centroids
        plot_graph(centroids, c_, points)

    return centroids,c_


def main():
    df = pd.read_csv('data/kmtest.csv', header=None)  # (19, 2)
    # print(f'shape of the data: {df.shape}')

    # use first column as x-axis and second column as y-axis
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
    plt.show()

    k = 4  # Number of clusters
    n = df.shape[0]  # Number of training data
    c = df.shape[1]  # Number of features in the data

    # Run the k-means algorithm
    all_points = df.values
    centroids, c = kmeans(all_points, k)
    print(f'centroids: {centroids}')
    plot_graph(centroids, c, all_points)



if __name__ == '__main__':
    main()

# Output:
