# Implementing K-Means Clustering Algorithm with customized kmeans function
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pygments.lexer import default
import logging

from scipy.cluster.hierarchy import centroid

logging.basicConfig(level=logging.ERROR)

# Implementation of a custom k-means function
def plot_graph(centroids, c_, cordi):
    # Plot markers and colors
    mark = ['D', '1', 'o', '>', '*', 'p', 's']  # List of markers
    color = ['lime', 'blue', 'green', 'black', 'brown', 'pink']  # List of colors

    # Plot the clusters
    for i in range(len(centroids)):
        cluster_points = cordi[c_ == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], marker=mark[i], s=50, c=color[i])

    # Plot the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='+', c='red', s=400)
    plt.show()


def find_centroids(points_, k_, c_):
    """
    Find the centroids for the given number of clusters, data points and cluster assignments
    Also handles the case where a cluster has 0 length
    :param points_: data points
    :param k_: number of clusters
    :param c_:  assignments of the data points to the clusters
    :return centroids: list of centroids
    """
    centroids = []                                              # List to store the centroids
    for cluster_index in range(k_):
        cluster_points = points_[c_ == cluster_index]           # Get the points in the current cluster
        if len(cluster_points) != 0:
            cluster_mean = cluster_points.mean(axis=0)          # Calculate the mean of the cluster points
            centroids.append(cluster_mean)                      # Append the mean to the list of centroids
    return np.array(centroids)


def initialize_centroids(k):
    """
    Get the centroids for the given number of clusters
    :param k: number of clusters
    :return centroids: list of centroids
    """

    if k==2:
        return np.array([[0, 5], [180, 5]])
    elif k==4:
        return np.array([[0, 0], [100, 0], [100, 10], [160, 0]])
    elif k==6:
        return np.array([[20, 7], [50, 7], [90, 7], [100, 7], [150, 7], [160, 7]])
    else:

        logging.error('Invalid number of clusters. Please choose 2, 4 or 6.')
        return None


def custom_kmeans(points, k, max_iters):
    # initialize the centroids randomly
    # raw_centroids = points[np.random.choice(points.shape[0], k, replace=False)]

    # chosen centroids manually, that are far apart from each other
    centroids = initialize_centroids(k)

    last_centroids = centroids
    # Run the k-means algorithm for a maximum of max_iters iterations
    for i in range(max_iters):
        cluster_assignments = []                            # List to store the cluster assignments
        for x_i in points:                                  # Iterate over each data point
            distances = []                                  # List to store the distances to each centroid
            for y_k in centroids:                           # Iterate over each centroid
                distance = np.linalg.norm(x_i - y_k)        # Calculate the Euclidean distance
                distances.append(distance)                  # Append the distance to the list
            closest_centroid = np.argmin(distances)         # Find the closest centroid to assign the data point to
            cluster_assignments.append(closest_centroid)    # Assign the data point to the closest centroid
        c_ = np.array(cluster_assignments)                  # Convert the list to a numpy array
        plot_graph(centroids, c_, points)                   # Plot the graph

        # Find the new centroids
        # centroids = np.array([points[c_ == k].mean(axis=0) for k in range(k)])
        centroids = find_centroids(points, k, c_)
        print(f'centroids for iteration {i}: {centroids}')

        try:
            if np.all(np.round(centroids, 2) == np.round(last_centroids, 2)):
                print(f'Converged after {i} iterations.')
                break
        except ValueError:
            logging.info(f'Number of centroids have changed from: {len(last_centroids)} to: {len(centroids) }'
                         f'skipping the comparison in this iteration.')
        last_centroids = centroids

    return centroids,c_


def z_score_normalize(data):
    """
    Normalize the data using the z-score normalization
    :param data: data to be normalized
    :return: normalized data
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


def main():
    # read the data from command line
    parser = argparse.ArgumentParser(description='K-Means Clustering Algorithm')
    parser.add_argument('--train_dir', type=str, help='data file', default='data/kmtest.csv')
    parser.add_argument('--k', type=int, help='number of clusters', default=2)
    parser.add_argument('--max_iters', type=int, help='maximum number of iterations', default=5)
    args = parser.parse_args()


    df = pd.read_csv(args.train_dir, header=None)  # (19, 2)
    # print(f'shape of the data: {df.shape}')

    # use first column as x-axis and second column as y-axis
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
    plt.show()

    all_points = df.values                              # Convert the dataframe to a numpy array
    print(f'Original data: {all_points}')

    all_points = z_score_normalize(all_points)          # Normalize the data using z-score normalization
    print(f'Normalized data: {all_points}')

    # Run the customized k-means algorithm
    # centroids, c = custom_kmeans(all_points, args.k, args.max_iters)
    # print(f'Final centroids: {centroids}')
    # plot_graph(centroids, c, all_points)


if __name__ == '__main__':
    main()

# Output:
