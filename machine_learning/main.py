# Implementing K-Means Clustering Algorithm with customized kmeans function
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pygments.lexer import default
import logging

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


def get_centroids(k):
    """
    Get the centroids for the given number of clusters
    :param k: number of clusters
    :return centroids: list of centroids
    """
    if k==2:
        return [[0, 5], [180, 5]]
    elif k==4:
        return [[0, 0], [100, 0], [100, 10], [160, 0]]
    elif k==6:
        return [[20, 0], [20, 10], [100, 0], [100, 10], [160, 0], [160, 10]]
    else:
        logging.error('Invalid number of clusters. Please choose 2, 4 or 6.')
        return None


def custom_kmeans(points, k, max_iters):
    # initialize the centroids randomly
    # raw_centroids = points[np.random.choice(points.shape[0], k, replace=False)]

    # chosen centroids manually, that are far apart from each other
    raw_centroids = get_centroids(k)
    centroids = np.array(raw_centroids)

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

        # Find the new centroids
        centroids = np.array([points[c_ == k].mean(axis=0) for k in range(k)])
        print(f'centroids for iteration {i}: {centroids}')
        # compare the new centroids with the last centroids upto 2 decimal places
        if np.all(np.round(centroids, 2) == np.round(last_centroids, 2)):
            print(f'Converged after {i} iterations.')
            break
        last_centroids = centroids
        plot_graph(centroids, c_, points)

    return centroids,c_


def main():
    # read the data from command line
    parser = argparse.ArgumentParser(description='K-Means Clustering Algorithm')
    parser.add_argument('--train_dir', type=str, help='data file', default='data/kmtest.csv')
    parser.add_argument('--k', type=int, help='number of clusters', default=6)
    parser.add_argument('--max_iters', type=int, help='maximum number of iterations', default=100)
    args = parser.parse_args()


    df = pd.read_csv(args.train_dir, header=None)  # (19, 2)
    # print(f'shape of the data: {df.shape}')

    # use first column as x-axis and second column as y-axis
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
    plt.show()

    # Run the customized k-means algorithm
    all_points = df.values
    centroids, c = custom_kmeans(all_points, args.k, args.max_iters)
    print(f'Final centroids: {centroids}')
    plot_graph(centroids, c, all_points)



if __name__ == '__main__':
    main()

# Output:
