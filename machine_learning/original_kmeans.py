"""
This is the implementation of the original k-means algorithm.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from machine_learning.customized_kmeans import plot_graph


def z_score_normalize(data):
    """
    Normalize the data using the z-score normalization
    :param data: data to be normalized
    :return: normalized data
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std_replaced = np.where(std == 0, 1, std)                       # Replace 0 with 1 to avoid division by zero
    return (data - mean) / std_replaced



def main():
    parser = argparse.ArgumentParser(description='K-Means Clustering Algorithm')
    parser.add_argument('--train_dir', type=str, help='Data file', default='data/kmtest.csv')
    parser.add_argument('--k', type=int, help='Number of clusters', default=4)
    parser.add_argument('--max_iters', type=int, help='Maximum number of iterations', default=50)
    parser.add_argument('--random_state', type=int, help='Random state for initialization', default=42)
    args = parser.parse_args()

    # read the data from the file
    df = pd.read_csv(args.train_dir, header=None)  # (19, 2)
    # print(f'shape of the data: {df.shape}')

    # use first column as x-axis and second column as y-axis to plot the initial data points
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
    plt.show()

    all_points = df.values
    all_points = z_score_normalize(all_points)    # Normalize the data points

    # Running the original k-means algorithm
    kmeans = KMeans(n_clusters=args.k, max_iter=args.max_iters, random_state=args.random_state)
    kmeans.fit(all_points)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    print(f'Final centroids: {centroids}')
    plot_graph(centroids, labels, all_points)


if __name__ == '__main__':
    main()
