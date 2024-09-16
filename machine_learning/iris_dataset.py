"""
This is the implementation of the original k-means algorithm for Iris dataset.
"""

import argparse
from random import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from machine_learning.customized_kmeans import plot_graph

scaler = StandardScaler()

def main():
    parser = argparse.ArgumentParser(description='K-Means Clustering Algorithm')
    parser.add_argument('--train_dir', type=str, help='Data file', default='data/iris.csv')
    parser.add_argument('--k', type=int, help='Number of clusters', default=3)
    parser.add_argument('--max_iters', type=int, help='Maximum number of iterations', default=50)
    parser.add_argument('--random_state', type=int, help='Random state for initialization', default=42)
    args = parser.parse_args()

    # read the data from the file
    df = pd.read_csv(args.train_dir, header=None)  # (150, 4)
    # read 1-4 columns as features
    features = df.iloc[:, 0:4].values
    features = scaler.fit_transform(features)

    ground_truth = df.iloc[:, 4].values                                         # read the 5th column as ground truth
    iris_to_int = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}           # map the flower to integer
    ground_truth_int = np.array([iris_to_int[label] for label in ground_truth])       # convert the flower to integer


    # Running the original k-means algorithm
    kmeans = KMeans(n_clusters=args.k, max_iter=args.max_iters, init=random)  # random_state=args.random_state)
    kmeans.fit(features)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    print(f'Final centroids: {centroids}')

    # Calculate Stats
    ari = adjusted_rand_score(ground_truth_int, labels)
    inertia = kmeans.inertia_
    silhouette = silhouette_score(features, labels)
    # plot the clusters
    plot_graph(centroids, labels, features)

    # print the stats
    print(f'Adjusted Rand Index: {ari}')
    print(f'Inertia: {inertia}')
    print(f'Silhouette Score: {silhouette}')


    # Plot the clusters using attributes 3 and 4
    plt.scatter(features[:, 2], features[:, 3], c=labels)
    plt.scatter(centroids[:, 2], centroids[:, 3], c='red', marker='x')

    plt.title('Plot using feature 3 and 4.')
    plt.show()


if __name__ == '__main__':
    main()
