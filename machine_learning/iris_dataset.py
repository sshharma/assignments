"""
This is the implementation of the original k-means algorithm for Iris dataset.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
    original_labels = df.iloc[:, 4].values
    features = scaler.fit_transform(features)

    # Running the original k-means algorithm
    kmeans = KMeans(n_clusters=args.k, max_iter=args.max_iters)  # random_state=args.random_state)
    kmeans.fit(features)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    print(f'Final centroids: {centroids}')

    # plot the clusters
    plot_graph(centroids, labels, features)

    # Plot the clusters using attributes 3 and 4
    plt.scatter(features[:, 2], features[:, 3], c=labels)
    plt.scatter(centroids[:, 2], centroids[:, 3], c='red', marker='x')

    plt.title('Plot using feature 3 and 4.')
    plt.show()

    # Print accuracy
    accuracy = np.mean(original_labels == labels)
    print(f'Accuracy: {accuracy}')


if __name__ == '__main__':
    main()
