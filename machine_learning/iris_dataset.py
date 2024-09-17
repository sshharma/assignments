"""
This is the implementation of the original k-means algorithm for Iris dataset.
"""

import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

scaler = StandardScaler()

def compute_original_centers(features, labels):
    unique_labels = np.unique(labels)
    centers = []
    for label in unique_labels:
        class_features = features[labels == label]
        center = np.mean(class_features, axis=0)
        centers.append(center)
    return np.array(centers)




def plot_graph(centroids, labels, data, title='K-Means Clustering of Iris Data'):
    """
    Plot the kmeans clustering of the data

    :param centroids: centroids of the clusters
    :param labels: cluster assignments
    :param data: data points
    :param title: title of the plot
    """
    markers = ['o', 's', 'D']
    colors = ['red', 'green', 'blue']

    for i in range(len(centroids)):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 2], cluster_points[:, 3],
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    label=f'Cluster {i+1}')

    # Plot the centroids
    plt.scatter(centroids[:, 2], centroids[:, 3],
                marker='x', color='black', s=100, label='Centroids')

    plt.title(title)
    plt.xlabel('Petal Length (standardized)')
    plt.ylabel('Petal Width (standardized)')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='K-Means Clustering Algorithm')
    parser.add_argument('--train_dir', type=str, help='Data file', default='data/iris.csv')
    parser.add_argument('--k', type=int, help='Number of clusters', default=3)
    parser.add_argument('--max_iters', type=int, help='Maximum number of iterations', default=10)
    parser.add_argument('--random_state', type=int, help='Random state for initialization', default=149)
    args = parser.parse_args()

    np.set_printoptions(precision=3, suppress=True)         # Set the print options for numpy for 3 decimal points
    df = pd.read_csv(args.train_dir, header=None)           # read the data from the file
    features = df.iloc[:, 0:4].values                       # read 1-4 columns as features
    features = scaler.fit_transform(features)

    ground_truth = df.iloc[:, 4].values                                         # read the 5th column as ground truth
    # remove "'" from the flower names
    ground_truth = [name.replace("'", "") for name in ground_truth]

    iris_to_int = {'setosa': 0, 'versicolor': 1, 'virginica': 2}           # map the flower to integer
    ground_truth_int = np.array([iris_to_int[label] for label in ground_truth])       # convert the flower to integer
    # Compute the original centers using the ground truth labels
    original_centers = compute_original_centers(features, ground_truth_int)
    print(f'Original centers: {original_centers}')

    # Running the original k-means algorithm
    kmeans = KMeans(n_clusters=args.k, max_iter=args.max_iters, init='random', random_state=args.random_state, n_init=10)
    kmeans.fit(features)
    kmeans_centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    print(f'Final centroids: {kmeans_centroids}')

    # Calculate Stats
    ari = adjusted_rand_score(ground_truth_int, labels)
    inertia = kmeans.inertia_
    silhouette = silhouette_score(features, labels)
    # plot the clusters
    plot_graph(kmeans_centroids, labels, features)

    # print the stats
    print(f'Adjusted Rand Index: {ari}')
    print(f'Inertia: {inertia}')
    print(f'Silhouette Score: {silhouette}')


    # plot the ground truth
    plot_graph(original_centers, ground_truth_int, features, title='Ground Truth Plot')

    # Compute the cost matrix (Euclidean distances between centers)
    cost_matrix = pairwise_distances(kmeans_centroids, original_centers)

    # Find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Map K-Means centers to original centers
    matched_centroids = kmeans_centroids[row_ind]
    matched_original_centers = original_centers[col_ind]

    # Compute the distances between matched centers
    center_distances = np.linalg.norm(matched_centroids - matched_original_centers, axis=1)
    print(f'Distances between matched centers: {center_distances}')



if __name__ == '__main__':
    main()
