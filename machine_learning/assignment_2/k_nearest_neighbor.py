"""
Name: Sachin Sharma
KSUID: 0011
Project: 2
Title: K-Nearest Neighbors Algorithm Implementation and Comparison
"""

import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
import warnings
warnings.filterwarnings('ignore')

class CustomKNNClassifier:
    def __init__(self, k) -> None:
        self.k = k
        self.x_train = None             # Training feature set
        self.y_train = None             # Training labels

    def fit(self, x_train, y_train) -> None:
        """Store the training data."""
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test) -> np.ndarray:
        """Predict the class labels for the provided data."""
        predictions = []
        for test_point in x_test:
            distances = self._calculate_distances(test_point)       # To Do 3: Calculate the distances
            neighbors = self._get_neighbors(distances)              # To Do 4-A: Get the indices of k nearest neighbors
            assigned_cat = self._assign_class(neighbors)            # To Do 4-B: Assign the category based on k nearest neighbors
            predictions.append(assigned_cat)
        return np.array(predictions)

    def _calculate_distances(self, test_point) -> np.ndarray:
        """Calculate the Euclidean distance between a test point and all training points."""
        distances = np.sqrt(np.sum((self.x_train - test_point) ** 2, axis=1))
        return distances

    def _get_neighbors(self, distances) -> np.ndarray:
        """Get the indices of k nearest neighbors."""
        return np.argsort(distances)[:self.k]

    def _assign_class(self, neighbor_indices) -> int:
        """Assign the class based on majority voting."""
        neighbor_labels = self.y_train[neighbor_indices]
        counts = np.bincount(neighbor_labels + 1)                       # Shift labels to make them non-negative
        return counts.argmax() - 1                                      # Shift back to original labels

def main():
    parser = argparse.ArgumentParser(description='K-Nearest Neighbors Algorithm')
    parser.add_argument('--k', type=int, help='Number of neighbors', default=3)
    parser.add_argument('--train_dir', type=str, help='Data file', default='data/wdbc.data.mb.csv')
    args = parser.parse_args()

    # Load the dataset
    data = pd.read_csv(args.train_dir, header=None)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # To Do 1: Check if normalization is needed
    feature_ranges = x.max(axis=0) - x.min(axis=0)
    print("Feature ranges before normalization:\n", feature_ranges)

    # Since features have varying ranges, using normalization to scale them to the same range
    scaler = MinMaxScaler()
    x_normalized = scaler.fit_transform(x)
    print("\nFeature ranges after normalization:\n", x_normalized.max(axis=0) - x_normalized.min(axis=0))

    # To Do 2: Splitting the dataset into training and testing sets (70% training, 30% testing)
    x_train, x_test, y_train, y_test = train_test_split(
        x_normalized,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y)

    # To Do 5 & 6: Test the custom kNN algorithm for given k value
    print(f"\nTesting custom kNN classifier with k={args.k}")
    knn = CustomKNNClassifier(args.k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # To Do 7: Compare with built-in kNN classifier
    print(f"\nTesting built-in kNN classifier with k={args.k}")
    sklearn_knn = SklearnKNN(n_neighbors=args.k)
    sklearn_knn.fit(x_train, y_train)
    y_pred = sklearn_knn.predict(x_test)

    # Calculate accuracy and confusion matrix for built-in kNN
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    main()
