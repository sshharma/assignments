import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
import warnings
warnings.filterwarnings('ignore')

class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.x_train = None  # Training feature set
        self.y_train = None  # Training labels

    def fit(self, x_train, y_train):
        """Store the training data."""
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        """Predict the class labels for the provided data."""
        predictions = []
        for test_point in x_test:
            distances = self._calculate_distances(test_point)
            neighbors = self._get_neighbors(distances)
            assigned_class = self._assign_class(neighbors)
            predictions.append(assigned_class)
        return np.array(predictions)

    def _calculate_distances(self, test_point):
        """Calculate the Euclidean distance between a test point and all training points."""
        distances = np.sqrt(np.sum((self.x_train - test_point) ** 2, axis=1))
        return distances

    def _get_neighbors(self, distances):
        """Get the indices of k nearest neighbors."""
        return np.argsort(distances)[:self.k]

    def _assign_class(self, neighbor_indices):
        """Assign the class based on majority voting."""
        neighbor_labels = self.y_train[neighbor_indices]
        counts = np.bincount(neighbor_labels + 1)  # Shift labels to make them non-negative
        return counts.argmax() - 1  # Shift back to original labels

def main():
    # Step 1: Load the dataset
    data = pd.read_csv('data/wdbc.data.mb.csv', header=None)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Step 1: Check if normalization is needed
    feature_ranges = x.max(axis=0) - x.min(axis=0)
    print("Feature ranges before normalization:\n", feature_ranges)

    # Since features have varying ranges, normalization is needed
    scaler = MinMaxScaler()
    x_normalized = scaler.fit_transform(x)
    print("\nFeature ranges after normalization:\n", x_normalized.max(axis=0) - x_normalized.min(axis=0))

    # Step 2: Split the dataset into training and testing sets (70% training, 30% testing)
    x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.3, random_state=42, stratify=y)

    # Step 5 & 6: Test the kNN algorithm for different k values
    for k in [3, 5, 7, 9]:
        print(f"\nTesting custom kNN classifier with k={k}")
        knn = KNNClassifier(k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)

        # Calculate accuracy and confusion matrix
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)

    # Step 7: Compare with built-in kNN classifier
    for k in [3, 5, 7, 9]:
        print(f"\nTesting built-in kNN classifier with k={k}")
        sklearn_knn = SklearnKNN(n_neighbors=k)
        sklearn_knn.fit(x_train, y_train)
        y_pred = sklearn_knn.predict(x_test)

        # Calculate accuracy and confusion matrix
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)


if __name__ == "__main__":
    main()
