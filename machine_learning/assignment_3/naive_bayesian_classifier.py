"""
Name: Sachin Sharma
KSUID: 0011
Project: 3
Title: Naive Bayesian Classifier
"""



import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from collections import defaultdict
from sklearn.metrics import roc_curve, auc

class NaiveBayesClassifier:
    def __init__(self)-> None:
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y)-> None:
        """
        Fit the Naive Bayes classifier to the data.
        :param X: features
        :param y: labels
        :returns: None
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        # Calculate priors and mean, var for each class
        for cls in self.classes:
            X_c = X[y == cls]
            self.mean[cls] = X_c.mean(axis=0)
            self.var[cls] = X_c.var(axis=0) + 1e-9  # Add small value to prevent division by zero
            self.priors[cls] = X_c.shape[0] / n_samples

    def _calculate_likelihood(self, cls, x)-> np.ndarray:
        """
        ToDo 2A: Calculate the likelihood of the features given the class i.e. P(x|y)
        :param cls: class
        :param x: features
        :return: likelihood
        """
        mean = self.mean[cls]
        var = self.var[cls]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        likelihood = numerator / denominator
        return likelihood.prod()

    def _calculate_posterior(self, x)-> dict:
        """
        ToDo 2B: Calculate the posterior probability of the classes given the features i.e. P(y|x)
        :param x: features
        :return: posterior probabilities
        """
        posteriors = {}
        for cls in self.classes:
            prior = np.log(self.priors[cls])
            # Calculate likelihood of the features                          -- added small value to prevent log(0)
            likelihood = np.sum(np.log(self._calculate_likelihood(cls, x) + 1e-9))
            posteriors[cls] = prior + likelihood
        return posteriors

    def predict(self, x)-> np.ndarray:
        """
        Predict the class labels for the provided data.
        :param x: features
        :return: class labels
        """
        y_pred = []
        for _, x_ in x.iterrows():
            posteriors = self._calculate_posterior(x_)
            y_pred.append(max(posteriors, key=posteriors.get))
        return np.array(y_pred)



def accuracy(y_true, y_pred)-> float:
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred)-> pd.DataFrame:
    classes = np.unique(y_true)
    matrix = pd.DataFrame(
        np.zeros((len(classes), len(classes)), dtype=int),
        index=classes, columns=classes
    )
    for true_label, pred_label in zip(y_true, y_pred):
        matrix.loc[true_label, pred_label] += 1
    return matrix


def stratified_k_fold_cross_validation(data, k=5)-> list:
    # Separate data by class
    data_pos = data[data['label'] == 1]
    data_neg = data[data['label'] == -1]

    # Shuffle data
    data_pos = data_pos.sample(frac=1, random_state=42).reset_index(drop=True)
    data_neg = data_neg.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split data into k folds
    folds_pos = np.array_split(data_pos, k)
    folds_neg = np.array_split(data_neg, k)

    # Combine folds
    folds = []
    for i in range(k):
        fold = pd.concat([folds_pos[i], folds_neg[i]], ignore_index=True)
        folds.append(fold.sample(frac=1, random_state=42).reset_index(drop=True))
    return folds

def predict_proba(self, X)-> np.ndarray:
    """
    ToDo 3: Predict the probabilities of the positive class for the provided data.
    :param self:
    :param X:
    :return:
    """
    proba = []
    for _, x in X.iterrows():
        posteriors = self._calculate_posterior(x)
        # Convert log probabilities back to normal scale
        total = np.exp(list(posteriors.values())).sum()
        probs = {cls: np.exp(posteriors[cls])/total for cls in self.classes}
        proba.append(probs[1])  # Probability of positive class
    return np.array(proba)


def train_test_split_custom(data, test_size=0.3, random_state=42)-> tuple:
    """
    ToDo 1: Splits the data into training and testing sets (Default 70% training, 30% testing)
    :param data: data to be split
    :param test_size: proportion of the data to be used for testing (Default 0.3, can be customized)
    :param random_state: random seed for reproducibility (Default 42, can be customized)
    :return: training and testing sets
    """
    # Shuffle the data
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Calculate split index
    split_idx = int(len(data) * (1 - test_size))

    # Split the data
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    return X_train, X_test, y_train, y_test


def main()-> None:
    parser = argparse.ArgumentParser(description='Naive Bayesian Classifier')
    parser.add_argument('--train_dir', type=str, help='Data file', default='data/wdbc.data.mb.csv')
    args = parser.parse_args()

    # Load the dataset
    data = pd.read_csv(args.train_dir, header=None)

    # Assign column names
    num_features = data.shape[1] - 1
    feature_cols = [f'feature_{i}' for i in range(num_features)]
    data.columns = feature_cols + ['label']

    # Split the data into training and testing sets                             --0.3 test size
    X_train, X_test, y_train, y_test = train_test_split_custom(data, test_size = 0.3, random_state = 42)

    # Initialize and train the classifier
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = nb_classifier.predict(X_test)

    # Calculate accuracy
    test_accuracy = accuracy(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Perform 5-fold cross-validation
    folds = stratified_k_fold_cross_validation(data, k=5)

    accuracies = []
    confusion_matrices = []

    for i in tqdm(range(5), desc='Cross Validation: '):
        # Prepare training and validation data
        test_fold = folds[i]
        train_folds = [folds[j] for j in range(5) if j != i]
        train_data = pd.concat(train_folds, ignore_index=True)

        X_train_cv = train_data.iloc[:, :-1]
        y_train_cv = train_data.iloc[:, -1]
        X_test_cv = test_fold.iloc[:, :-1]
        y_test_cv = test_fold.iloc[:, -1]

        # Train the classifier
        nb_classifier_cv = NaiveBayesClassifier()
        nb_classifier_cv.fit(X_train_cv, y_train_cv)

        # Predict on the validation set
        y_pred_cv = nb_classifier_cv.predict(X_test_cv)

        # Calculate accuracy
        accuracy_cv = accuracy(y_test_cv, y_pred_cv)
        accuracies.append(accuracy_cv)

        # Generate confusion matrix
        conf_matrix_cv = confusion_matrix(y_test_cv, y_pred_cv)
        confusion_matrices.append(conf_matrix_cv)

        print(f"Fold {i + 1} Accuracy: {accuracy_cv:.4f}")
        print(f"Fold {i + 1} Confusion Matrix:")
        print(conf_matrix_cv)
        print("\n")

    # Add the method to our classifier
    NaiveBayesClassifier.predict_proba = predict_proba

    # Use the classifier from the first fold
    nb_classifier_cv = NaiveBayesClassifier()
    nb_classifier_cv.fit(X_train_cv, y_train_cv)
    y_scores = nb_classifier_cv.predict_proba(X_test_cv)

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test_cv.replace({-1: 0}), y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Fold 1')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()
