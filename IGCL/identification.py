from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm

from model import *

def custom_recognition_by_correlation(train_features, test_features):
    """Use correlation for individual recognition and calculate accuracy,
    while saving the correlation matrix and calculating the average difference between diagonal and non-diagonal elements"""
    test_curr = 0
    correct_predictions = 0
    n_train = len(train_features)
    n_test = len(test_features)

    # Create an n_train x n_test correlation matrix
    correlation_matrix = np.zeros((n_train, n_test))

    for test_sample in test_features:
        correlations = []

        for j, train_sample in enumerate(train_features):
            # Calculate correlation with training sample (using cosine similarity or Pearson correlation coefficient)
            correlation = 1 - cosine(train_sample, test_sample)
            # correlation = np.corrcoef(train_sample, test_sample)[0, 1]
            correlations.append(correlation)

            # Save correlation to correlation matrix
            correlation_matrix[j, test_curr] = correlation

        # Find the index of the sample with the highest correlation
        closest_index = np.argmax(correlations)
        if test_curr == closest_index:
            correct_predictions += 1
        test_curr += 1

    # Calculate accuracy
    accuracy = correct_predictions / n_test  # Accuracy = number of correct predictions / number of test samples
    # Return accuracy
    return accuracy
