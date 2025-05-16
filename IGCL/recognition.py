from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm

from model import *

def compute_correlation_for_test_sample(test_sample, train_features, train_labels, index):
    correlations = []

    for train_sample in train_features:
        # Initialize row correlation list
        row_correlations = []
        # train_sample and test_sample are 2D matrices (each row represents a feature vector), calculate the correlation row by row.
        for row_train, row_test in zip(train_sample, test_sample):
            # Directly calculate the correlation of each row.
            correlation = 1 - cosine(row_train, row_test)
            row_correlations.append(correlation)

        # Calculate the average correlation of each training sample and test sample.
        avg_correlation = np.mean(row_correlations)
        correlations.append(avg_correlation)

    # Find the index with the highest correlation to the sample.
    closest_index = np.argmax(correlations)
    zheng = correlations[index]
    correlations[index] = -1
    index_fu = np.argmax(correlations)
    top_fu = correlations[index_fu]
    return train_labels[closest_index], zheng-top_fu  # Return the label closest to the sample.


def custom_recognition_by_correlation(train_features, test_features, train_labels, max_workers=6):
    """Use relevance to identify individuals, compare similarity line by line and then average, utilizing multi-core acceleration."""
    predictions = []
    differ = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to display a progress bar, where the iterable consists of the indices and sample data of the test samples.
        futures = [
            executor.submit(compute_correlation_for_test_sample, test_sample, train_features, train_labels, index)
            for index, test_sample in enumerate(test_features)
        ]

        # Use tqdm to display a progress bar
        for future in tqdm(futures, desc="Processing test samples", total=len(futures)):
            closest_label, correlation_difference = future.result()
            predictions.append(closest_label)
            differ.append(correlation_difference)


    return predictions