import os

import numpy as np
import torch
from scipy.io import loadmat



def some_csr_matrix_object_torch(matrix, threshold):
    # Ensure input is a PyTorch tensor
    T = 0.2
    matrix = torch.as_tensor(matrix)
    matrix = matrix.view(-1, 360, 360)
    B, N, _ = matrix.shape  # Get batch size and matrix dimensions

    # Initialize output sparse matrix as a zero tensor
    sparse_matrix = torch.zeros_like(matrix)

    for i in range(B):  # Iterate over each matrix
        # Use the sigmoid function to perform a smooth threshold operation
        mask = torch.sigmoid((matrix[i] - threshold) / T)
        sparse_matrix[i] = matrix[i] * mask

    sparse_matrix = sparse_matrix.view(-1, 360)
    return sparse_matrix

def some_csr_matrix_object(matrix, sparsity_ratio=0.7):
    B, N, _ = matrix.shape  # Get batch size and matrix dimensions
    sparse_matrix = np.zeros_like(matrix)  # Initialize output matrix

    for i in range(B):  # Iterate over each matrix
        flat_matrix = matrix[i].flatten()  # Flatten current matrix to 1D
        k = int(flat_matrix.size * sparsity_ratio)  # Calculate 70th percentile position
        threshold = np.partition(flat_matrix, k)[k]  # Get 70th percentile value

        # Set elements below 70th percentile to 0, keep original values of elements above or equal
        sparse_matrix[i] = np.where(matrix[i] >= threshold, matrix[i], 0.0)

    return sparse_matrix


def load_fc_matrices(directory):
    """Load functional connectivity matrices"""
    fc_matrices, ids = [], []
    for filename in os.listdir(directory):
        if filename.endswith('.mat'):
            data = loadmat(os.path.join(directory, filename))  # Load .mat file
            fc_matrix = data['FC']  # Adjust the key if necessary
            fc_matrices.append(fc_matrix)
            ids.append(int(filename.split('_')[1].split('.')[0]))  # Extract ID from filename
    return np.array(fc_matrices), np.array(ids)  # Return matrices and ID array