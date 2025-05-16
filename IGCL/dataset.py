import torch
from torch_geometric.data import Data


def create_dataset(X_1, X_1_sparse, y_1, X_2, X_2_sparse, y_2):
    data_list = []
    num_graphs = X_1.shape[0]  # Number of graphs

    for i in range(num_graphs):
        # Get node features and adjacency matrix of the current graph
        node_features = torch.tensor(X_1[i], dtype=torch.float)  # [360, 360]
        adj_matrix = torch.tensor(X_1_sparse[i], dtype=torch.float)  # [360, 360]

        # Calculate edge list and edge weights
        edge_index = torch.nonzero(adj_matrix).t()  # Extract non-zero element indices, shape [2, num_edges]
        edge_attr = adj_matrix[edge_index[0], edge_index[1]]  # Extract edge weights, shape [num_edges]

        # Get label of the current graph
        y = torch.tensor([y_1[i]], dtype=torch.long)  # Assume each graph has one label

        # Create original Data object with node features, edge list, edge weights, and label
        data = Data(
            x=node_features,  # Node features
            edge_index=edge_index,  # Edge list
            edge_attr=edge_attr,  # Edge weights
            y=y  # Graph label
        )

        # Get node features and adjacency matrix of the current test graph
        node_features_2 = torch.tensor(X_2[i], dtype=torch.float)  # [360, 360]
        adj_matrix_2 = torch.tensor(X_2_sparse[i], dtype=torch.float)  # [360, 360]

        # Calculate edge list and edge weights
        edge_index_2 = torch.nonzero(adj_matrix_2).t()  # Extract non-zero element indices, shape [2, num_edges]
        edge_attr_2 = adj_matrix_2[edge_index_2[0], edge_index_2[1]]  # Extract edge weights, shape [num_edges]

        # Get label of the current test graph
        y_02 = torch.tensor([y_2[i]], dtype=torch.long)  # Assume each graph has one label

        # Create augmented Data object with node features, edge list, edge weights, and label
        data_aug = Data(
            x=node_features_2,  # Node features
            edge_index=edge_index_2,  # Edge list
            edge_attr=edge_attr_2,  # Edge weights
            y=y_02  # Graph label
        )

        # Store both original and augmented data together
        data_list.append((data, data_aug))
    return data_list