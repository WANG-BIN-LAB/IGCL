from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import init
from torch_geometric.nn import GATConv
import numpy as np
from methods import some_csr_matrix_object_torch


class Encoder(torch.nn.Module):

    def _initialize_gat_weights(self):
        # Iterate over GAT layer parameters
        for gat_layer in [self.gat, self.gat2]:
            for param in gat_layer.parameters():
                if param.dim() == 2:  # If it's a weight matrix
                    with torch.no_grad():
                        # Create a 360x1440 matrix containing four 360x360 identity matrices concatenated
                        param.copy_(torch.cat([torch.eye(360) for _ in range(4)], dim=0))  # Concatenate four identity matrices
                elif param.dim() == 1:  # If it's a bias
                    init.zeros_(param)  # Initialize bias to 0

    def __init__(self, num_features, dim, num_gc_layers, v1, v2, v3, v4):
        super(Encoder, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.bns = torch.nn.ModuleList()

        # Define GATConv layers
        self.gat = GATConv(360, dim, heads=4, concat=False)
        self.gat2 = GATConv(360, dim, heads=4, concat=False)
        # Initialize multi-scale sparsity
        self.weights = nn.Parameter(torch.tensor([v1, v2, v3, v4], requires_grad=True))

        # Initialize weights and biases
        self._initialize_gat_weights()
        for i in range(num_gc_layers):
            bn = torch.nn.BatchNorm1d(dim)
            self.bns.append(bn)


    def forward(self, x, edge_index):
        xs = []
        copy_x = x
        # Adaptive multi-scale topology enhancement
        if self.weights[0] <= 1:
            x = x + some_csr_matrix_object_torch(copy_x, self.weights[0])
        if self.weights[1] <= 1:
            x = x + some_csr_matrix_object_torch(copy_x, self.weights[1])
        if self.weights[2] <= 1:
            x = x + some_csr_matrix_object_torch(copy_x, self.weights[2])
        if self.weights[3] <= 1:
            x = x + some_csr_matrix_object_torch(copy_x, self.weights[3])
        x = F.tanh(self.gat(x, edge_index))
        x = self.bns[0](x)
        xs.append(x)
        x = x.view(-1, 360, 360)
        recon_x = x
        # Get batch size
        batch_size = recon_x.shape[0]

        # Set diagonal elements of each matrix to 0
        for i in range(batch_size):
            recon_x[i, torch.arange(360), torch.arange(360)] = 0  # Set diagonal elements to 0
        #
        # # Flatten the matrix to (batch, 360*360)
        recon_x = recon_x.reshape(batch_size, -1)

        return recon_x, torch.cat(xs, 1), xs[0]

    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data[0]
                data.to(device)
                x, edge_index, weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x, _, _ = self.forward(x, edge_index)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

    def get_embeddings1(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data[1]
                data.to(device)
                x, edge_index, weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x, _, _ = self.forward(x, edge_index)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y