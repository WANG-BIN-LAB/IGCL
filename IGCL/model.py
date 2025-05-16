import torch
from torch import nn
from MSTE_GAT import Encoder
class IGCL(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, v1, v2, v3, v4):
        super(IGCL, self).__init__()
        self.encoder = Encoder(360, hidden_dim, num_gc_layers, v1, v2, v3, v4)

    def forward(self, x, edge_index):
        y, M, f_e = self.encoder(x, edge_index)
        return y, M, f_e

    def HCNT_Xent(self, x, x_aug):
        T = 0.1
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        # Compute similarity matrix
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)

        # Process negative samples
        neg_sim_matrix = sim_matrix.clone()
        neg_sim_matrix[range(batch_size), range(batch_size)] = 0  # Mask positive samples
        neg_sim_matrix = torch.exp(neg_sim_matrix / T) - 1
        neg_sim_matrix = torch.clamp(neg_sim_matrix, min=1e-10)

        # Process positive samples
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        pos_sim = torch.exp(pos_sim / T)
        pos_sim = torch.clamp(pos_sim, min=1e-10)

        # Select top 5% hardest negatives
        k = max(int(batch_size * 0.05), 1)
        top_k_neg_sim = torch.topk(neg_sim_matrix.view(batch_size, -1), k=k, dim=1)[0]

        # Calculate final loss
        loss = pos_sim / (top_k_neg_sim.sum(dim=1) + pos_sim)
        return -torch.log(loss).mean()
