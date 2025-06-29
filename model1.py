import torch
import torch.nn as nn


class MultiHeadParetoSetModel(nn.Module):
    def __init__(self, n_dim, n_obj, n_sets):
        """
        n_dim: decision variable dimension
        n_obj: objective space dimension
        n_sets: number of Pareto Sets (clusters from DBSCAN, for example)
        """
        super(MultiHeadParetoSetModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.n_sets = n_sets
        self.n_node = 1024

        # Shared backbone
        self.shared_fc1 = nn.Linear(n_obj, self.n_node)
        self.shared_fc2 = nn.Linear(self.n_node, self.n_node)

        # Multi-head: separate output layer for each Pareto Set
        self.output_heads = nn.ModuleList([
            nn.Linear(self.n_node, n_dim) for _ in range(n_sets)
        ])

    def forward(self, pref, ps_id):
        """
        pref: tensor of shape [batch_size, n_obj]
        ps_id: tensor of shape [batch_size] with values in [0, n_sets-1]
        """
        # Shared feature extraction
        x = torch.relu(self.shared_fc1(pref))
        x = torch.relu(self.shared_fc2(x))

        # Prepare an output tensor
        output = torch.zeros(pref.size(0), self.n_dim, dtype=torch.float64, device=pref.device)

        # Dispatch to correct head per sample in batch
        for i in range(self.n_sets):
            idx = (ps_id == i)  # boolean mask
            if idx.any():
                out_i = self.output_heads[i](x[idx])
                output[idx] = torch.sigmoid(out_i).to(torch.float64)

        return output
