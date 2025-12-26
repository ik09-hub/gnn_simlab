import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNModel(torch.nn.Module):
    """
    Base interface for GNN models.
    """
    def forward(self, data):
        raise NotImplementedError

class GCN(GNNModel):
    def __init__(self, in_channels: int, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze()  # [num_nodes]


MODEL_REGISTRY = {
    "gcn": GCN,
}
