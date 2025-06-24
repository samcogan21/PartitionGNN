import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv.graph_conv import GraphConv
from torch_geometric.nn.pool import global_mean_pool

class PartitionGNN(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(1, hidden_dim))  # input: 1-d node feature (T)
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, 1)          # output: scalar lnZ

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # Incorporate edge weights by multiplying adjacency messages
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_attr.view(-1))  # weighted message passing
            x = F.relu(x)                                            # ReLU activation
        # Pool node embeddings to graph embedding
        g = global_mean_pool(x, batch)                               # graph-level read-out
        return self.lin(g).squeeze(-1)                               # regression head
