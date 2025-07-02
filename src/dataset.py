# src/dataset.py

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class PartitionDataset(Dataset):
    """
    PyTorch Dataset for (graph, temperature) → lnZ examples.

    Args:
        graphs (List[nx.Graph]): list of NetworkX graphs with a 'J' edge attribute
        temps (List[float]): list of temperatures
        lnZ (Dict[(int, float), float]): mapping from (graph_index, T) to exact lnZ
    """

    def __init__(self, graphs, temps, lnZ):
        self.graphs = graphs
        self.temps = temps
        self.lnZ = lnZ
        # Build all (graph_index, temperature_index) pairs
        self.examples = [
            (g_idx, t_idx)
            for g_idx in range(len(graphs))
            for t_idx in range(len(temps))
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        g_idx, t_idx = self.examples[idx]
        G = self.graphs[g_idx]
        T = self.temps[t_idx]
        lnZ = self.lnZ[(g_idx, T)]
        num_nodes = G.number_of_nodes()
        y = lnZ

        # 1) Build edge_index for an undirected graph (and its reverse)
        edges = list(G.edges())
        # shape [2, E]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # add reverse edges for undirected processing
        rev = edge_index[[1, 0], :]
        edge_index = torch.cat([edge_index, rev], dim=1)  # [2, 2E]

        # 2) Edge attributes: coupling J_ij per edge (duplicated for reverse)
        Js = [G[u][v]['J'] for u, v in edges]
        edge_attr = torch.tensor(Js, dtype=torch.float).view(-1, 1)  # [E,1]
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)  # [2E,1]

        # 3) Node features: broadcast temperature T to each node
        num_nodes = G.number_of_nodes()
        x = torch.full((num_nodes, 1), float(T), dtype=torch.float)

        # 4) Package into a PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([y], dtype=torch.float)
        )
        return data

