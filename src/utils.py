import numpy as np
from src.enumerate import compute_energies
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import json
import warnings
from typing import List, Dict
from networkx.readwrite import gpickle


def load_graphs_from_raw(raw_dir: str) -> List[nx.Graph]:
    """
    Load all graphs from raw_dir (.gpickle or .pkl) into NetworkX objects.
    Skips files that cannot be loaded, logging a warning for each.
    """
    graphs = []
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Directory not found: {raw_dir}")
    for fn in sorted(os.listdir(raw_dir)):
        path = os.path.join(raw_dir, fn)
        try:
            if fn.endswith('.gpickle'):
                G = gpickle.read_gpickle(path)
            elif fn.endswith('.pkl') or fn.endswith('.pickle'):
                with open(path, 'rb') as f:
                    G = pickle.load(f)
            else:
                warnings.warn(f"Skipping unsupported file type: {fn}")
                continue
            graphs.append(G)
        except Exception as e:
            warnings.warn(f"Failed to load graph '{fn}': {e}")
    if not graphs:
        warnings.warn(f"No graphs loaded from {raw_dir}")
    return graphs




def split_dataset(num_graphs: int,
                  train_frac: float = 0.7,
                  val_frac: float = 0.15,
                  test_frac: float = 0.15,
                  seed: int = 42,
                  out_path: str = None
                 ) -> Dict[str, List[int]]:
    """
    Randomly split graph indices into train/val/test; save JSON if out_path.
    """
    assert abs(train_frac+val_frac+test_frac-1.0)<1e-6
    idxs = list(range(num_graphs))
    rng = np.random.default_rng(seed)
    rng.shuffle(idxs)

    ntr = int(train_frac*num_graphs)
    nval = int(val_frac*num_graphs)
    train = idxs[:ntr]
    val   = idxs[ntr:ntr+nval]
    test  = idxs[ntr+nval:]
    split = {'train': train, 'val': val, 'test': test}

    if out_path:
        with open(out_path,'w') as f:
            json.dump(split, f, indent=2)
    return split





def temps_grid(start: float = 0.5,
               stop: float = 4.0,
               count: int = 10,
               log_space: bool = False
              ) -> List[float]:
    """
    Generate a dimensionless temperature list.

    Args:
        start, stop: range endpoints
        count: number of values
        log_space: if True, use geometric spacing

    Returns:
        list of floats
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if start <= 0 or stop <= 0:
        raise ValueError("temperature endpoints must be positive")
    if log_space:
        return np.exp(np.linspace(np.log(start), np.log(stop), count)).tolist()
    return np.linspace(start, stop, count).tolist()



def visualize_graph(G: nx.Graph) -> None:
    """
    Visualizes the given graph with node labels and edge couplings.
    """
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    edge_labels = {(u, v): f"{G[u][v].get('J', 0):.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

# ---- Test on a raw graph ----
with open("../db/raw/graph_000.gpickle", "rb") as f:
    G_loaded = pickle.load(f)

energies = compute_energies(G_loaded)
for T in [0.5, 1.0, 2.0]:
    print(f"lnZ(G, T={T}) = {lnZ(energies, T):.4f}")
visualize_graph(G_loaded)