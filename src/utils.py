import numpy as np
from src.enumerate import compute_energies
import networkx as nx
import matplotlib.pyplot as plt
import pickle


def lnZ(energies: np.ndarray, T: float) -> float:
    """
    Compute ln Z at temperature T from the energy array using
    the log-sum-exp trick for stability.
    """
    E_min = energies.min()
    return -E_min/T + np.log(np.sum(np.exp(-(energies - E_min)/T)))

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