import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import glob


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

def load_graphs_from_raw(raw_folder: str = None) -> list:
    """
    Loads all .gpickle files from the raw folder and returns a list of graphs.
    Assumes graphs already have 'J' values.
    """
    if raw_folder is None:
        # Resolve raw folder relative to this file
        here = os.path.dirname(os.path.abspath(__file__))
        raw_folder = os.path.normpath(os.path.join(here, "../db/raw"))
    graphs = []
    files = glob.glob(os.path.join(raw_folder, "*.gpickle"))
    for file in files:
        with open(file, "rb") as f:
            G = pickle.load(f)
        graphs.append(G)
    return graphs