import numpy as np
from src.enumerate import compute_energies
import networkx as nx


def lnZ(energies: np.ndarray, T: float) -> float:
    """
    Compute ln Z at temperature T from the energy array using
    the log-sum-exp trick for stability.
    """
    E_min = energies.min()
    return -E_min/T + np.log(np.sum(np.exp(-(energies - E_min)/T)))

# ---- Test on a toy graph ----

G = nx.erdos_renyi_graph(16, 0.2)
for u, v in G.edges():
    G[u][v]['J'] = np.random.uniform(0, 2)

energies = compute_energies(G)
for T in [0.5, 1.0, 2.0]:
    print(f"lnZ(G, T={T}) = {lnZ(energies, T):.4f}")
