import numpy as np
import networkx as nx


def compute_energies(G):
    """
    Compute the energy H(s) = -sum_{(i,j)} J_ij * s_i * s_j
    for all 2^N spin configurations of graph G.
    Returns:
        energies: numpy.ndarray of shape (2**N,)
    """
    N = G.number_of_nodes()

    # Build arrays of edges and couplings
    edges = np.array(list(G.edges()))
    Js = np.array([G[u][v]['J'] for u, v in edges], dtype=float)

    energies = np.empty(2 ** N, dtype=np.float64)
    for mask in range(2 ** N):
        # Decode the integer mask into a spin vector s of ±1
        bits = (mask >> np.arange(N)) & 1
        s = 2 * bits - 1

        # Compute the total energy for this spin pattern
        # via vectorized operation over all edges
        H = -np.sum(Js * s[edges[:, 0]] * s[edges[:, 1]])
        energies[mask] = H

    return energies


# ---- Test on a toy graph ----
"""
# 1. Create a simple 4-node path graph
G = nx.path_graph(4)

# 2. Assign a random coupling J in [0, 2) to every edge
for u, v in G.edges():
   G[u][v]['J'] = np.random.uniform(0, 2)

# 3. Compute all configuration energies
energies = compute_energies(G)

# 4. Print basic statistics
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of configurations (2^N): {len(energies)}")
print(f"Min energy: {energies.min():.2f}")
print(f"Max energy: {energies.max():.2f}")
print("All energies:", energies)

"""