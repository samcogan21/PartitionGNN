import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


def sample_graph(n=16, p=0.3):
    """
    Generates a random graph using the Erdős–Rényi model.
    Ensures the graph is connected by resampling if necessary, and no isolated nodes.

    Args:
    n: Number of nodes in the graph.
    p: Probability of edge creation between any two nodes.

    Returns:
    G: A connected random graph with n nodes.
    """
    # Generate a random graph using the Erdős–Rényi model
    G = nx.erdos_renyi_graph(n, p)

    # Ensure the graph is connected, resample if not
    while not nx.is_connected(G) or any(len(list(G.neighbors(node))) == 0 for node in G.nodes()):
        G = nx.erdos_renyi_graph(n, p)

    return G


def attach_couplings(G, coupling_range=(0, 2)):
    """
    Attach random coupling constants to each edge in the graph.

    Args:
    G: NetworkX graph object.
    coupling_range: Tuple (min, max) for the uniform distribution of the couplings.

    Returns:
    G: The graph with couplings assigned to the edges.
    """
    for u, v in G.edges():
        # Assign a random coupling constant to each edge from the uniform distribution
        G[u][v]['J'] = np.random.uniform(coupling_range[0], coupling_range[1])

    return G


def save_graphs_to_gpickle(graph_list, output_dir=None):
    """
    Save generated graphs as .gpickle files in the project's db/raw directory by default.
    """
    # Determine default output directory if not provided
    if output_dir is None:
        # Find project root relative to this script
        this_dir = os.path.dirname(__file__)             # e.g. .../src
        project_root = os.path.abspath(os.path.join(this_dir, os.pardir))
        output_dir = os.path.join(project_root, "db", "raw")

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each graph
    for idx, G in enumerate(graph_list):
        filename = f"graph_{idx:03d}.gpickle"
        path = os.path.join(output_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(G, f)

    print(f"Saved {len(graph_list)} graphs to {output_dir}")


def create_graphs(num_graphs=500, n=16, p=0.3):
    """
    Creates a list of random graphs with 16 nodes, ensuring no isolated nodes and attaching couplings to edges.

    Args:
    num_graphs: The number of graphs to create.
    n: Number of nodes in each graph.
    p: Probability of edge creation between nodes.

    Returns:
    graphs: A list of generated graphs with couplings attached.
    """
    graphs = []

    for _ in range(num_graphs):
        # Step 1: Generate a connected graph with no isolated nodes
        G = sample_graph(n=n, p=p)

        # Step 2: Attach random couplings to the edges
        G = attach_couplings(G)

        # Add the generated graph to the list
        graphs.append(G)

    return graphs


def create_graphs_varying_size(num_graphs=500, n_range=(6, 20), p=0.3):
    """
    Creates a list of random graphs with variable node sizes randomly chosen from n_range.
    Ensures no isolated nodes and attaches random couplings to edges.

    Args:
    num_graphs: The number of graphs to create.
    n_range: Tuple (min, max) for the range of node numbers in each graph.
    p: Probability of edge creation between nodes.

    Returns:
    graphs: A list of generated graphs with couplings attached.
    """
    graphs = []
    for _ in range(num_graphs):
        n = np.random.choice(np.arange(n_range[0], n_range[1]+1, 4))
        G = sample_graph(n=n, p=p)
        G = attach_couplings(G)
        graphs.append(G)
    return graphs


# Example usage: Create 500 graphs and save them as gpickle files
graphs = create_graphs_varying_size(num_graphs=500)

# Sanity check: Print number of nodes in the first 20 graphs
for i, G in enumerate(graphs[:20]):
    print(f"Graph {i} has {G.number_of_nodes()} nodes")


# Visualize the second graph (graphs[1]) using matplotlib, for checking only
nx.draw(graphs[1], with_labels=True)
plt.show()

# Save the generated graphs to the raw data folder
save_graphs_to_gpickle(graphs)