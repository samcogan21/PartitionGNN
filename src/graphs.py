import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os


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


def save_graphs_to_gpickle(graph_list, output_dir=r"C:\Users\omerk\OneDrive - Technion\מסמכים\תואר שני\Courses\Statistical Thermodynamics\Final Project\PartitionGNN\data\raw"):
    """
    Save generated graphs as .gpickle files in the specified directory.

    Args:
    graph_list: List of NetworkX graphs to save.
    output_dir: Directory to save the .gpickle files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist

    for idx, G in enumerate(graph_list):
        path = os.path.join(output_dir, f"graph_{idx:03d}.gpickle")
        nx.write_gpickle(G, path)


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


# Example usage: Create 500 graphs and save them as gpickle files
graphs = create_graphs(num_graphs=500)


# Visualize the second graph (graphs[1]) using matplotlib, for checking only
nx.draw(graphs[1], with_labels=True)
plt.show()

G = nx.read_gpickle("data/raw/graph_001.gpickle")
for u, v, data in G.edges(data=True):
    print(f"Edge ({u}, {v}) has coupling J = {data['J']}")


# Save the generated graphs to the raw data folder
save_graphs_to_gpickle(graphs)