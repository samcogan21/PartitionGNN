import os
import pickle
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

from utils import load_graphs_from_raw, lnZ
from dataset import PartitionDataset
from gnn import PartitionGNN
from enumerate import compute_energies

def load_lnZ_dict(graphs, temps):
    lnz_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../lnZ_dict.pkl")
    if os.path.exists(lnz_filepath):
        print("Loading precomputed lnZ dictionary...")
        with open(lnz_filepath, "rb") as f:
            lnZ_dict = pickle.load(f)
    else:
        lnZ_dict = {}
        for i, G in enumerate(graphs):
            energies = compute_energies(G)
            for T in temps:
                lnZ_dict[(i, T)] = lnZ(energies, T)
            print(f"Computed lnZ for graph {i}.")
        with open(lnz_filepath, "wb") as f:
            pickle.dump(lnZ_dict, f)
        print("Saved lnZ dictionary for future runs.")
    return lnZ_dict

def main():
    # 1. Load graphs from raw files
    graphs = load_graphs_from_raw()
    print("Unique graph sizes:", sorted({G.number_of_nodes() for G in graphs}))
    print(f"Loaded {len(graphs)} graphs.")

    # 2. Define temperatures
    temps = list(torch.linspace(0.5, 5.0, steps=10).tolist())
    print("Temperatures defined.")

    # 3. Load (or compute) lnZ dictionary
    lnZ_dict = load_lnZ_dict(graphs, temps)
    print("Completed computing/loading lnZ for all graphs.")

    dataset = PartitionDataset(graphs, temps, lnZ_dict)
    print(f"Dataset created with {len(dataset)} examples.")

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    _, _, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size],
                                                   generator=torch.Generator().manual_seed(42))
    print(f"Dataset split: {test_size} test examples.")

    test_loader = DataLoader(test_ds, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PartitionGNN(hidden_dim=64, num_layers=3).to(device)

    # Load the best checkpoint from training
    checkpoint_path = "PartitionGNN_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model checkpoint from {checkpoint_path}")
    else:
        print("Checkpoint not found. Exiting.")
        return

    model.eval()
    preds_density, trues_density, Ns = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            preds_density.append(out.cpu())
            trues_density.append(batch.y.view(-1).cpu())
            counts = torch.bincount(batch.batch)
            Ns.extend(counts.tolist())
        preds_density = torch.cat(preds_density)
        trues_density = torch.cat(trues_density)
        Ns = torch.tensor(Ns, dtype=torch.float)

    # Recover total lnZ values per graph
    preds = preds_density * Ns
    trues = trues_density * Ns

    # Compute overall metrics
    rmse = torch.sqrt(torch.mean((preds - trues)**2))
    mae = torch.mean(torch.abs(preds - trues))
    print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    error_tensor = torch.abs(preds - trues)
    mean_error = error_tensor.mean().item()
    median_error = error_tensor.median().item()
    std_error = error_tensor.std().item()

    print("Test set results:")
    print(f"  RMSE: {rmse.item():.4f}")
    print(f"  MAE: {mae.item():.4f}")
    print(f"  Mean Absolute Error: {mean_error:.4f}")
    print(f"  Median Absolute Error: {median_error:.4f}")
    print(f"  Std. Dev. of Absolute Error: {std_error:.4f}")
    print(f"True lnZ stats: min = {trues.min().item():.4f}, max = {trues.max().item():.4f}, mean = {trues.mean().item():.4f}")

    # Recompute per-graph metrics for plotting
    temp_list, error_list, relative_error_list, density_list, nodes_list = [], [], [], [], []
    for idx in test_ds.indices:
        data = dataset[idx]
        data = data.to(device)
        # Set batch attribute for a single graph
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(data).item()
        n = data.x.shape[0]
        pred_total = pred * n
        true_total = data.y.item() * n
        error = abs(pred_total - true_total)
        rel_error = (error / true_total)*100 if true_total != 0 else 0
        T = data.x[0].item()
        unique_edges = data.edge_index.shape[1] // 2
        max_edges = n * (n - 1) / 2
        density = unique_edges / max_edges if max_edges > 0 else 0

        temp_list.append(T)
        error_list.append(error)
        relative_error_list.append(rel_error)
        density_list.append(density)
        nodes_list.append(n)

    # Plot 1: Predicted vs. True lnZ
    plt.figure()
    plt.scatter(trues.numpy(), preds.numpy(), c='blue', alpha=0.6)
    plt.xlabel("True lnZ")
    plt.ylabel("Predicted lnZ")
    plt.title("Predicted vs True lnZ")
    min_val = min(trues.min().item(), preds.min().item())
    max_val = max(trues.max().item(), preds.max().item())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.savefig("pred_vs_true.png")
    print("Scatter plot saved as pred_vs_true.png")
    plt.show()

    # Plot 2: Temperature vs Absolute Error in lnZ Prediction
    plt.figure()
    plt.scatter(temp_list, error_list, c='purple', alpha=0.6)
    plt.xlabel("Temperature")
    plt.ylabel("Absolute Error")
    plt.title("Temperature vs Absolute Error in lnZ Prediction")
    plt.savefig("temp_vs_abs_error.png")
    print("Plot saved as temp_vs_abs_error.png")
    plt.show()

    # Plot 3: Graph Density vs Absolute Error in lnZ Prediction
    plt.figure()
    plt.scatter(density_list, error_list, c='orange', alpha=0.6)
    plt.xlabel("Graph Density")
    plt.ylabel("Absolute Error")
    plt.title("Graph Density vs Absolute Error in lnZ Prediction")
    plt.savefig("density_vs_abs_error.png")
    print("Plot saved as density_vs_abs_error.png")
    plt.show()

    # Plot 4: Number of Nodes vs Absolute Error in lnZ Prediction
    plt.figure()
    plt.scatter(nodes_list, error_list, c='brown', alpha=0.6)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Absolute Error")
    plt.title("Number of Nodes vs Absolute Error in lnZ Prediction")
    plt.xlim(5, 20)
    plt.savefig("nodes_vs_abs_error.png")
    print("Plot saved as nodes_vs_abs_error.png")
    plt.show()

    # Plot 5: Number of Nodes vs Relative Error % in lnZ Prediction
    plt.figure()
    plt.scatter(nodes_list, relative_error_list, c='magenta', alpha=0.6)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Relative Error Percentage (%)")
    plt.title("Number of Nodes vs Relative Error in lnZ Prediction")
    plt.xlim(5, 20)
    plt.savefig("nodes_vs_rel_error.png")
    print("Plot saved as nodes_vs_rel_error.png")
    plt.show()

if __name__ == '__main__':
    main()
