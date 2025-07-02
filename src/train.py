import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Fixes issue with MKL in PyTorch

import pickle
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

from enumerate import compute_energies
from utils import lnZ, load_graphs_from_raw
from dataset import PartitionDataset
from gnn import PartitionGNN

def main():
    print("Starting training process...")

    # 1. Load graphs from raw files
    graphs = load_graphs_from_raw()
    print(f"Loaded {len(graphs)} graphs.")

    # 2. Define temperatures
    temps = list(torch.linspace(0.5, 100, steps=10).tolist())
    print("Temperatures defined.")

    # 3. Load or compute lnZ dictionary and save it for future runs
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

    print("Completed computing/loading lnZ for all graphs.")

    dataset = PartitionDataset(graphs, temps, lnZ_dict)
    print(f"Dataset created with {len(dataset)} examples.")

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    print(f"Dataset split into {train_size} train, {val_size} val, and {test_size} test examples.")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PartitionGNN(hidden_dim=64, num_layers=3).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 6. Training loop
    epochs = 50
    for epoch in range(1, epochs + 1):
        print(f"Starting epoch {epoch}...")
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        avg_train = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}: Training completed. Avg MSE: {avg_train:.4f}")

        # Validation
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                total_val += criterion(out, batch.y.view(-1)).item() * batch.num_graphs
        avg_val = total_val / len(val_loader.dataset)
        print(f"Epoch {epoch}: Validation completed. Avg MSE: {avg_val:.4f}")

    print("Training completed. Starting test evaluation...")

    # 7. Test set evaluation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.cpu())
            trues.append(batch.y.view(-1).cpu())
    preds = torch.cat(preds)
    trues = torch.cat(trues)

    rmse = torch.sqrt(torch.mean((preds - trues)**2))
    mae = torch.mean(torch.abs(preds - trues))
    # Compute relative MAE as the average of absolute relative errors
    relative_mae = torch.mean(torch.abs((preds - trues) / trues)) * 100
    print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, Relative MAE (%): {relative_mae:.2f}%")
    print("Test evaluation completed.")

    # Save the trained model checkpoint
    torch.save(model.state_dict(), "PartitionGNN_checkpoint.pth")
    print("Model checkpoint saved as PartitionGNN_checkpoint.pth")

    # Predicted vs. True scatter plot
    plt.figure()
    plt.scatter(trues.numpy(), preds.numpy(), c='blue', alpha=0.6)
    plt.xlabel("True lnZ")
    plt.ylabel("Predicted lnZ")
    plt.title("Predicted vs. True lnZ")
    # plot a diagonal line
    min_val = min(trues.min().item(), preds.min().item())
    max_val = max(trues.max().item(), preds.max().item())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')

    # Save the scatter plot before showing
    plt.savefig("pred_vs_true.png")
    print("Scatter plot saved as pred_vs_true.png")
    plt.show()

    # 8. Additional analysis: Absolute Error vs Temperature and Graph Density
    temp_list, abs_error_list, density_list = [], [], []
    for idx in test_ds.indices:
        data = dataset[idx]
        data = data.to(device)
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(data).item()
        pred_value = pred
        true_value = data.y.item()
        error = abs(pred_value - true_value)
        T = data.x[0].item()
        n = data.x.shape[0]
        unique_edges = data.edge_index.shape[1] // 2
        max_edges = n * (n - 1) / 2
        density = unique_edges / max_edges if max_edges > 0 else 0
        temp_list.append(T)
        abs_error_list.append(error)
        density_list.append(density)

    # Plot: Absolute Error vs Temperature
    plt.figure()
    plt.scatter(temp_list, abs_error_list, c='blue', alpha=0.6)
    plt.xlabel("Temperature")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error vs Temperature")
    plt.savefig("abs_error_vs_temp.png")
    print("Plot saved as abs_error_vs_temp.png")
    plt.show()

    # Plot: Absolute Error vs Graph Density
    plt.figure()
    plt.scatter(density_list, abs_error_list, c='green', alpha=0.6)
    plt.xlabel("Graph Density")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error vs Graph Density")
    plt.savefig("abs_error_vs_graph_density.png")
    print("Plot saved as abs_error_vs_graph_density.png")
    plt.show()

if __name__ == '__main__':
    main()