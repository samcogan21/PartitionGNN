import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from enumerate import compute_energies
from utils import lnZ, load_graphs_from_raw
from dataset import PartitionDataset
from gnn import PartitionGNN

def main():

    #ToDo
    # 1. Load graphs from raw files
    graphs = []
    # 2. Define temperatures
    temps = temps = list(torch.linspace(0.5, 5.0, steps=10).tolist())
    # 3. Compute energies and lnZ for each graph at each temperature
    lnZ_dict = {}

    dataset = PartitionDataset(graphs, temps, lnZ_dict)
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

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

        # Validation
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                total_val += criterion(out, batch.y.view(-1)).item() * batch.num_graphs
        avg_val = total_val / len(val_loader.dataset)

        print(f"Epoch {epoch:02d}: Train MSE={avg_train:.4f}, Val MSE={avg_val:.4f}")

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
    print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}")


if __name__ == '__main__':
    main()
