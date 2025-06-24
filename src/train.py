import os
import pickle
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from preprocessing import run_preprocessing
from dataset import PartitionDataset
from gnn import PartitionGNN


def main():
    # Paths
    raw_dir = "db/raw"
    energy_cache = "cache/energies"
    lnz_cache = "cache/lnz"
    split_out = "cache/split.json"

    # 1-5. Preprocess: load graphs, cache energies, lnZ, and split indices
    split = run_preprocessing(
        raw_dir=raw_dir,
        energy_cache=energy_cache,
        lnz_cache=lnz_cache,
        split_out=split_out,
        temp_start=0.5,
        temp_stop=5.0,
        temp_count=10
    )

    # Load split indices
    with open(split_out, 'r') as f:
        split = json.load(f)

    # Reconstruct lnZ_dict from cached files
    lnZ_dict = {}
    for idx in range(len(split['train']) + len(split['val']) + len(split['test'])):
        pkl_path = os.path.join(lnz_cache, f"lnz_{idx:03d}.pkl")
        with open(pkl_path, 'rb') as f:
            per_graph = pickle.load(f)
        for T, val in per_graph.items():
            lnZ_dict[(idx, float(T))] = val

    # Temperature grid
    from preprocessing import temps_grid
    temps = temps_grid(start=0.5, stop=5.0, count=10)

    # 6. Create datasets per split
    # graphs loaded inside PartitionDataset via raw_dir
    train_ds = PartitionDataset("db/raw", temps, lnZ_dict, split['train'])
    val_ds = PartitionDataset("db/raw", temps, lnZ_dict, split['val'])
    test_ds = PartitionDataset("db/raw", temps, lnZ_dict, split['test'])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # 7. Model, device, optimizer, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartitionGNN(hidden_dim=64, num_layers=3).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 8. Training loop
    epochs = 50
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.num_graphs
        avg_train = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_loss += criterion(model(batch), batch.y.view(-1)).item() * batch.num_graphs
        avg_val = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch:02d}  Train MSE={avg_train:.4f}  Val MSE={avg_val:.4f}")

    # 9. Testing
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds.append(model(batch).cpu())
            trues.append(batch.y.view(-1).cpu())
    preds = torch.cat(preds)
    trues = torch.cat(trues)

    rmse = torch.sqrt(torch.mean((preds - trues) ** 2))
    mae = torch.mean(torch.abs(preds - trues))
    print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}")


if __name__ == '__main__':
    main()
