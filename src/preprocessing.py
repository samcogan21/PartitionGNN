from typing import List, Dict

# preprocessing.py



from utils import load_graphs_from_raw, temps_grid, split_dataset
from enumerate import compute_and_cache_energies, compute_and_cache_lnZ



def run_preprocessing(raw_dir: str,
                      energy_cache: str,
                      lnz_cache: str,
                      split_out: str,
                      temp_start: float = 0.5,
                      temp_stop: float = 4.0,
                      temp_count: int = 10) -> Dict[str, List[int]]:
    """
    Complete preprocessing pipeline:
      1. Load raw graphs
      2. Compute & cache energies
      3. Build temperature grid
      4. Compute & cache lnZ
      5. Split dataset and save split JSON

    Returns split dict with 'train','val','test' index lists.
    """
    # 1. Load graphs
    graphs = load_graphs_from_raw(raw_dir)

    # 2. Compute & cache energy arrays
    compute_and_cache_energies(graphs, energy_cache)

    # 3. Temperature list
    temps = temps_grid(start=temp_start, stop=temp_stop, count=temp_count)

    # 4. Compute & cache lnZ labels
    compute_and_cache_lnZ(energy_cache, temps, lnz_cache)

    # 5. Split graphs
    prep_data = split_dataset(
        num_graphs=len(graphs),
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        seed=42,
        shuffle=True,
        out_path=split_out
    )

    return prep_data
