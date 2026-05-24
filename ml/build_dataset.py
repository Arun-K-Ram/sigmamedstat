"""
Build the full scalogram dataset from all 750 PhysioNet records.
Saves as numpy arrays ready for PyTorch training.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
from cwt_pipeline import record_to_scalograms, DATA_DIR

OUTPUT_DIR = Path("../backend/data/scalograms")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_dataset():
    records = list(DATA_DIR.glob("*.hea"))
    records = [r.stem for r in records]
    
    print(f"Found {len(records)} records")
    
    X = []
    y = []
    names = []
    failed = []

    for record_stem in tqdm(records, desc="Processing records"):
        try:
            tensor, label = record_to_scalograms(record_stem)
            if tensor.shape != (4, 64, 64):
                failed.append((record_stem, f"wrong shape: {tensor.shape}"))
                continue
            X.append(tensor)
            y.append(label)
            names.append(record_stem)
        except Exception as e:
            failed.append((record_stem, str(e)))

    X = np.array(X, dtype=np.float32)  # (N, 4, 64, 64)
    y = np.array(y, dtype=np.int64)    # (N,)

    print(f"\nDataset shape: {X.shape}")
    print(f"Labels: {y.sum()} true alarms, {(1-y).sum()} false alarms")
    print(f"Failed: {len(failed)}")
    if failed:
        for name, err in failed[:5]:
            print(f"  {name}: {err}")

    # Save
    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)
    np.save(OUTPUT_DIR / "names.npy", np.array(names))

    print(f"\nSaved to {OUTPUT_DIR}")
    print(f"X: {X.shape} - {X.nbytes / 1e6:.1f} MB")

if __name__ == "__main__":
    build_dataset()