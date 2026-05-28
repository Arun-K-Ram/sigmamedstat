"""
SigmaMedStat -Per-Alarm Static Model (Exp 01)
Computes per-alarm-type AUC for the static EfficientNet
baseline to enable direct comparison with temporal model.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import json

DATA_DIR_STATIC = Path("C:/Users/Arun/Documents/git/crip-x/backend/data/scalograms")
DATA_DIR_NAMES  = Path("C:/Users/Arun/Documents/git/crip-x/backend/data/scalograms_temporal")
RESULTS_DIR     = Path("results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED        = 42
EPOCHS      = 30
PATIENCE    = 6
BATCH_SIZE  = 16

torch.manual_seed(SEED)
np.random.seed(SEED)


def build_static_model():
    m    = models.efficientnet_b0(
               weights=models.EfficientNet_B0_Weights.DEFAULT)
    orig = m.features[0][0]
    new  = nn.Conv2d(4, orig.out_channels, orig.kernel_size,
                     orig.stride, orig.padding, bias=False)
    with torch.no_grad():
        new.weight[:, :3] = orig.weight
        new.weight[:, 3]  = orig.weight.mean(dim=1)
    m.features[0][0] = new
    m.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    return m


def main():
    print("Per-alarm AUC -Static EfficientNet (Exp 01)")
    print("=" * 55)

    X     = np.load(DATA_DIR_STATIC / "X.npy")
    y     = np.load(DATA_DIR_STATIC / "y.npy")
    names = list(np.load(DATA_DIR_NAMES / "names_seq.npy"))

    print(f"Dataset: {X.shape}")

    alarm_types = {
        "v": "Ventricular Flutter",
        "a": "Asystole",
        "t": "Tachycardia",
        "b": "Bradycardia",
        "f": "Ventricular Fibrillation",
    }

    skf       = StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=SEED)
    all_probs = np.zeros(len(y))

    for fold, (tr_idx, vl_idx) in enumerate(
            skf.split(X, y), 1):
        print(f"  Fold {fold}/5...")

        n0 = (y[tr_idx] == 0).sum()
        n1 = (y[tr_idx] == 1).sum()
        w  = torch.tensor([len(tr_idx)/(2*n0),
                            len(tr_idx)/(2*n1)],
                           dtype=torch.float32).to(DEVICE)

        X_tr = torch.tensor(X[tr_idx], dtype=torch.float32)
        y_tr = torch.tensor(y[tr_idx], dtype=torch.long)
        X_vl = torch.tensor(X[vl_idx], dtype=torch.float32)
        y_vl = torch.tensor(y[vl_idx], dtype=torch.long)

        tr_ld = DataLoader(TensorDataset(X_tr, y_tr),
                           batch_size=BATCH_SIZE, shuffle=True)
        vl_ld = DataLoader(TensorDataset(X_vl, y_vl),
                           batch_size=BATCH_SIZE)

        model = build_static_model().to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-4)
        crit  = nn.CrossEntropyLoss(weight=w)

        best_auc = 0
        pat      = 0
        best_st  = None

        for epoch in range(1, EPOCHS + 1):
            model.train()
            for Xb, yb in tr_ld:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                crit(model(Xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)
                opt.step()

            model.eval()
            probs, trues = [], []
            with torch.no_grad():
                for Xb, yb in vl_ld:
                    p = torch.softmax(
                        model(Xb.to(DEVICE)), dim=1)[:, 1]
                    probs.extend(p.cpu().numpy())
                    trues.extend(yb.numpy())

            auc = roc_auc_score(trues, probs)
            if auc > best_auc:
                best_auc = auc
                pat      = 0
                best_st  = {k: v.clone()
                            for k, v in model.state_dict().items()}
            else:
                pat += 1
                if pat >= PATIENCE:
                    break

        model.load_state_dict(best_st)
        model.eval()
        final_probs = []
        with torch.no_grad():
            for Xb, _ in DataLoader(
                    TensorDataset(X_vl, y_vl),
                    batch_size=BATCH_SIZE):
                p = torch.softmax(
                    model(Xb.to(DEVICE)), dim=1)[:, 1]
                final_probs.extend(p.cpu().numpy())
        all_probs[vl_idx] = final_probs

    # Per-alarm AUC
    print("\nPer-alarm-type AUC -Static (Exp 01):")
    print(f"{'Alarm Type':<25} {'n':>4} {'AUC':>7}")
    print("-" * 38)

    static_results = {}
    for code, name in alarm_types.items():
        mask = np.array([n[0] == code for n in names])
        if mask.sum() < 5:
            continue
        auc = roc_auc_score(y[mask], all_probs[mask])
        static_results[name] = round(float(auc), 4)
        print(f"  {name:<25} {mask.sum():>4} {auc:>7.4f}")

    overall = roc_auc_score(y, all_probs)
    print(f"\n  {'Overall':<25} {len(y):>4} {overall:>7.4f}")

    # Compare with temporal
    temporal = {
        "Ventricular Flutter": 0.820,
        "Asystole":            0.722,
        "Tachycardia":         0.750,
        "Bradycardia":         0.810,
        "Ventricular Fibrillation": 0.733,
    }

    print("\nTemporal vs Static improvement per alarm type:")
    print(f"{'Alarm Type':<25} {'Static':>7} "
          f"{'Temporal':>9} {'Gain':>7}")
    print("-" * 52)
    improvements = {}
    for name in alarm_types.values():
        if name not in static_results:
            continue
        static_auc   = static_results[name]
        temporal_auc = temporal.get(name, 0)
        gain         = temporal_auc - static_auc
        improvements[name] = {
            "static":   static_auc,
            "temporal": temporal_auc,
            "gain":     round(gain, 4),
        }
        print(f"  {name:<25} {static_auc:>7.4f} "
              f"{temporal_auc:>9.4f} {gain:>+7.4f}")

    results = {
        "static_per_alarm":  static_results,
        "temporal_per_alarm": temporal,
        "improvements":      improvements,
        "static_overall":    round(float(overall), 4),
        "temporal_overall":  0.822,
    }

    out = RESULTS_DIR / "per_alarm_comparison.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()