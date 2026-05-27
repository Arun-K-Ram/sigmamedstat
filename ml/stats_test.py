"""
DeLong test for statistical significance of AUC improvement.
Compares Experiment 01 (static) vs Experiment 04 (LSTM)
on matched predictions from k-fold CV.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import json

DATA_DIR = Path("C:/Users/Arun/Documents/git/crip-x/backend/data")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED     = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

#  DeLong AUC variance estimator 
def delong_roc_variance(ground_truth, predictions):
    """
    Computes AUC variance using the DeLong method.
    Returns (auc, variance).
    """
    order      = np.argsort(-predictions)
    label_1_count = sum(ground_truth)
    label_0_count = len(ground_truth) - label_1_count

    sorted_labels = ground_truth[order]
    sorted_preds  = predictions[order]

    tp_count  = np.cumsum(sorted_labels)
    fp_count  = np.cumsum(1 - sorted_labels)

    # Placement of positive cases among negatives
    placement_neg = []
    for i, label in enumerate(sorted_labels):
        if label == 1:
            placement_neg.append(fp_count[i])

    placement_neg = np.array(placement_neg, dtype=float)

    # Placement of negative cases among positives
    placement_pos = []
    tp_at_end = tp_count[-1]
    for i, label in enumerate(sorted_labels):
        if label == 0:
            tp_before = tp_count[i] - sorted_labels[i]
            placement_pos.append(tp_before)

    placement_pos = np.array(placement_pos, dtype=float)

    auc = roc_auc_score(ground_truth, predictions)

    # Structural components
    V10 = placement_neg / label_1_count  if label_1_count > 0 else np.array([])
    V01 = placement_pos / label_0_count  if label_0_count > 0 else np.array([])

    S10 = np.var(V10, ddof=1) if len(V10) > 1 else 0
    S01 = np.var(V01, ddof=1) if len(V01) > 1 else 0

    variance = S10 / label_1_count + S01 / label_0_count
    return auc, variance


def delong_compare(y_true, probs_a, probs_b):
    """
    Tests H0: AUC_A == AUC_B using DeLong method.
    Returns (z_stat, p_value).
    """
    auc_a, var_a = delong_roc_variance(y_true, probs_a)
    auc_b, var_b = delong_roc_variance(y_true, probs_b)

    # Covariance approximation (independent models)
    se   = np.sqrt(var_a + var_b)
    z    = (auc_a - auc_b) / se if se > 0 else 0
    p    = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, z, p


#  Minimal static model (Exp 01) 
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
        nn.Linear(1280, 256), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    return m


#  Temporal model (Exp 04) 
def build_encoder():
    m    = models.efficientnet_b0(
               weights=models.EfficientNet_B0_Weights.DEFAULT)
    orig = m.features[0][0]
    new  = nn.Conv2d(4, orig.out_channels, orig.kernel_size,
                     orig.stride, orig.padding, bias=False)
    with torch.no_grad():
        new.weight[:, :3] = orig.weight
        new.weight[:, 3]  = orig.weight.mean(dim=1)
    m.features[0][0] = new
    return nn.Sequential(m.features, m.avgpool, nn.Flatten())


class TemporalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = build_encoder()
        self.lstm    = nn.LSTM(1280, 64, 2,
                               batch_first=True, dropout=0.3)
        self.head    = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.3),   nn.Linear(64, 2))

    def forward(self, x):
        b, c, ch, H, W = x.shape
        f = self.encoder(x.view(b*c, ch, H, W))
        f = f.view(b, c, -1)
        _, (hn, _) = self.lstm(f)
        return self.head(hn[-1])


#  Train one fold, return OOF predictions 
def train_get_probs(model_fn, X_tr, y_tr, X_vl, y_vl,
                    epochs=30, patience=6, batch=16):
    n0 = (y_tr == 0).sum()
    n1 = (y_tr == 1).sum()
    w  = torch.tensor([len(y_tr)/(2*n0),
                        len(y_tr)/(2*n1)],
                       dtype=torch.float32).to(DEVICE)

    tr_ld = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.long)),
        batch_size=batch, shuffle=True)
    vl_ld = DataLoader(
        TensorDataset(torch.tensor(X_vl, dtype=torch.float32),
                      torch.tensor(y_vl, dtype=torch.long)),
        batch_size=batch)

    model  = model_fn().to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=0.001)
    crit   = nn.CrossEntropyLoss(weight=w)
    best   = 0.0
    pat    = 0
    best_s = None

    for ep in range(1, epochs+1):
        model.train()
        for Xb, yb in tr_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(Xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for Xb, yb in vl_ld:
                p = torch.softmax(model(Xb.to(DEVICE)),
                                  dim=1)[:, 1]
                probs.extend(p.cpu().numpy())
                trues.extend(yb.numpy())

        auc = roc_auc_score(trues, probs)
        if auc > best:
            best   = auc
            pat    = 0
            best_s = {k: v.clone()
                      for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= patience:
                break

    model.load_state_dict(best_s)
    model.eval()
    final_p = []
    with torch.no_grad():
        for Xb, _ in vl_ld:
            p = torch.softmax(model(Xb.to(DEVICE)), dim=1)[:, 1]
            final_p.extend(p.cpu().numpy())
    return np.array(final_p)


#  Main 
def main():
    print("Loading data...")
    # Static dataset (Exp 01) - full window scalograms
    X_static = np.load(
        DATA_DIR / "scalograms/X.npy")          # (498, 4, 64, 64)
    X_temp   = np.load(
        DATA_DIR / "scalograms_temporal/X_seq.npy") # (498,6,4,64,64)
    y        = np.load(
        DATA_DIR / "scalograms_temporal/y_seq.npy")

    print(f"Static X: {X_static.shape}")
    print(f"Temporal X: {X_temp.shape}")
    print(f"y: {y.shape}\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True,
                          random_state=SEED)

    all_probs_static = np.zeros(len(y))
    all_probs_temp   = np.zeros(len(y))
    all_y            = np.zeros(len(y))

    print("Running matched 5-fold CV for DeLong test...")
    print("(This trains both models on identical splits)\n")

    for fold, (tr_idx, vl_idx) in enumerate(
            skf.split(X_static, y), 1):

        print(f"Fold {fold}/5...")

        # Static model
        # Need to add batch dim for static: (N,4,64,64)→(N,1,4,64,64)
        # Actually static takes (N,4,64,64) directly
        probs_s = train_get_probs(
            build_static_model,
            X_static[tr_idx], y[tr_idx],
            X_static[vl_idx], y[vl_idx])

        # Temporal model
        probs_t = train_get_probs(
            TemporalModel,
            X_temp[tr_idx], y[tr_idx],
            X_temp[vl_idx], y[vl_idx])

        all_probs_static[vl_idx] = probs_s
        all_probs_temp[vl_idx]   = probs_t
        all_y[vl_idx]            = y[vl_idx]

        auc_s = roc_auc_score(y[vl_idx], probs_s)
        auc_t = roc_auc_score(y[vl_idx], probs_t)
        print(f"  Static AUC: {auc_s:.4f}  "
              f"Temporal AUC: {auc_t:.4f}")

    # DeLong test on pooled OOF predictions
    print("\n" + "="*50)
    print("DELONG TEST RESULTS")
    print("="*50)
    auc_s, auc_t, z, p = delong_compare(
        all_y.astype(int),
        all_probs_static,
        all_probs_temp)

    print(f"  Static AUC (Exp 01):   {auc_s:.4f}")
    print(f"  Temporal AUC (Exp 04): {auc_t:.4f}")
    print(f"  Difference:            {auc_t - auc_s:+.4f}")
    print(f"  Z-statistic:           {z:.4f}")
    print(f"  P-value:               {p:.6f}")
    print(f"  Significant (p<0.05):  {p < 0.05}")
    print("="*50)

    # Bootstrap CI on the difference
    print("\nBootstrap CI on AUC difference (1000 iterations)...")
    diffs = []
    rng   = np.random.default_rng(SEED)
    n     = len(all_y)
    for _ in range(1000):
        idx    = rng.integers(0, n, size=n)
        auc_a  = roc_auc_score(all_y[idx].astype(int),
                                all_probs_static[idx])
        auc_b  = roc_auc_score(all_y[idx].astype(int),
                                all_probs_temp[idx])
        diffs.append(auc_b - auc_a)

    diffs  = np.array(diffs)
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    print(f"  Bootstrap 95% CI on difference: "
          f"[{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Excludes zero: {ci_low > 0}")

    # Save
    results = {
        "static_auc":   round(float(auc_s), 4),
        "temporal_auc": round(float(auc_t), 4),
        "difference":   round(float(auc_t - auc_s), 4),
        "z_statistic":  round(float(z), 4),
        "p_value":      round(float(p), 6),
        "significant":  bool(p < 0.05),
        "bootstrap_ci_low":  round(float(ci_low), 4),
        "bootstrap_ci_high": round(float(ci_high), 4),
        "bootstrap_excludes_zero": bool(ci_low > 0),
    }
    with open("results/stats_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved → results/stats_test.json")


if __name__ == "__main__":
    main()