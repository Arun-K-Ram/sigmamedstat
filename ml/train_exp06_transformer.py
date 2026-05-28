"""
SigmaMedStat - Experiment 06: Temporal CNN + Transformer
Architecture:
  EfficientNet-B0 encoder (shared across chunks)
  → Positional encoding
  → Transformer encoder (self-attention over 6 chunks)
  → Classifier head

Transformer self-attention lets each chunk attend to
all other chunks simultaneously - unlike LSTM which
reads sequentially. May capture non-sequential
temporal relationships better.

Structured one-parameter-at-a-time sweep:
  n_heads:   2, 4, 8
  n_layers:  1, 2, 3
  dropout:   0.2, 0.3, 0.4
  lr:        0.01, 0.001, 0.0001, 0.00001
"""

import torch
import torch.nn as nn
import numpy as np
import math
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import json

# Paths
DATA_DIR   = Path("C:/Users/Arun/Documents/git/crip-x/backend/data/scalograms_temporal")
OUTPUT_DIR = Path("results/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Fixed config
FEAT_DIM   = 1280
EPOCHS     = 30
BATCH_SIZE = 16
PATIENCE   = 6
SEED       = 42

# Sweep grid
HEADS_VALUES   = [2, 4, 8]
LAYERS_VALUES  = [1, 2, 3]
DROPOUT_VALUES = [0.2, 0.3, 0.4]
LR_VALUES      = [0.01, 0.001, 0.0001, 0.00001]

DEFAULT_HEADS   = 4
DEFAULT_LAYERS  = 2
DEFAULT_DROPOUT = 0.3
DEFAULT_LR      = 0.001

torch.manual_seed(SEED)
np.random.seed(SEED)


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# Transformer Model
class TransformerClassifier(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM,
                 n_heads=DEFAULT_HEADS,
                 n_layers=DEFAULT_LAYERS,
                 dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.encoder = self._build_encoder()

        # Project to smaller dim for attention efficiency
        self.proj = nn.Linear(feat_dim, 256)

        self.pos_enc = PositionalEncoding(
            256, max_len=10, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model    = 256,
            nhead      = n_heads,
            dim_feedforward = 512,
            dropout    = dropout,
            batch_first = True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def _build_encoder(self):
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

    def forward(self, x):
        batch, chunks, C, H, W = x.shape
        feats = self.encoder(x.view(batch * chunks, C, H, W))
        feats = feats.view(batch, chunks, -1)   # (B, 6, 1280)

        # Project + positional encoding
        feats = self.proj(feats)                # (B, 6, 256)
        feats = self.pos_enc(feats)

        # Transformer
        out   = self.transformer(feats)         # (B, 6, 256)

        # Global average pool over chunks
        out   = out.mean(dim=1)                 # (B, 256)
        return self.classifier(out)


# Single run
def run(X_tr, y_tr, X_vl, y_vl,
        n_heads, n_layers, dropout, lr, label=""):
    n0 = (y_tr.numpy() == 0).sum()
    n1 = (y_tr.numpy() == 1).sum()
    w  = torch.tensor([len(y_tr)/(2*n0),
                        len(y_tr)/(2*n1)],
                       dtype=torch.float32).to(DEVICE)

    tr_ld = DataLoader(TensorDataset(X_tr, y_tr),
                       batch_size=BATCH_SIZE, shuffle=True)
    vl_ld = DataLoader(TensorDataset(X_vl, y_vl),
                       batch_size=BATCH_SIZE)

    model = TransformerClassifier(
        n_heads=n_heads, n_layers=n_layers,
        dropout=dropout).to(DEVICE)
    crit  = nn.CrossEntropyLoss(weight=w)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc = 0.0
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

    print(f"  {label:<40} val_auc={best_auc:.4f}")
    return best_auc, best_st


# Main sweep 
def main():
    print("SigmaMedStat - Experiment 06: Transformer Sweep")
    print(f"Device: {DEVICE}\n")

    X = np.load(DATA_DIR / "X_seq.npy")
    y = np.load(DATA_DIR / "y_seq.npy")

    idx = np.arange(len(X))
    idx_tv, idx_test = train_test_split(
        idx, test_size=0.15, random_state=SEED, stratify=y)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=0.15/0.85,
        random_state=SEED, stratify=y[idx_tv])

    X_tr = torch.tensor(X[idx_train], dtype=torch.float32)
    y_tr = torch.tensor(y[idx_train], dtype=torch.long)
    X_vl = torch.tensor(X[idx_val],   dtype=torch.float32)
    y_vl = torch.tensor(y[idx_val],   dtype=torch.long)

    print(f"Split: {len(idx_train)} train / "
          f"{len(idx_val)} val / {len(idx_test)} test\n")

    sweep_results = {}

    # Sweep 1: attention heads
    print("=" * 55)
    print("SWEEP 1 - Number of attention heads")
    print(f"  Fixed: layers={DEFAULT_LAYERS} "
          f"dropout={DEFAULT_DROPOUT} lr={DEFAULT_LR}")
    print("=" * 55)
    heads_res = {}
    for h in HEADS_VALUES:
        auc, _ = run(X_tr, y_tr, X_vl, y_vl,
                     n_heads=h,
                     n_layers=DEFAULT_LAYERS,
                     dropout=DEFAULT_DROPOUT,
                     lr=DEFAULT_LR,
                     label=f"heads={h}")
        heads_res[h] = auc
    best_heads = max(heads_res, key=lambda k: heads_res[k])
    print(f"\n  → Best heads: {best_heads} "
          f"(val AUC {heads_res[best_heads]:.4f})\n")
    sweep_results["heads"] = heads_res

    # Sweep 2: transformer layers
    print("=" * 55)
    print("SWEEP 2 - Number of transformer layers")
    print(f"  Fixed: heads={best_heads} "
          f"dropout={DEFAULT_DROPOUT} lr={DEFAULT_LR}")
    print("=" * 55)
    layers_res = {}
    for l in LAYERS_VALUES:
        auc, _ = run(X_tr, y_tr, X_vl, y_vl,
                     n_heads=best_heads,
                     n_layers=l,
                     dropout=DEFAULT_DROPOUT,
                     lr=DEFAULT_LR,
                     label=f"layers={l}")
        layers_res[l] = auc
    best_layers = max(layers_res, key=lambda k: layers_res[k])
    print(f"\n  → Best layers: {best_layers} "
          f"(val AUC {layers_res[best_layers]:.4f})\n")
    sweep_results["layers"] = layers_res

    # Sweep 3: dropout
    print("=" * 55)
    print("SWEEP 3 - Dropout rate")
    print(f"  Fixed: heads={best_heads} layers={best_layers} "
          f"lr={DEFAULT_LR}")
    print("=" * 55)
    dropout_res = {}
    for d in DROPOUT_VALUES:
        auc, _ = run(X_tr, y_tr, X_vl, y_vl,
                     n_heads=best_heads,
                     n_layers=best_layers,
                     dropout=d,
                     lr=DEFAULT_LR,
                     label=f"dropout={d}")
        dropout_res[d] = auc
    best_dropout = max(dropout_res,
                       key=lambda k: dropout_res[k])
    print(f"\n  → Best dropout: {best_dropout} "
          f"(val AUC {dropout_res[best_dropout]:.4f})\n")
    sweep_results["dropout"] = dropout_res

    # Sweep 4: lr
    print("=" * 55)
    print("SWEEP 4 - Learning rate")
    print(f"  Fixed: heads={best_heads} layers={best_layers} "
          f"dropout={best_dropout}")
    print("=" * 55)
    lr_res = {}
    for lr in LR_VALUES:
        auc, _ = run(X_tr, y_tr, X_vl, y_vl,
                     n_heads=best_heads,
                     n_layers=best_layers,
                     dropout=best_dropout,
                     lr=lr,
                     label=f"lr={lr}")
        lr_res[lr] = auc
    best_lr = max(lr_res, key=lambda k: lr_res[k])
    print(f"\n  → Best lr: {best_lr} "
          f"(val AUC {lr_res[best_lr]:.4f})\n")
    sweep_results["lr"] = lr_res

    # Final 5-fold CV
    print("=" * 55)
    print("FINAL - 5-fold CV with best config")
    print(f"  heads={best_heads} layers={best_layers} "
          f"dropout={best_dropout} lr={best_lr}")
    print("=" * 55)

    global EPOCHS, PATIENCE
    EPOCHS  = 40
    PATIENCE = 8

    skf       = StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=SEED)
    fold_aucs = []
    all_probs = np.zeros(len(y))

    for fold, (tr_idx, vl_idx) in enumerate(
            skf.split(X, y), 1):
        X_f_tr = torch.tensor(X[tr_idx], dtype=torch.float32)
        y_f_tr = torch.tensor(y[tr_idx], dtype=torch.long)
        X_f_vl = torch.tensor(X[vl_idx], dtype=torch.float32)
        y_f_vl = torch.tensor(y[vl_idx], dtype=torch.long)

        auc, best_st = run(X_f_tr, y_f_tr,
                           X_f_vl, y_f_vl,
                           n_heads=best_heads,
                           n_layers=best_layers,
                           dropout=best_dropout,
                           lr=best_lr,
                           label=f"Fold {fold}")
        fold_aucs.append(auc)

        model = TransformerClassifier(
            n_heads=best_heads, n_layers=best_layers,
            dropout=best_dropout).to(DEVICE)
        model.load_state_dict(best_st)
        model.eval()
        vl_ld = DataLoader(
            TensorDataset(X_f_vl, y_f_vl),
            batch_size=BATCH_SIZE)
        probs = []
        with torch.no_grad():
            for Xb, _ in vl_ld:
                p = torch.softmax(
                    model(Xb.to(DEVICE)), dim=1)[:, 1]
                probs.extend(p.cpu().numpy())
        all_probs[vl_idx] = probs

    mean_auc = np.mean(fold_aucs)
    std_auc  = np.std(fold_aucs)
    ci_low   = mean_auc - 1.96 * std_auc
    ci_high  = mean_auc + 1.96 * std_auc

    print(f"\nFold AUCs: {[round(a,4) for a in fold_aucs]}")
    print(f"Mean AUC:  {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"95% CI:    [{ci_low:.4f}, {ci_high:.4f}]")

    # Challenge metric
    from challenge_metric import compute_at_threshold
    best_ch = 0
    best_ch_thr = 0
    for thr in np.arange(0.05, 0.95, 0.05):
        r = compute_at_threshold(y, all_probs, thr)
        if r["score"] > best_ch:
            best_ch     = r["score"]
            best_ch_thr = thr
    print(f"\nBest challenge score: {best_ch:.2f} "
          f"at threshold {best_ch_thr:.2f}")

    # Save
    results = {
        "experiment":   "06_transformer",
        "architecture": "EfficientNet-B0 + Transformer",
        "best_config": {
            "n_heads":   best_heads,
            "n_layers":  best_layers,
            "dropout":   best_dropout,
            "lr":        best_lr,
            "proj_dim":  256,
        },
        "sweep":         sweep_results,
        "fold_aucs":     [round(a, 4) for a in fold_aucs],
        "mean_auc":      round(float(mean_auc), 4),
        "std_auc":       round(float(std_auc), 4),
        "ci_95_low":     round(float(ci_low), 4),
        "ci_95_high":    round(float(ci_high), 4),
        "best_challenge_score":     round(float(best_ch), 2),
        "best_challenge_threshold": round(float(best_ch_thr), 2),
        "oof_probs":     all_probs.tolist(),
        "oof_labels":    [int(i) for i in y],
    }

    out = Path("results/experiment_06_transformer.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved → {out}")
    print("\n" + "=" * 55)
    print("EXPERIMENT 06 SUMMARY")
    print("=" * 55)
    print(f"  Architecture : EfficientNet-B0 + Transformer")
    print(f"  Best config  : heads={best_heads} "
          f"layers={best_layers} dropout={best_dropout} "
          f"lr={best_lr}")
    print(f"  Mean AUC     : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Challenge    : {best_ch:.2f}")
    print(f"\nFull comparison:")
    print(f"  Exp 04 LSTM        : AUC 0.822  Challenge 58.41")
    print(f"  Exp 05 BiLSTM      : see exp05 results")
    print(f"  Exp 06 Transformer : AUC {mean_auc:.3f}  "
          f"Challenge {best_ch:.2f}")
    print("=" * 55)


if __name__ == "__main__":
    main()