"""
SigmaMedStat - Experiment 05: Temporal CNN + BiLSTM
Architecture:
  EfficientNet-B0 encoder (shared across chunks)
  → Bidirectional LSTM (reads sequence forward + backward)
  → Classifier head
  → false/true alarm probability

Bidirectional LSTM effectively doubles the hidden state:
  forward pass  → captures signal buildup toward alarm
  backward pass → captures signal context before buildup

Structured one-parameter-at-a-time sweep:
  hidden:   32, 64, 128, 256
  dropout:  0.2, 0.3, 0.4, 0.5
  lr:       0.01, 0.001, 0.0001, 0.00001
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import json

#  Paths 
DATA_DIR   = Path("C:/Users/Arun/Documents/git/crip-x/backend/data/scalograms_temporal")
OUTPUT_DIR = Path("results/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

#  Fixed config 
FEAT_DIM    = 1280
LSTM_LAYERS = 2
EPOCHS      = 30
BATCH_SIZE  = 16
PATIENCE    = 6
SEED        = 42

#  Sweep grid 
HIDDEN_VALUES  = [32, 64, 128, 256]
DROPOUT_VALUES = [0.2, 0.3, 0.4, 0.5]
LR_VALUES      = [0.01, 0.001, 0.0001, 0.00001]

DEFAULT_HIDDEN  = 64
DEFAULT_DROPOUT = 0.3
DEFAULT_LR      = 0.001

torch.manual_seed(SEED)
np.random.seed(SEED)


#  Encoder 
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


#  BiLSTM Model 
class BiLSTMClassifier(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM,
                 lstm_hidden=DEFAULT_HIDDEN,
                 lstm_layers=LSTM_LAYERS,
                 dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.encoder = build_encoder()
        self.lstm = nn.LSTM(
            input_size  = feat_dim,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0,
            bidirectional = True,          # ← key difference
        )
        # BiLSTM output is 2× hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        batch, chunks, C, H, W = x.shape
        feats = self.encoder(x.view(batch * chunks, C, H, W))
        feats = feats.view(batch, chunks, -1)
        _, (hn, _) = self.lstm(feats)
        # hn shape: (num_layers * 2, batch, hidden)
        # take last layer forward + backward
        forward_h  = hn[-2]   # last layer forward
        backward_h = hn[-1]   # last layer backward
        combined   = torch.cat([forward_h, backward_h], dim=1)
        return self.classifier(combined)


#  Single run 
def run(X_tr, y_tr, X_vl, y_vl,
        lstm_hidden, dropout, lr, label=""):
    n0 = (y_tr.numpy() == 0).sum()
    n1 = (y_tr.numpy() == 1).sum()
    w  = torch.tensor([len(y_tr)/(2*n0),
                        len(y_tr)/(2*n1)],
                       dtype=torch.float32).to(DEVICE)

    tr_ld = DataLoader(TensorDataset(X_tr, y_tr),
                       batch_size=BATCH_SIZE, shuffle=True)
    vl_ld = DataLoader(TensorDataset(X_vl, y_vl),
                       batch_size=BATCH_SIZE)

    model  = BiLSTMClassifier(lstm_hidden=lstm_hidden,
                               dropout=dropout).to(DEVICE)
    crit   = nn.CrossEntropyLoss(weight=w)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)

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


#  Main sweep 
def main():
    print("SigmaMedStat - Experiment 05: BiLSTM Sweep")
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
    X_te = torch.tensor(X[idx_test],  dtype=torch.float32)
    y_te = torch.tensor(y[idx_test],  dtype=torch.long)

    print(f"Split: {len(idx_train)} train / "
          f"{len(idx_val)} val / {len(idx_test)} test\n")

    sweep_results = {}

    # Sweep 1: hidden
    print("=" * 55)
    print("SWEEP 1 - BiLSTM hidden size")
    print(f"  Fixed: dropout={DEFAULT_DROPOUT} lr={DEFAULT_LR}")
    print("=" * 55)
    hidden_res = {}
    for h in HIDDEN_VALUES:
        auc, _ = run(X_tr, y_tr, X_vl, y_vl,
                     lstm_hidden=h,
                     dropout=DEFAULT_DROPOUT,
                     lr=DEFAULT_LR,
                     label=f"hidden={h}")
        hidden_res[h] = auc
    best_hidden = max(hidden_res, key=lambda k: hidden_res[k])
    print(f"\n  → Best hidden: {best_hidden} "
          f"(val AUC {hidden_res[best_hidden]:.4f})\n")
    sweep_results["hidden"] = hidden_res

    # Sweep 2: dropout
    print("=" * 55)
    print("SWEEP 2 - Dropout rate")
    print(f"  Fixed: hidden={best_hidden} lr={DEFAULT_LR}")
    print("=" * 55)
    dropout_res = {}
    for d in DROPOUT_VALUES:
        auc, _ = run(X_tr, y_tr, X_vl, y_vl,
                     lstm_hidden=best_hidden,
                     dropout=d,
                     lr=DEFAULT_LR,
                     label=f"dropout={d}")
        dropout_res[d] = auc
    best_dropout = max(dropout_res,
                       key=lambda k: dropout_res[k])
    print(f"\n  → Best dropout: {best_dropout} "
          f"(val AUC {dropout_res[best_dropout]:.4f})\n")
    sweep_results["dropout"] = dropout_res

    # Sweep 3: lr
    print("=" * 55)
    print("SWEEP 3 - Learning rate")
    print(f"  Fixed: hidden={best_hidden} "
          f"dropout={best_dropout}")
    print("=" * 55)
    lr_res = {}
    for lr in LR_VALUES:
        auc, _ = run(X_tr, y_tr, X_vl, y_vl,
                     lstm_hidden=best_hidden,
                     dropout=best_dropout,
                     lr=lr,
                     label=f"lr={lr}")
        lr_res[lr] = auc
    best_lr = max(lr_res, key=lambda k: lr_res[k])
    print(f"\n  → Best lr: {best_lr} "
          f"(val AUC {lr_res[best_lr]:.4f})\n")
    sweep_results["lr"] = lr_res

    # Final run + 5-fold CV
    print("=" * 55)
    print("FINAL - 5-fold CV with best config")
    print(f"  hidden={best_hidden} dropout={best_dropout} "
          f"lr={best_lr}")
    print("=" * 55)

    global EPOCHS, PATIENCE
    EPOCHS  = 40
    PATIENCE = 8

    skf       = StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=SEED)
    fold_aucs = []
    all_probs = np.zeros(len(y))
    all_preds = np.zeros(len(y))

    for fold, (tr_idx, vl_idx) in enumerate(
            skf.split(X, y), 1):
        X_f_tr = torch.tensor(X[tr_idx], dtype=torch.float32)
        y_f_tr = torch.tensor(y[tr_idx], dtype=torch.long)
        X_f_vl = torch.tensor(X[vl_idx], dtype=torch.float32)
        y_f_vl = torch.tensor(y[vl_idx], dtype=torch.long)

        auc, best_st = run(X_f_tr, y_f_tr,
                           X_f_vl, y_f_vl,
                           lstm_hidden=best_hidden,
                           dropout=best_dropout,
                           lr=best_lr,
                           label=f"Fold {fold}")
        fold_aucs.append(auc)

        # OOF predictions
        model = BiLSTMClassifier(
            lstm_hidden=best_hidden,
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

    # Challenge metric on OOF
    from challenge_metric import compute_at_threshold
    print("\nChallenge metric sweep:")
    best_ch = 0
    best_ch_thr = 0
    for thr in np.arange(0.05, 0.95, 0.05):
        r = compute_at_threshold(y, all_probs, thr)
        if r["score"] > best_ch:
            best_ch     = r["score"]
            best_ch_thr = thr
    print(f"  Best challenge score: {best_ch:.2f} "
          f"at threshold {best_ch_thr:.2f}")
    print(f"  Exp 04 (LSTM):        58.41")
    diff = best_ch - 58.41
    print(f"  Difference:           {diff:+.2f}")

    # Save
    results = {
        "experiment":   "05_bilstm",
        "architecture": "EfficientNet-B0 + BiLSTM",
        "best_config": {
            "lstm_hidden": best_hidden,
            "dropout":     best_dropout,
            "lr":          best_lr,
            "lstm_layers": LSTM_LAYERS,
            "bidirectional": True,
        },
        "sweep":          sweep_results,
        "fold_aucs":      [round(a, 4) for a in fold_aucs],
        "mean_auc":       round(float(mean_auc), 4),
        "std_auc":        round(float(std_auc), 4),
        "ci_95_low":      round(float(ci_low), 4),
        "ci_95_high":     round(float(ci_high), 4),
        "best_challenge_score": round(float(best_ch), 2),
        "best_challenge_threshold": round(float(best_ch_thr), 2),
        "oof_probs":      all_probs.tolist(),
        "oof_labels":     [int(i) for i in y],
    }

    out = Path("results/experiment_05_bilstm.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved → {out}")
    print("\n" + "=" * 55)
    print("EXPERIMENT 05 SUMMARY")
    print("=" * 55)
    print(f"  Architecture : EfficientNet-B0 + BiLSTM")
    print(f"  Best config  : hidden={best_hidden} "
          f"dropout={best_dropout} lr={best_lr}")
    print(f"  Mean AUC     : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Challenge    : {best_ch:.2f}")
    print(f"\nComparison:")
    print(f"  Exp 04 LSTM    : AUC 0.822  Challenge 58.41")
    print(f"  Exp 05 BiLSTM  : AUC {mean_auc:.3f}  "
          f"Challenge {best_ch:.2f}")
    print("=" * 55)


if __name__ == "__main__":
    main()