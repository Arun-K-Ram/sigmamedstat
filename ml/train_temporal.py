"""
SigmaMedStat - Experiment 04: Temporal CNN + LSTM
Structured hyperparameter sweep - one parameter at a time.

Sweep grid:
  LSTM hidden:  64, 128, 256, 512     (dropout=0.5, lr=1e-4 fixed)
  Dropout:      0.2, 0.3, 0.4, 0.5   (hidden=256,  lr=1e-4 fixed)
  Learning rate: 0.01, 0.001, 0.0001, 0.00001 (hidden=256, dropout=0.5 fixed)

Best config from sweep → final test evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
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
N_CHUNKS    = 6
N_CHANNELS  = 4
FEAT_DIM    = 1280
LSTM_LAYERS = 2
EPOCHS      = 20       # early stopping built in - 20 is enough
BATCH_SIZE  = 16
PATIENCE    = 5        # stop if no improvement for 5 epochs
SEED        = 42

#  Sweep grid 
HIDDEN_VALUES  = [64, 128, 256, 512]
DROPOUT_VALUES = [0.2, 0.3, 0.4, 0.5]
LR_VALUES      = [0.01, 0.001, 0.0001, 0.00001]

# Defaults held fixed while sweeping one parameter
DEFAULT_HIDDEN  = 256
DEFAULT_DROPOUT = 0.5
DEFAULT_LR      = 1e-4

torch.manual_seed(SEED)
np.random.seed(SEED)


#  Model 
def build_encoder():
    m    = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    orig = m.features[0][0]
    new  = nn.Conv2d(4, orig.out_channels, orig.kernel_size,
                     orig.stride, orig.padding, bias=False)
    with torch.no_grad():
        new.weight[:, :3] = orig.weight
        new.weight[:, 3]  = orig.weight.mean(dim=1)
    m.features[0][0] = new
    return nn.Sequential(m.features, m.avgpool, nn.Flatten())


class TemporalAlarmClassifier(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, lstm_hidden=DEFAULT_HIDDEN,
                 lstm_layers=LSTM_LAYERS, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.encoder = build_encoder()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        batch, chunks, C, H, W = x.shape
        x_flat = x.view(batch * chunks, C, H, W)
        feats  = self.encoder(x_flat)
        feats  = feats.view(batch, chunks, -1)
        out, (hn, _) = self.lstm(feats)
        last_hidden  = hn[-1]
        return self.classifier(last_hidden)


#  Single training run 
def run(X_train, y_train, X_val, y_val, idx_train,
        lstm_hidden, dropout, lr, label="", save_history=False):
    """Train one configuration. Returns best val AUC, best epoch, history."""

    n_false = (y_train.numpy() == 0).sum()
    n_true  = (y_train.numpy() == 1).sum()
    w_false = len(y_train) / (2 * n_false)
    w_true  = len(y_train) / (2 * n_true)
    weights = torch.tensor([w_false, w_true], dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=BATCH_SIZE)

    model     = TemporalAlarmClassifier(lstm_hidden=lstm_hidden, dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auc   = 0.0
    best_epoch     = 0
    patience_count = 0
    history        = []

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(Xb)

        # Validate
        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                p = torch.softmax(model(Xb.to(DEVICE)), dim=1)[:, 1]
                probs.extend(p.cpu().numpy())
                trues.extend(yb.numpy())

        val_auc = roc_auc_score(trues, probs)
        avg_loss = total_loss / len(y_train)

        if save_history:
            history.append({"epoch": epoch, "loss": round(avg_loss, 4),
                            "val_auc": round(val_auc, 4)})

        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            best_epoch     = epoch
            patience_count = 0
            torch.save(model.state_dict(),
                       OUTPUT_DIR / "lstm_sweep_best_tmp.pt")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    print(f"  {label:<40} val_auc={best_val_auc:.4f}  best_epoch={best_epoch}")
    return best_val_auc, best_epoch, history


#  Main sweep 
def main():
    # Load data
    print("Loading temporal dataset...")
    X = np.load(DATA_DIR / "X_seq.npy")
    y = np.load(DATA_DIR / "y_seq.npy")
    print(f"  X: {X.shape}  y: {y.shape}\n")

    # Split - same seed as before
    idx = np.arange(len(X))
    idx_tv, idx_test = train_test_split(idx, test_size=0.15,
                                        random_state=SEED, stratify=y)
    idx_train, idx_val = train_test_split(idx_tv, test_size=0.15/0.85,
                                          random_state=SEED, stratify=y[idx_tv])

    X_train = torch.tensor(X[idx_train], dtype=torch.float32)
    y_train = torch.tensor(y[idx_train], dtype=torch.long)
    X_val   = torch.tensor(X[idx_val],   dtype=torch.float32)
    y_val   = torch.tensor(y[idx_val],   dtype=torch.long)
    X_test  = torch.tensor(X[idx_test],  dtype=torch.float32)
    y_test  = torch.tensor(y[idx_test],  dtype=torch.long)

    print(f"Split: {len(idx_train)} train / {len(idx_val)} val / {len(idx_test)} test\n")

    sweep_results = {}

    #  Sweep 1: LSTM hidden size 
    print("=" * 55)
    print("SWEEP 1 - LSTM hidden size")
    print(f"  Fixed: dropout={DEFAULT_DROPOUT}  lr={DEFAULT_LR}")
    print("=" * 55)
    hidden_results = {}
    for h in HIDDEN_VALUES:
        auc, ep, _ = run(X_train, y_train, X_val, y_val, idx_train,
                      lstm_hidden=h, dropout=DEFAULT_DROPOUT, lr=DEFAULT_LR,
                      label=f"hidden={h}")
        hidden_results[h] = {"val_auc": auc, "best_epoch": ep}

    best_hidden = max(hidden_results, key=lambda k: hidden_results[k]["val_auc"])
    print(f"\n  → Best hidden size: {best_hidden} "
          f"(val AUC {hidden_results[best_hidden]['val_auc']:.4f})\n")
    sweep_results["hidden"] = hidden_results

    #  Sweep 2: Dropout 
    print("=" * 55)
    print("SWEEP 2 - Dropout rate")
    print(f"  Fixed: hidden={best_hidden}  lr={DEFAULT_LR}")
    print("=" * 55)
    dropout_results = {}
    for d in DROPOUT_VALUES:
        auc, ep, _ = run(X_train, y_train, X_val, y_val, idx_train,
                      lstm_hidden=best_hidden, dropout=d, lr=DEFAULT_LR,
                      label=f"dropout={d}")
        dropout_results[d] = {"val_auc": auc, "best_epoch": ep}

    best_dropout = max(dropout_results, key=lambda k: dropout_results[k]["val_auc"])
    print(f"\n  → Best dropout: {best_dropout} "
          f"(val AUC {dropout_results[best_dropout]['val_auc']:.4f})\n")
    sweep_results["dropout"] = dropout_results

    #  Sweep 3: Learning rate 
    print("=" * 55)
    print("SWEEP 3 - Learning rate")
    print(f"  Fixed: hidden={best_hidden}  dropout={best_dropout}")
    print("=" * 55)
    lr_results = {}
    for lr in LR_VALUES:
        auc, ep, _ = run(X_train, y_train, X_val, y_val, idx_train,
                      lstm_hidden=best_hidden, dropout=best_dropout, lr=lr,
                      label=f"lr={lr}")
        lr_results[lr] = {"val_auc": auc, "best_epoch": ep}

    best_lr = max(lr_results, key=lambda k: lr_results[k]["val_auc"])
    print(f"\n  → Best lr: {best_lr} "
          f"(val AUC {lr_results[best_lr]['val_auc']:.4f})\n")
    sweep_results["lr"] = lr_results

    #  Final training with best config 
    print("=" * 55)
    print("FINAL RUN - Best config")
    print(f"  hidden={best_hidden}  dropout={best_dropout}  lr={best_lr}")
    print("=" * 55)

    # Train longer for final run
    global EPOCHS, PATIENCE
    EPOCHS  = 60
    PATIENCE = 10

    auc, ep, history = run(X_train, y_train, X_val, y_val, idx_train,
                           lstm_hidden=best_hidden, dropout=best_dropout,
                           lr=best_lr, label="FINAL", save_history=True)

    # Load best model and evaluate on test set
    final_model = TemporalAlarmClassifier(
        lstm_hidden=best_hidden, dropout=best_dropout
    ).to(DEVICE)
    final_model.load_state_dict(
        torch.load(OUTPUT_DIR / "lstm_sweep_best_tmp.pt",
                   map_location=DEVICE, weights_only=True)
    )
    final_model.eval()

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)
    test_probs, test_true = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            p = torch.softmax(final_model(Xb.to(DEVICE)), dim=1)[:, 1]
            test_probs.extend(p.cpu().numpy())
            test_true.extend(yb.numpy())

    test_auc = roc_auc_score(test_true, test_probs)

    # Save final model
    torch.save(final_model.state_dict(),
               OUTPUT_DIR / "efficientnet_lstm_best.pt")

    # Save all results
    final_results = {
        "experiment":   "04_efficientnet_lstm_sweep",
        "architecture": "EfficientNet-B0 + LSTM",
        "sweep": sweep_results,
        "best_config": {
            "lstm_hidden": best_hidden,
            "dropout":     best_dropout,
            "lr":          best_lr,
            "lstm_layers": LSTM_LAYERS,
        },
        "val_auc":   round(auc, 4),
        "test_auc":  round(test_auc, 4),
        "history":   history,
    }

    out_path = Path("results/experiment_04_lstm.json")
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved → {out_path}")
    print("\n" + "=" * 55)
    print("EXPERIMENT 04 COMPLETE")
    print("=" * 55)
    print(f"  Best config  : hidden={best_hidden}  dropout={best_dropout}  lr={best_lr}")
    print(f"  Val AUC      : {auc:.4f}")
    print(f"  Test AUC     : {test_auc:.4f}")
    print("\nComparison:")
    print(f"  Exp 01 - EfficientNet + Neural Classifier : AUC 0.641")
    print(f"  Exp 04 - EfficientNet + LSTM (tuned)      : AUC {test_auc:.3f}")
    print(f"  Difference: {test_auc - 0.641:+.3f}")
    print("=" * 55)

    #  Plot training curve 
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs   = [h["epoch"]   for h in history]
    val_aucs = [h["val_auc"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#f4f4f2")
    ax.set_facecolor("#f4f4f2")

    # Exp 04 val AUC curve
    ax.plot(epochs, val_aucs, color="#c0392b", linewidth=2,
            label="Exp 04 - EfficientNet + LSTM (val AUC)")

    # Best epoch marker
    best_idx = val_aucs.index(max(val_aucs))
    ax.scatter([epochs[best_idx]], [val_aucs[best_idx]],
               color="#c0392b", s=80, zorder=5,
               label=f"Best: {val_aucs[best_idx]:.3f} at epoch {epochs[best_idx]}")

    # Exp 01 baseline
    ax.axhline(y=0.641, color="#2c3e50", linewidth=1.5,
               linestyle="--", label="Exp 01 - EfficientNet + Neural Classifier (0.641)")

    # Final test AUC
    ax.axhline(y=test_auc, color="#27ae60", linewidth=1.2,
               linestyle=":", label=f"Exp 04 test AUC ({test_auc:.3f})")

    ax.set_xlabel("Epoch", fontsize=12, color="#2c3e50")
    ax.set_ylabel("AUC", fontsize=12, color="#2c3e50")
    ax.set_title("Experiment 04 - Temporal LSTM Training Curve",
                 fontsize=13, color="#2c3e50", fontweight="normal")
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.tick_params(colors="#7f8c8d")
    for spine in ax.spines.values():
        spine.set_edgecolor("#e8e8e5")
    ax.grid(True, alpha=0.3, color="#e8e8e5")

    plot_path = Path("results/plots/experiment_04_training_curve.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#f4f4f2")
    plt.close()
    print(f"\nTraining curve saved → {plot_path}")


if __name__ == "__main__":
    main()