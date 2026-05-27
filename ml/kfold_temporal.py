"""
SigmaMedStat - Experiment 04: K-Fold Cross Validation
Validates the LSTM result is statistically robust, not a lucky split.

Uses 5-fold stratified cross-validation on the full 498-record dataset.
Reports mean AUC, std deviation, and per-fold results.
Best config from sweep: hidden=64, dropout=0.3, lr=0.001
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

#  Paths 
DATA_DIR   = Path("C:/Users/Arun/Documents/git/crip-x/backend/data/scalograms_temporal")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

#  Best config from sweep 
LSTM_HIDDEN = 64
DROPOUT     = 0.3
LR          = 0.001
LSTM_LAYERS = 2
FEAT_DIM    = 1280
EPOCHS      = 40
PATIENCE    = 8
BATCH_SIZE  = 16
N_FOLDS     = 5
SEED        = 42

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
    def __init__(self):
        super().__init__()
        self.encoder = build_encoder()
        self.lstm = nn.LSTM(
            input_size  = FEAT_DIM,
            hidden_size = LSTM_HIDDEN,
            num_layers  = LSTM_LAYERS,
            batch_first = True,
            dropout     = DROPOUT,
        )
        self.classifier = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        batch, chunks, C, H, W = x.shape
        feats = self.encoder(x.view(batch * chunks, C, H, W))
        feats = feats.view(batch, chunks, -1)
        _, (hn, _) = self.lstm(feats)
        return self.classifier(hn[-1])


#  Train one fold 
def train_fold(X_train, y_train, X_val, y_val, fold_num):
    # Class weights
    n_false = (y_train == 0).sum()
    n_true  = (y_train == 1).sum()
    w_false = len(y_train) / (2 * n_false)
    w_true  = len(y_train) / (2 * n_true)
    weights = torch.tensor([w_false, w_true], dtype=torch.float32).to(DEVICE)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_vl = torch.tensor(X_val,   dtype=torch.float32)
    y_vl = torch.tensor(y_val,   dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_vl, y_vl),
                              batch_size=BATCH_SIZE)

    model     = TemporalAlarmClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc       = 0.0
    best_epoch     = 0
    patience_count = 0
    best_state     = None

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                p = torch.softmax(model(Xb.to(DEVICE)), dim=1)[:, 1]
                probs.extend(p.cpu().numpy())
                trues.extend(yb.numpy())

        auc = roc_auc_score(trues, probs)

        if auc > best_auc:
            best_auc       = auc
            best_epoch     = epoch
            patience_count = 0
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    print(f"  Fold {fold_num}  val_auc={best_auc:.4f}  best_epoch={best_epoch}")
    return best_auc, best_epoch


#  Error analysis 
def error_analysis(X, y, names, skf):
    """
    Run full error analysis on all folds combined.
    Tracks per-alarm-type performance and failure cases.
    """
    print("\nRunning error analysis...")

    alarm_types = {
        "v": "Ventricular Flutter",
        "a": "Asystole",
        "b": "Bradycardia",
        "t": "Tachycardia",
        "f": "Ventricular Fibrillation",
    }

    all_preds  = np.zeros(len(y))
    all_probs  = np.zeros(len(y))
    all_tested = np.zeros(len(y), dtype=bool)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        # Train
        n_false = (y_tr == 0).sum()
        n_true  = (y_tr == 1).sum()
        w_false = len(y_tr) / (2 * n_false)
        w_true  = len(y_tr) / (2 * n_true)
        weights = torch.tensor([w_false, w_true], dtype=torch.float32).to(DEVICE)

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        X_vl_t = torch.tensor(X_vl, dtype=torch.float32)

        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                            batch_size=BATCH_SIZE, shuffle=True)

        model     = TemporalAlarmClassifier().to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_auc   = 0
        patience_c = 0
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            model.train()
            for Xb, yb in loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            probs_e, trues_e = [], []
            val_loader_e = DataLoader(
                TensorDataset(X_vl_t, torch.tensor(y_vl, dtype=torch.long)),
                batch_size=BATCH_SIZE)
            with torch.no_grad():
                for Xb, yb in val_loader_e:
                    p = torch.softmax(model(Xb.to(DEVICE)), dim=1)[:, 1]
                    probs_e.extend(p.cpu().numpy())
                    trues_e.extend(yb.numpy())

            auc = roc_auc_score(trues_e, probs_e)
            if auc > best_auc:
                best_auc   = auc
                patience_c = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_c += 1
                if patience_c >= PATIENCE:
                    break

        # Final predictions on val set
        model.load_state_dict(best_state)
        model.eval()
        final_probs = []
        with torch.no_grad():
            for Xb, _ in DataLoader(TensorDataset(X_vl_t,
                                    torch.zeros(len(X_vl_t))),
                                    batch_size=BATCH_SIZE):
                p = torch.softmax(model(Xb.to(DEVICE)), dim=1)[:, 1]
                final_probs.extend(p.cpu().numpy())

        all_probs[val_idx]  = final_probs
        all_preds[val_idx]  = (np.array(final_probs) > 0.5).astype(int)
        all_tested[val_idx] = True

    # Per-alarm-type analysis
    print("\n Per-alarm-type AUC ")
    type_results = {}
    for code, name in alarm_types.items():
        mask = np.array([n[0] == code for n in names])
        if mask.sum() < 5:
            continue
        type_auc = roc_auc_score(y[mask], all_probs[mask])
        n_correct = ((all_preds[mask] == y[mask])).sum()
        type_results[name] = {
            "auc":       round(float(type_auc), 4),
            "n_records": int(mask.sum()),
            "n_correct": int(n_correct),
            "accuracy":  round(float(n_correct / mask.sum()), 4),
        }
        print(f"  {name:<30} n={mask.sum():>3}  "
              f"AUC={type_auc:.4f}  "
              f"acc={n_correct/mask.sum():.2%}")

    # Overall
    overall_auc = roc_auc_score(y, all_probs)
    correct     = (all_preds == y).sum()
    print(f"\n  Overall  n={len(y)}  AUC={overall_auc:.4f}  "
          f"acc={correct/len(y):.2%}")

    # Failure cases
    print("\n Failure cases ")
    errors = np.where(all_preds != y)[0]
    false_neg = np.where((all_preds == 0) & (y == 1))[0]  # missed real alarms
    false_pos = np.where((all_preds == 1) & (y == 0))[0]  # false alarm missed

    print(f"  Total errors:          {len(errors)}")
    print(f"  False negatives:       {len(false_neg)}  "
          f"(real alarm called safe - DANGEROUS)")
    print(f"  False positives:       {len(false_pos)}  "
          f"(false alarm called real - wasteful)")

    # High confidence errors
    high_conf_errors = [i for i in errors if max(all_probs[i], 1-all_probs[i]) > 0.8]
    print(f"  High-confidence errors (>80%): {len(high_conf_errors)}")
    for i in high_conf_errors[:5]:
        atype = alarm_types.get(names[i][0], "Unknown")
        print(f"    {names[i]}  gt={'TRUE' if y[i]==1 else 'FALSE'}"
              f"  pred={'TRUE' if all_preds[i]==1 else 'FALSE'}"
              f"  conf={max(all_probs[i], 1-all_probs[i]):.2%}"
              f"  type={atype}")

    # Class imbalance discussion
    print("\n Class imbalance ")
    print(f"  True alarms:   {y.sum()} ({y.sum()/len(y):.1%})")
    print(f"  False alarms:  {(y==0).sum()} ({(y==0).sum()/len(y):.1%})")
    print(f"  Imbalance ratio: 1:{(y==0).sum()//y.sum()}")
    print(f"  Mitigation: class-weighted loss applied during training")
    print(f"    w_true={len(y)/(2*y.sum()):.3f}  "
          f"w_false={len(y)/(2*(y==0).sum()):.3f}")

    return {
        "overall_auc":    round(float(overall_auc), 4),
        "overall_acc":    round(float(correct/len(y)), 4),
        "total_errors":   int(len(errors)),
        "false_negatives":int(len(false_neg)),
        "false_positives":int(len(false_pos)),
        "high_conf_errors":int(len(high_conf_errors)),
        "per_alarm_type": type_results,
    }


#  Main 
def main():
    print("SigmaMedStat - K-Fold Cross Validation + Error Analysis")
    print(f"Config: hidden={LSTM_HIDDEN}  dropout={DROPOUT}  lr={LR}")
    print(f"Folds: {N_FOLDS}  Epochs: {EPOCHS}  Patience: {PATIENCE}\n")

    # Load data
    X     = np.load(DATA_DIR / "X_seq.npy")
    y     = np.load(DATA_DIR / "y_seq.npy")
    names = list(np.load(DATA_DIR / "names_seq.npy"))
    print(f"Dataset: {X.shape}  "
          f"True={y.sum()}  False={(y==0).sum()}\n")

    # Stratified k-fold
    skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aucs = []

    print("=" * 50)
    print("K-FOLD CROSS VALIDATION")
    print("=" * 50)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]
        auc, ep    = train_fold(X_tr, y_tr, X_vl, y_vl, fold)
        fold_aucs.append(auc)

    mean_auc = np.mean(fold_aucs)
    std_auc  = np.std(fold_aucs)

    print(f"\n{''*50}")
    print(f"  Fold AUCs: {[round(a,4) for a in fold_aucs]}")
    print(f"  Mean AUC:  {mean_auc:.4f}")
    print(f"  Std Dev:   {std_auc:.4f}")
    print(f"  95% CI:    [{mean_auc - 1.96*std_auc:.4f}, "
          f"{mean_auc + 1.96*std_auc:.4f}]")
    print(f"{''*50}\n")

    # Error analysis across all folds
    error_results = error_analysis(X, y, names, skf)

    # Save results
    results = {
        "experiment":   "04_kfold_cv",
        "config": {
            "lstm_hidden": LSTM_HIDDEN,
            "dropout":     DROPOUT,
            "lr":          LR,
            "lstm_layers": LSTM_LAYERS,
            "n_folds":     N_FOLDS,
            "epochs":      EPOCHS,
            "patience":    PATIENCE,
        },
        "fold_aucs":  [round(a, 4) for a in fold_aucs],
        "mean_auc":   round(float(mean_auc), 4),
        "std_auc":    round(float(std_auc),  4),
        "ci_95_low":  round(float(mean_auc - 1.96*std_auc), 4),
        "ci_95_high": round(float(mean_auc + 1.96*std_auc), 4),
        "error_analysis": error_results,
    }

    out = OUTPUT_DIR / "experiment_04_kfold.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out}")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Mean AUC ± Std:  {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  95% CI:          [{mean_auc - 1.96*std_auc:.4f}, "
          f"{mean_auc + 1.96*std_auc:.4f}]")
    print(f"  Total errors:    {error_results['total_errors']}")
    print(f"  False negatives: {error_results['false_negatives']} "
          f"(missed real alarms)")
    print(f"  False positives: {error_results['false_positives']} "
          f"(false alarms called real)")
    print("=" * 50)


if __name__ == "__main__":
    main()