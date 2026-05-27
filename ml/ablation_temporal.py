"""
SigmaMedStat - Ablation Study
Tests two design decisions:

  1. Number of chunks: 1 (static) vs 2 vs 3 vs 6
     Answers: does temporal modeling actually help,
     and is 6 chunks the right number?

  2. Number of channels: 1 (ECG II only) vs 2 (ECG only)
     vs 4 (all channels)
     Answers: does multi-signal fusion actually help?

Each condition runs 3-fold CV for speed.
Best config: hidden=64, dropout=0.3, lr=0.001
"""

import torch
import torch.nn as nn
import numpy as np
import pywt
import wfdb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import json

# ── Paths-──────────────────────────────
DATA_DIR      = Path("C:/Users/Arun/Documents/git/crip-x/backend/data/scalograms_temporal")
PHYSIONET_DIR = Path("C:/Users/Arun/Documents/git/crip-x/backend/data/physionet/training")
OUTPUT_DIR    = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Config-─────────────────────────────
LSTM_HIDDEN = 64
DROPOUT     = 0.3
LR          = 0.001
LSTM_LAYERS = 2
FEAT_DIM    = 1280
EPOCHS      = 30
PATIENCE    = 6
BATCH_SIZE  = 16
N_FOLDS     = 3     # 3-fold for speed
SEED        = 42
SCALE_SIZE  = 64
SCALES      = np.geomspace(1, 128, num=SCALE_SIZE)
WAVELET     = "morl"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── CWT-────────────────────────────────
def cwt_scalogram(signal):
    indices   = np.linspace(0, len(signal)-1, SCALE_SIZE, dtype=int)
    signal_ds = signal[indices]
    coeffs, _ = pywt.cwt(signal_ds, SCALES, WAVELET)
    scalogram = np.abs(coeffs)
    s_min, s_max = scalogram.min(), scalogram.max()
    if s_max > s_min:
        scalogram = (scalogram - s_min) / (s_max - s_min)
    return scalogram.astype(np.float32)


# ── Build dataset for ablation conditions ──────────────────────
def build_ablation_dataset(X_full, n_chunks, n_channels):
    """
    X_full: (N, 6, 4, 64, 64) - full temporal dataset

    n_chunks:   1, 2, 3, or 6
    n_channels: 1, 2, or 4

    Returns X_abl: (N, n_chunks, n_channels, 64, 64)
    """
    N = X_full.shape[0]

    if n_chunks == 6:
        # Use existing data, just slice channels
        X_abl = X_full[:, :, :n_channels, :, :]
    else:
        # Merge 6 chunks into n_chunks by combining adjacent chunks
        # e.g. 6→3: merge pairs; 6→2: merge triples; 6→1: merge all
        chunks_per_new = 6 // n_chunks
        X_abl = np.zeros((N, n_chunks, n_channels, 64, 64), dtype=np.float32)
        for new_i in range(n_chunks):
            start = new_i * chunks_per_new
            end   = start + chunks_per_new
            # Average the scalograms of merged chunks
            X_abl[:, new_i] = X_full[:, start:end, :n_channels].mean(axis=1)

    return X_abl.astype(np.float32)


# ── Model-──────────────────────────────
def build_encoder(n_channels):
    m    = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    orig = m.features[0][0]
    new  = nn.Conv2d(n_channels, orig.out_channels, orig.kernel_size,
                     orig.stride, orig.padding, bias=False)
    with torch.no_grad():
        if n_channels <= 3:
            new.weight[:, :n_channels] = orig.weight[:, :n_channels]
        else:
            new.weight[:, :3] = orig.weight
            new.weight[:, 3]  = orig.weight.mean(dim=1)
    m.features[0][0] = new
    return nn.Sequential(m.features, m.avgpool, nn.Flatten())


class AblationClassifier(nn.Module):
    def __init__(self, n_channels, n_chunks):
        super().__init__()
        self.n_chunks = n_chunks
        self.encoder  = build_encoder(n_channels)

        if n_chunks == 1:
            # Static - no LSTM, just MLP
            self.use_lstm = False
            self.classifier = nn.Sequential(
                nn.Linear(FEAT_DIM, 64),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(64, 2)
            )
        else:
            self.use_lstm = True
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

        if self.use_lstm:
            _, (hn, _) = self.lstm(feats)
            return self.classifier(hn[-1])
        else:
            return self.classifier(feats.squeeze(1))


# ── Train one condition-────────────────
def run_condition(X, y, n_channels, n_chunks, label):
    X_abl = build_ablation_dataset(X, n_chunks, n_channels)
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs  = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_abl, y), 1):
        X_tr, X_vl = X_abl[tr_idx], X_abl[vl_idx]
        y_tr, y_vl = y[tr_idx],     y[vl_idx]

        n_false = (y_tr == 0).sum()
        n_true  = (y_tr == 1).sum()
        weights = torch.tensor([
            len(y_tr)/(2*n_false),
            len(y_tr)/(2*n_true)
        ], dtype=torch.float32).to(DEVICE)

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        X_vl_t = torch.tensor(X_vl, dtype=torch.float32)
        y_vl_t = torch.tensor(y_vl, dtype=torch.long)

        train_ldr = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                               batch_size=BATCH_SIZE, shuffle=True)
        val_ldr   = DataLoader(TensorDataset(X_vl_t, y_vl_t),
                               batch_size=BATCH_SIZE)

        model     = AblationClassifier(n_channels, n_chunks).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_auc   = 0.0
        patience_c = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            for Xb, yb in train_ldr:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            probs, trues = [], []
            with torch.no_grad():
                for Xb, yb in val_ldr:
                    p = torch.softmax(model(Xb.to(DEVICE)), dim=1)[:, 1]
                    probs.extend(p.cpu().numpy())
                    trues.extend(yb.numpy())

            auc = roc_auc_score(trues, probs)
            if auc > best_auc:
                best_auc   = auc
                patience_c = 0
            else:
                patience_c += 1
                if patience_c >= PATIENCE:
                    break

        aucs.append(best_auc)

    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)
    print(f"  {label:<35} mean={mean_auc:.4f} ± {std_auc:.4f}  "
          f"folds={[round(a,4) for a in aucs]}")
    return {"mean_auc": round(float(mean_auc), 4),
            "std_auc":  round(float(std_auc),  4),
            "fold_aucs": [round(a, 4) for a in aucs]}


# Main
def main():
    print("SigmaMedStat - Ablation Study")
    print(f"Config: hidden={LSTM_HIDDEN}  dropout={DROPOUT}  lr={LR}")
    print(f"Folds: {N_FOLDS} (3-fold for speed)\n")

    X = np.load(DATA_DIR / "X_seq.npy")
    y = np.load(DATA_DIR / "y_seq.npy")
    print(f"Dataset: {X.shape}\n")

    results = {}

    # Ablation 1: Number of chunks
    print("=" * 60)
    print("ABLATION 1 - Number of temporal chunks (4 channels fixed)")
    print("=" * 60)
    chunk_results = {}
    for n_chunks in [1, 2, 3, 6]:
        label = f"chunks={n_chunks}" + (" (static, no LSTM)" if n_chunks==1 else "")
        r = run_condition(X, y, n_channels=4, n_chunks=n_chunks, label=label)
        chunk_results[n_chunks] = r
    results["chunks_ablation"] = chunk_results

    print(f"\n  → Best: {max(chunk_results, key=lambda k: chunk_results[k]['mean_auc'])} chunks\n")

    # Ablation 2: Number of channels
    print("=" * 60)
    print("ABLATION 2 - Number of signal channels (6 chunks fixed)")
    print("=" * 60)
    channel_labels = {
        1: "channels=1 (ECG Lead II only)",
        2: "channels=2 (ECG Lead II + V)",
        4: "channels=4 (all: ECG + SpO2 + Resp)",
    }
    channel_results = {}
    for n_ch in [1, 2, 4]:
        r = run_condition(X, y, n_channels=n_ch, n_chunks=6,
                          label=channel_labels[n_ch])
        channel_results[n_ch] = r
    results["channels_ablation"] = channel_results

    print(f"\n  → Best: {max(channel_results, key=lambda k: channel_results[k]['mean_auc'])} channels\n")

    # Save
    out = OUTPUT_DIR / "ablation_study.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out}")

    # Summary table
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    print("\nChunks ablation:")
    for k, v in chunk_results.items():
        bar = "★" if k == max(chunk_results,
                              key=lambda x: chunk_results[x]["mean_auc"]) else " "
        print(f"  {bar} chunks={k:<3}  AUC={v['mean_auc']:.4f} ± {v['std_auc']:.4f}")

    print("\nChannels ablation:")
    for k, v in channel_results.items():
        bar = "★" if k == max(channel_results,
                              key=lambda x: channel_results[x]["mean_auc"]) else " "
        print(f"  {bar} channels={k:<2}  AUC={v['mean_auc']:.4f} ± {v['std_auc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()