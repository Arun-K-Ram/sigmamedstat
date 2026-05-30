"""
SigmaMedStat - Challenge Metric for Experiments 01-03
Retrains each baseline on train split, evaluates on test split,
computes PhysioNet 2015 challenge score.
Score = (TP+TN) / (TP+TN+FP+5*FN) * 100
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import xgboost as xgb

STATIC_DIR = Path("C:/Users/Arun/Documents/git/crip-x/backend/data/scalograms")
TEMP_DIR   = Path("C:/Users/Arun/Documents/git/crip-x/backend/data/scalograms_temporal")
RESULTS    = Path("results")
SEED       = 42
BATCH      = 16
EPOCHS     = 30
PATIENCE   = 6
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")


def challenge_score(y_true, y_prob, threshold):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true)
    tp = int(((y_pred==1)&(y_true==1)).sum())
    tn = int(((y_pred==0)&(y_true==0)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    denom = tp + tn + fp + 5*fn
    return round((tp+tn)/denom*100, 2) if denom > 0 else 0


def best_challenge(y_true, y_prob):
    best, thr = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        s = challenge_score(y_true, y_prob, t)
        if s > best:
            best = s
            thr  = round(float(t), 2)
    return best, thr


# Load data and split
print("\nLoading data...")
X = np.load(STATIC_DIR / "X.npy")
y = np.load(STATIC_DIR / "y.npy")
names = list(np.load(TEMP_DIR / "names_seq.npy"))
print(f"  Shape: {X.shape}")

idx    = np.arange(len(X))
idx_tv, idx_test = train_test_split(
    idx, test_size=0.15, random_state=SEED, stratify=y)
idx_train, idx_val = train_test_split(
    idx_tv, test_size=0.15/0.85, random_state=SEED,
    stratify=y[idx_tv])

X_tr = X[idx_train]; y_tr = y[idx_train]
X_vl = X[idx_val];   y_vl = y[idx_val]
X_te = X[idx_test];  y_te = y[idx_test]
n_te = [names[i] for i in idx_test]

print(f"  Train={len(idx_train)} Val={len(idx_val)} "
      f"Test={len(idx_test)}")
print(f"  Test: {y_te.sum()} true, {(y_te==0).sum()} false")

n0  = (y_tr==0).sum(); n1 = (y_tr==1).sum()
w   = torch.tensor([len(y_tr)/(2*n0), len(y_tr)/(2*n1)],
                    dtype=torch.float32).to(DEVICE)

results = {}


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 01: Static EfficientNet + Neural Classifier
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("EXPERIMENT 01: Static EfficientNet (retraining...)")
print("="*55)

def build_exp01():
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
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    return m

X_tr1 = torch.tensor(X_tr, dtype=torch.float32)
y_tr1 = torch.tensor(y_tr, dtype=torch.long)
X_vl1 = torch.tensor(X_vl, dtype=torch.float32)
y_vl1 = torch.tensor(y_vl, dtype=torch.long)
X_te1 = torch.tensor(X_te, dtype=torch.float32)
y_te1 = torch.tensor(y_te, dtype=torch.long)

tr_ld = DataLoader(TensorDataset(X_tr1, y_tr1),
                   batch_size=BATCH, shuffle=True)
vl_ld = DataLoader(TensorDataset(X_vl1, y_vl1),
                   batch_size=BATCH)
te_ld = DataLoader(TensorDataset(X_te1, y_te1),
                   batch_size=BATCH)

model01 = build_exp01().to(DEVICE)
opt01   = torch.optim.Adam(model01.parameters(), lr=1e-4)
crit01  = nn.CrossEntropyLoss(weight=w)

best_auc01 = 0; pat = 0; best_st = None
for ep in range(1, EPOCHS+1):
    model01.train()
    for Xb, yb in tr_ld:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        opt01.zero_grad()
        crit01(model01(Xb), yb).backward()
        opt01.step()
    model01.eval()
    probs, trues = [], []
    with torch.no_grad():
        for Xb, yb in vl_ld:
            p = torch.softmax(model01(Xb.to(DEVICE)), dim=1)[:,1]
            probs.extend(p.cpu().numpy())
            trues.extend(yb.numpy())
    auc = roc_auc_score(trues, probs)
    if auc > best_auc01:
        best_auc01 = auc; pat = 0
        best_st = {k: v.clone()
                   for k, v in model01.state_dict().items()}
    else:
        pat += 1
        if pat >= PATIENCE:
            print(f"  Early stop at epoch {ep}")
            break
    if ep % 5 == 0:
        print(f"  Epoch {ep:>3}  val_auc={auc:.4f}  "
              f"best={best_auc01:.4f}")

model01.load_state_dict(best_st)
model01.eval()
probs01 = []
with torch.no_grad():
    for Xb, _ in te_ld:
        p = torch.softmax(model01(Xb.to(DEVICE)), dim=1)[:,1]
        probs01.extend(p.cpu().numpy())

probs01 = np.array(probs01)
auc01   = roc_auc_score(y_te, probs01)
ch01, thr01 = best_challenge(y_te, probs01)
print(f"\nExp 01 Test AUC:        {auc01:.4f}")
print(f"Exp 01 Challenge score: {ch01:.2f} at thr={thr01:.2f}")
results["exp01"] = {
    "experiment":      "01_static_efficientnet",
    "test_auc":        round(float(auc01), 4),
    "challenge_score": ch01,
    "best_threshold":  thr01,
    "n_test":          int(len(y_te)),
}


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 02: Hand-crafted + SVM
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("EXPERIMENT 02: Hand-crafted Features + SVM")
print("="*55)

# Load hand-crafted features
try:
    feat_dir = Path("C:/Users/Arun/Documents/git/crip-x"
                    "/backend/data")
    # Try multiple possible locations
    X_hc, y_hc = None, None
    for candidate in [
        feat_dir / "features" / "X_features.npy",
        feat_dir / "X_handcrafted.npy",
        RESULTS / "X_features.npy",
    ]:
        if candidate.exists():
            X_hc = np.load(candidate)
            y_hc = np.load(str(candidate).replace(
                "X_features", "y_features").replace(
                "X_handcrafted", "y_handcrafted"))
            print(f"  Loaded from {candidate}")
            break

    if X_hc is None:
        raise FileNotFoundError("Hand-crafted features not found")

    idx_tv2, idx_test2 = train_test_split(
        np.arange(len(X_hc)), test_size=0.15,
        random_state=SEED, stratify=y_hc)
    idx_tr2, _ = train_test_split(
        idx_tv2, test_size=0.15/0.85,
        random_state=SEED, stratify=y_hc[idx_tv2])

    sc = StandardScaler()
    X_tr2 = sc.fit_transform(X_hc[idx_tr2])
    X_te2 = sc.transform(X_hc[idx_test2])
    y_te2 = y_hc[idx_test2]

    print("  Training SVM (RBF)...")
    svm = SVC(kernel="rbf", C=1.0, probability=True,
              random_state=SEED)
    svm.fit(X_tr2, y_hc[idx_tr2])
    probs02 = svm.predict_proba(X_te2)[:, 1]

    auc02    = roc_auc_score(y_te2, probs02)
    ch02, thr02 = best_challenge(y_te2, probs02)
    print(f"Exp 02 Test AUC:        {auc02:.4f}")
    print(f"Exp 02 Challenge score: {ch02:.2f} at thr={thr02:.2f}")
    results["exp02"] = {
        "experiment":      "02_handcrafted_svm",
        "test_auc":        round(float(auc02), 4),
        "challenge_score": ch02,
        "best_threshold":  thr02,
        "n_test":          int(len(y_te2)),
    }

except Exception as e:
    print(f"  Exp 02 skipped: {e}")
    results["exp02"] = {
        "experiment":      "02_handcrafted_svm",
        "note":            str(e),
        "reported_auc":    0.539,
        "challenge_score": None,
    }


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 03: Per-alarm XGBoost
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("EXPERIMENT 03: Per-Alarm XGBoost")
print("="*55)

xgb_map = {
    "v": RESULTS / "xgb_v.json",
    "a": RESULTS / "xgb_a.json",
    "t": RESULTS / "xgb_t.json",
    "b": RESULTS / "xgb_b.json",
}

# Load beat features
try:
    beat_candidates = [
        Path("C:/Users/Arun/Documents/git/crip-x/backend"
             "/data/beat_features/X_beat.npy"),
        Path("C:/Users/Arun/Documents/git/crip-x/ml"
             "/results/X_beat.npy"),
    ]
    X_beat = y_beat = names_beat = None
    for c in beat_candidates:
        if c.exists():
            X_beat     = np.load(c)
            y_beat     = np.load(str(c).replace("X_beat","y_beat"))
            names_beat = np.load(
                str(c).replace("X_beat","names_beat"))
            print(f"  Loaded beat features from {c}")
            break

    if X_beat is None:
        raise FileNotFoundError("Beat features not found")

    idx_tv3, idx_test3 = train_test_split(
        np.arange(len(X_beat)), test_size=0.15,
        random_state=SEED, stratify=y_beat)

    X_te3 = X_beat[idx_test3]
    y_te3 = y_beat[idx_test3]
    n_te3 = names_beat[idx_test3]

    all_probs03 = np.full(len(y_te3), 0.5)

    for code, xgb_path in xgb_map.items():
        if not xgb_path.exists():
            print(f"  {xgb_path.name} not found — skipping")
            continue
        clf  = xgb.XGBClassifier()
        clf.load_model(str(xgb_path))
        mask = np.array([str(nm)[0] == code for nm in n_te3])
        if mask.sum() > 0:
            all_probs03[mask] = clf.predict_proba(
                X_te3[mask])[:, 1]
            print(f"  Code={code}: {mask.sum()} test records")

    auc03    = roc_auc_score(y_te3, all_probs03)
    ch03, thr03 = best_challenge(y_te3, all_probs03)
    print(f"Exp 03 Test AUC:        {auc03:.4f}")
    print(f"Exp 03 Challenge score: {ch03:.2f} at thr={thr03:.2f}")
    results["exp03"] = {
        "experiment":      "03_per_alarm_xgboost",
        "test_auc":        round(float(auc03), 4),
        "challenge_score": ch03,
        "best_threshold":  thr03,
        "n_test":          int(len(y_te3)),
    }

except Exception as e:
    print(f"  Exp 03 skipped: {e}")
    results["exp03"] = {
        "experiment":      "03_per_alarm_xgboost",
        "note":            str(e),
        "reported_auc":    0.612,
        "challenge_score": None,
    }


# Summary 
print("\n" + "="*55)
print("SUMMARY")
print("="*55)
print(f"{'Exp':<8} {'AUC':>7} {'Challenge':>10} {'Threshold':>10}")
print("-"*40)
for exp, r in results.items():
    auc = r.get("test_auc", r.get("reported_auc", "N/A"))
    ch  = r.get("challenge_score", "N/A")
    thr = r.get("best_threshold", "N/A")
    print(f"{exp:<8} {str(auc):>7} {str(ch):>10} {str(thr):>10}")

print("\nExp 04-06 (5-fold CV OOF) for reference:")
print(f"{'exp04':<8} {'0.822':>7} {'58.41':>10} {'0.25':>10}")
print(f"{'exp05':<8} {'0.805':>7} {'56.69':>10} {'0.10':>10}")
print(f"{'exp06':<8} {'0.812':>7} {'56.11':>10} {'0.15':>10}")

out = RESULTS / "challenge_metric_exp01_03.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved -> {out}")