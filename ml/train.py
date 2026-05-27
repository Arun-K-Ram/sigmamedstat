"""
SigmaMedStat Training Pipeline
Full hyperparameter sweep across all extractors.

Sweep:
  extractors:  [resnet18, resnet50, efficientnet]
  dropout:     [0.2, 0.3, 0.4, 0.5]
  hidden_dim:  [64, 128, 256, 512]
  lr:          [1e-2, 1e-3, 1e-4, 1e-5]

Picks best extractor + params, trains final model with 150 epochs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from tqdm import tqdm

DATA_DIR   = Path("../backend/data/scalograms")
RESULTS    = Path("results")
MODELS_DIR = RESULTS / "models"
PLOTS_DIR  = RESULTS / "plots"
for d in [RESULTS, MODELS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


#  Feature Extractors 
def build_extractor(name: str) -> nn.Module:
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        orig = m.conv1
        new = nn.Conv2d(4, orig.out_channels, orig.kernel_size,
                        orig.stride, orig.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = orig.weight
            new.weight[:, 3]  = orig.weight.mean(dim=1)
        m.conv1 = new
        extractor = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
        for p in extractor.parameters():
            p.requires_grad = False
        return extractor

    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        orig = m.conv1
        new = nn.Conv2d(4, orig.out_channels, orig.kernel_size,
                        orig.stride, orig.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = orig.weight
            new.weight[:, 3]  = orig.weight.mean(dim=1)
        m.conv1 = new
        extractor = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
        for p in extractor.parameters():
            p.requires_grad = False
        return extractor

    elif name == "efficientnet":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT)
        orig = m.features[0][0]
        new = nn.Conv2d(4, orig.out_channels, orig.kernel_size,
                        orig.stride, orig.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = orig.weight
            new.weight[:, 3]  = orig.weight.mean(dim=1)
        m.features[0][0] = new
        extractor = nn.Sequential(m.features, m.avgpool, nn.Flatten())
        for p in extractor.parameters():
            p.requires_grad = False
        return extractor

    raise ValueError(f"Unknown extractor: {name}")


def extract_features(extractor: nn.Module, X: np.ndarray,
                     batch_size: int = 32) -> np.ndarray:
    extractor.eval().to(DEVICE)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = []
    with torch.no_grad():
        for (batch,) in tqdm(loader, desc="Extracting", leave=False):
            feat = extractor(batch.to(DEVICE))
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


#  Neural Classifier 
class NeuralClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 4, 16)),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(max(hidden_dim // 4, 16), 2)
        )

    def forward(self, x):
        return self.net(x)


def train_neural_classifier(feat_train, y_train, feat_val, y_val,
                             epochs=100, lr=1e-3, dropout=0.5,
                             hidden_dim=256, verbose=True):
    in_dim = feat_train.shape[1]
    model  = NeuralClassifier(in_dim, hidden_dim=hidden_dim,
                               dropout=dropout).to(DEVICE)
    opt    = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit   = nn.CrossEntropyLoss(label_smoothing=0.1)

    X_tr = torch.tensor(feat_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train,    dtype=torch.long)
    X_vl = torch.tensor(feat_val,   dtype=torch.float32)
    y_vl = torch.tensor(y_val,      dtype=torch.long)

    train_ds = TensorDataset(X_tr, y_tr)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    best_auc, best_state = 0, None
    history = {'train_loss': [], 'val_auc': []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss   = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        sched.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(X_vl.to(DEVICE))
            probs_val  = torch.softmax(logits_val, dim=1)[:, 1].cpu().numpy()
        auc = roc_auc_score(y_val, probs_val)

        history['train_loss'].append(epoch_loss / len(train_dl))
        history['val_auc'].append(auc)

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and epoch % 10 == 0:
            print(f"    Epoch {epoch:03d} | Loss {epoch_loss/len(train_dl):.4f}"
                  f" | Val AUC {auc:.3f}")

    model.load_state_dict(best_state)
    return model, best_auc, history


#  Hyperparameter Sweep 
def run_sweep(feat_dict, y_tr, y_vl, y_te):
    print(f"\n{'='*60}")
    print("Full Hyperparameter Sweep - All Extractors")
    print(f"{'='*60}")

    DEFAULT_DROPOUT    = 0.5
    DEFAULT_HIDDEN_DIM = 256
    DEFAULT_LR         = 1e-3
    EPOCHS             = 100

    sweep_space = {
        "dropout":    [0.2, 0.3, 0.4, 0.5],
        "hidden_dim": [64, 128, 256, 512],
        "lr":         [1e-2, 1e-3, 1e-4, 1e-5],
    }

    all_sweep_results = {}
    best_overall = {"test_auc": 0, "extractor": None, "params": {}}

    for ext_name, (feat_tr, feat_vl, feat_te) in feat_dict.items():
        print(f"\n{'='*60}")
        print(f"Sweeping extractor: {ext_name}")
        print(f"{'='*60}")

        sweep_results = {}

        for param_name, values in sweep_space.items():
            print(f"\n  Sweeping {param_name} over {values}")
            param_results = []

            for val in values:
                dropout    = val if param_name == "dropout"    else DEFAULT_DROPOUT
                hidden_dim = val if param_name == "hidden_dim" else DEFAULT_HIDDEN_DIM
                lr         = val if param_name == "lr"         else DEFAULT_LR

                print(f"\n    {param_name}={val} | dropout={dropout}"
                      f" | hidden={hidden_dim} | lr={lr}")

                model, best_val_auc, history = train_neural_classifier(
                    feat_tr, y_tr, feat_vl, y_vl,
                    epochs=EPOCHS, lr=lr,
                    dropout=dropout, hidden_dim=hidden_dim,
                    verbose=True
                )

                X_te_t = torch.tensor(feat_te, dtype=torch.float32)
                with torch.no_grad():
                    logits = model(X_te_t.to(DEVICE))
                    probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    preds  = logits.argmax(dim=1).cpu().numpy()
                test_auc = roc_auc_score(y_te, probs)
                test_acc = accuracy_score(y_te, preds)

                print(f"    Val AUC: {best_val_auc:.4f} "
                      f"| Test AUC: {test_auc:.4f}")

                param_results.append({
                    "value":    val,
                    "val_auc":  best_val_auc,
                    "test_auc": test_auc,
                    "history":  history['val_auc'],
                })

                if test_auc > best_overall["test_auc"]:
                    best_overall = {
                        "test_auc":  test_auc,
                        "test_acc":  test_acc,
                        "extractor": ext_name,
                        "params": {
                            "dropout":    dropout,
                            "hidden_dim": hidden_dim,
                            "lr":         lr,
                        }
                    }

            sweep_results[param_name] = param_results

        all_sweep_results[ext_name] = sweep_results
        plot_sweep(sweep_results, ext_name)

    #  Best overall 
    print(f"\n{'='*60}")
    print(f"BEST OVERALL:")
    print(f"  Extractor: {best_overall['extractor']}")
    print(f"  Test AUC:  {best_overall['test_auc']:.4f}")
    print(f"  Params:    {best_overall['params']}")
    print(f"{'='*60}")

    #  Train final model with best params 
    best_ext          = best_overall["extractor"]
    p                 = best_overall["params"]
    feat_tr, feat_vl, feat_te = feat_dict[best_ext]

    print(f"\nTraining final model: {best_ext} | {p}")

    final_model, final_val_auc, final_history = train_neural_classifier(
        feat_tr, y_tr, feat_vl, y_vl,
        epochs=150,
        lr=p["lr"],
        dropout=p["dropout"],
        hidden_dim=p["hidden_dim"],
        verbose=True
    )

    X_te_t = torch.tensor(feat_te, dtype=torch.float32)
    with torch.no_grad():
        logits         = final_model(X_te_t.to(DEVICE))
        probs          = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds          = logits.argmax(dim=1).cpu().numpy()
    final_test_auc = roc_auc_score(y_te, probs)
    final_test_acc = accuracy_score(y_te, preds)

    print(f"\nFINAL MODEL RESULTS:")
    print(f"  Extractor: {best_ext}")
    print(f"  Dropout:   {p['dropout']}")
    print(f"  Hidden:    {p['hidden_dim']}")
    print(f"  LR:        {p['lr']}")
    print(f"  Val AUC:   {final_val_auc:.4f}")
    print(f"  Test AUC:  {final_test_auc:.4f}")
    print(f"  Test Acc:  {final_test_acc:.4f}")
    print(classification_report(y_te, preds,
          target_names=["False Alarm", "True Alarm"]))

    torch.save(final_model.state_dict(),
               MODELS_DIR / "final_best_model.pt")

    with open(RESULTS / "final_model_params.json", "w") as f:
        json.dump({
            "extractor":  best_ext,
            "dropout":    p["dropout"],
            "hidden_dim": p["hidden_dim"],
            "lr":         p["lr"],
            "val_auc":    final_val_auc,
            "test_auc":   final_test_auc,
            "test_acc":   final_test_acc,
        }, f, indent=2)

    print("Saved: final_best_model.pt")
    print("Saved: final_model_params.json")
    return all_sweep_results, best_overall


#  Plot Functions 
def plot_sweep(sweep_results: dict, ext_name: str = ""):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0a0a0a')

    param_labels = {
        "dropout":    "Dropout Rate",
        "hidden_dim": "Hidden Dimension",
        "lr":         "Learning Rate",
    }

    for ax_idx, (param_name, results) in enumerate(sweep_results.items()):
        ax = axes[ax_idx]
        ax.set_facecolor('#111')
        ax.tick_params(colors='#888')
        for spine in ax.spines.values():
            spine.set_color('#333')

        values   = [str(r['value']) for r in results]
        val_aucs = [r['val_auc']  for r in results]
        tst_aucs = [r['test_auc'] for r in results]
        x        = np.arange(len(values))

        ax.plot(x, val_aucs,  'o-', color='#3b82f6',
                linewidth=2, markersize=7, label='Val AUC')
        ax.plot(x, tst_aucs, 's--', color='#ef4444',
                linewidth=2, markersize=7, label='Test AUC')
        ax.axhline(0.5, color='#444', linestyle=':', linewidth=1,
                   label='Random baseline')

        best_idx = int(np.argmax(tst_aucs))
        ax.annotate(f"Best: {tst_aucs[best_idx]:.3f}",
                    xy=(x[best_idx], tst_aucs[best_idx]),
                    xytext=(x[best_idx], tst_aucs[best_idx] + 0.03),
                    color='#ef4444', fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='->', color='#ef4444'))

        ax.set_xticks(x)
        ax.set_xticklabels(values, color='#888', fontsize=9)
        ax.set_ylim(0.35, 0.80)
        ax.set_xlabel(param_labels[param_name], color='#aaa', fontsize=11)
        ax.set_ylabel('AUC-ROC' if ax_idx == 0 else '', color='#aaa')
        ax.set_title(f'{param_labels[param_name]} Sweep',
                     color='white', fontsize=12)
        ax.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=9)

    fig.suptitle(
        f'SigmaMedStat - Hyperparameter Sweep ({ext_name})',
        color='white', fontsize=14
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"hyperparameter_sweep_{ext_name}.png",
                dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print(f"Saved: hyperparameter_sweep_{ext_name}.png")

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0a0a0a')
    colors = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b']

    for ax_idx, (param_name, results) in enumerate(sweep_results.items()):
        ax = axes[ax_idx]
        ax.set_facecolor('#111')
        ax.tick_params(colors='#888')
        for spine in ax.spines.values():
            spine.set_color('#333')

        for i, r in enumerate(results):
            epochs = list(range(1, len(r['history']) + 1))
            ax.plot(epochs, r['history'], color=colors[i],
                    linewidth=1.5, alpha=0.85,
                    label=f"{param_name}={r['value']}")

        ax.axhline(0.5, color='#444', linestyle=':', linewidth=1)
        ax.set_xlabel('Epoch', color='#aaa', fontsize=10)
        ax.set_ylabel('Val AUC' if ax_idx == 0 else '', color='#aaa')
        ax.set_title(f'{param_labels[param_name]} - Training Curves',
                     color='white', fontsize=11)
        ax.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=8)

    fig.suptitle(
        f'SigmaMedStat - Training Curves ({ext_name})',
        color='white', fontsize=13
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"sweep_curves_{ext_name}.png",
                dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print(f"Saved: sweep_curves_{ext_name}.png")


def plot_roc_curves(all_results: dict, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#111')
    ax.tick_params(colors='#888')
    for spine in ax.spines.values():
        spine.set_color('#333')

    colors = ['#ef4444','#3b82f6','#10b981','#f59e0b',
              '#8b5cf6','#ec4899','#06b6d4','#84cc16']
    for (name, res), color in zip(all_results.items(), colors):
        fpr, tpr, _ = roc_curve(res['labels'], res['probs'])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={res['auc']:.3f})")

    ax.plot([0,1],[0,1],'--',color='#444',linewidth=1)
    ax.set_xlabel('False Positive Rate', color='#aaa', fontsize=12)
    ax.set_ylabel('True Positive Rate',  color='#aaa', fontsize=12)
    ax.set_title(title, color='white', fontsize=13)
    ax.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150,
                facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def plot_model_comparison(summary: dict):
    names = list(summary.keys())
    aucs  = [summary[n]['test_auc'] for n in names]
    accs  = [summary[n]['test_acc'] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0a0a0a')

    colors = ['#ef4444','#3b82f6','#10b981','#f59e0b',
              '#8b5cf6','#ec4899','#06b6d4','#84cc16']

    for ax, vals, title in [(ax1, aucs, 'Test AUC-ROC'),
                             (ax2, accs, 'Test Accuracy')]:
        ax.set_facecolor('#111')
        ax.tick_params(colors='#888')
        for spine in ax.spines.values():
            spine.set_color('#333')
        bars = ax.bar(names, vals, color=colors[:len(names)], alpha=0.85)
        ax.set_title(title, color='white', fontsize=13)
        ax.set_ylim(0.4, 1.0)
        ax.set_xticklabels(names, rotation=30, ha='right',
                           color='#aaa', fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom',
                    color='white', fontsize=9)

    fig.suptitle('SigmaMedStat - Model Comparison', color='white', fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150,
                facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print("Saved: model_comparison.png")


def get_classifiers():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42),
        "SVM_RBF": SVC(
            kernel='rbf', C=1.0, probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42),
    }


#  Main 
def main():
    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")
    print(f"Dataset: {X.shape} | {y.sum()} true / {(1-y).sum()} false")

    np.random.seed(42)
    idx    = np.random.permutation(len(X))
    tr_end = int(0.70 * len(X))
    vl_end = int(0.85 * len(X))
    X_tr, y_tr = X[idx[:tr_end]],       y[idx[:tr_end]]
    X_vl, y_vl = X[idx[tr_end:vl_end]], y[idx[tr_end:vl_end]]
    X_te, y_te = X[idx[vl_end:]],       y[idx[vl_end:]]
    print(f"Train {len(X_tr)} | Val {len(X_vl)} | Test {len(X_te)}")

    #  Step 1: Extract features for all extractors 
    feat_dict = {}
    for ext_name in ["resnet18", "resnet50", "efficientnet"]:
        print(f"\nExtracting {ext_name} features...")
        ext   = build_extractor(ext_name)
        f_tr  = extract_features(ext, X_tr)
        f_vl  = extract_features(ext, X_vl)
        f_te  = extract_features(ext, X_te)
        sc    = StandardScaler()
        f_tr  = sc.fit_transform(f_tr)
        f_vl  = sc.transform(f_vl)
        f_te  = sc.transform(f_te)
        feat_dict[ext_name] = (f_tr, f_vl, f_te)
        print(f"  Feature dim: {f_tr.shape[1]}")

    #  Step 2: Full hyperparameter sweep 
    all_sweep_results, best_overall = run_sweep(
        feat_dict, y_tr, y_vl, y_te
    )

    #  Step 3: Full model comparison (all classifiers) 
    print(f"\n{'='*60}")
    print("Full model comparison...")
    print(f"{'='*60}")

    summary = {}
    all_roc = {}

    for ext_name, (f_tr, f_vl, f_te) in feat_dict.items():
        print(f"\nExtractor: {ext_name}")

        for clf_name, clf in get_classifiers().items():
            key = f"{ext_name}+{clf_name}"
            clf.fit(f_tr, y_tr)
            probs = clf.predict_proba(f_te)[:, 1]
            preds = clf.predict(f_te)
            auc   = roc_auc_score(y_te, probs)
            acc   = accuracy_score(y_te, preds)
            summary[key] = {'test_auc': auc, 'test_acc': acc}
            all_roc[key] = {'labels': y_te.tolist(),
                            'probs':  probs.tolist(), 'auc': auc}
            print(f"  {clf_name:25s} → AUC {auc:.3f}")

        # NeuralNet with best params from sweep
        p = best_overall["params"]
        model, best_auc, _ = train_neural_classifier(
            f_tr, y_tr, f_vl, y_vl,
            epochs=100, lr=p.get("lr", 1e-3),
            dropout=p.get("dropout", 0.5),
            hidden_dim=p.get("hidden_dim", 256),
            verbose=False
        )
        X_te_t = torch.tensor(f_te, dtype=torch.float32)
        with torch.no_grad():
            logits   = model(X_te_t.to(DEVICE))
            probs_te = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds_te = logits.argmax(dim=1).cpu().numpy()
        auc = roc_auc_score(y_te, probs_te)
        acc = accuracy_score(y_te, preds_te)
        key = f"{ext_name}+NeuralNet"
        summary[key] = {'test_auc': auc, 'test_acc': acc}
        all_roc[key] = {'labels': y_te.tolist(),
                        'probs':  probs_te.tolist(), 'auc': auc}
        print(f"  {'NeuralNet':25s} → AUC {auc:.3f}")
        if ext_name == best_overall["extractor"]:
            print(classification_report(y_te, preds_te,
                  target_names=["False Alarm", "True Alarm"]))
        torch.save(model.state_dict(),
                   MODELS_DIR / f"{ext_name}_neural_best.pt")

    plot_roc_curves(all_roc, "SigmaMedStat - ROC Curves", "roc_all.png")
    plot_model_comparison(summary)

    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("LEADERBOARD - Test AUC")
    print(f"{'='*60}")
    ranked = sorted(summary.items(),
                    key=lambda x: x[1]['test_auc'], reverse=True)
    for name, metrics in ranked:
        print(f"  {name:45s} AUC {metrics['test_auc']:.3f} "
              f"| Acc {metrics['test_acc']:.3f}")

    best_name, best_metrics = ranked[0]
    print(f"\nBEST: {best_name}")
    print(f"AUC:  {best_metrics['test_auc']:.3f}")
    print(f"ACC:  {best_metrics['test_acc']:.3f}")
    print(f"\nBest from sweep: {best_overall['extractor']}"
          f" | AUC {best_overall['test_auc']:.4f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()