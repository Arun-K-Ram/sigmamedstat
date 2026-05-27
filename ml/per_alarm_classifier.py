"""
SigmaMedStat - Per-Alarm-Type Classifier
Train separate XGBoost model for each alarm type.

Alarm types:
  a = Asystole        (63 true, 59 false)
  b = Bradycardia     (44 true, 45 false)
  t = Tachycardia     (68 true, 72 false)
  v = Ventricular     (172 true, 169 false)

Key insight: each alarm type has completely different
physiological signatures. A single model can't learn all four.
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report,
    roc_curve, confusion_matrix
)
import xgboost as xgb
import wfdb

DATA_DIR     = Path("../backend/data/physionet/training")
FEATURES_DIR = Path("../backend/data/features")
RESULTS      = Path("results")
PLOTS_DIR    = RESULTS / "plots"
MODELS_DIR   = RESULTS / "models"
for d in [RESULTS, PLOTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ALARM_TYPES = {
    'a': 'Asystole',
    'b': 'Bradycardia',
    't': 'Tachycardia',
    'v': 'Ventricular Flutter/Fib'
}


def load_beat_features():
    X = np.load(FEATURES_DIR / "X_beat.npy")
    y = np.load(FEATURES_DIR / "y_beat.npy")
    with open(FEATURES_DIR / "beat_feature_names.txt") as f:
        feat_names = [l.strip() for l in f.readlines()]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, feat_names


def get_alarm_type_indices():
    """Get indices for each alarm type from the dataset."""
    records = sorted([f.stem for f in DATA_DIR.glob("*.hea")])
    indices = {'a': [], 'b': [], 't': [], 'v': []}
    for i, rec in enumerate(records):
        alarm_type = rec[0]
        if alarm_type in indices:
            indices[alarm_type].append(i)
    return indices, records


def plot_per_type_roc(all_results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor('#0a0a0a')
    axes = axes.flatten()

    colors = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b']

    for idx, (alarm_type, name) in enumerate(ALARM_TYPES.items()):
        ax = axes[idx]
        ax.set_facecolor('#111')
        ax.tick_params(colors='#888')
        for spine in ax.spines.values():
            spine.set_color('#333')

        if alarm_type in all_results:
            res = all_results[alarm_type]
            fpr, tpr, _ = roc_curve(res['labels'], res['probs'])
            ax.plot(fpr, tpr, color=colors[idx], linewidth=2,
                    label=f"AUC={res['auc']:.3f}")
            ax.fill_between(fpr, tpr, alpha=0.1, color=colors[idx])

        ax.plot([0,1],[0,1],'--',color='#444',linewidth=1)
        ax.set_title(f'{name} ({alarm_type})',
                     color='white', fontsize=12)
        ax.set_xlabel('False Positive Rate', color='#aaa')
        ax.set_ylabel('True Positive Rate',  color='#aaa')
        ax.legend(facecolor='#1a1a1a', labelcolor='white')

    fig.suptitle('SigmaMedStat - Per-Alarm-Type ROC Curves',
                 color='white', fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_per_alarm_type.png", dpi=150,
                facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print("Saved: roc_per_alarm_type.png")


def plot_auc_summary(all_results: dict):
    types  = list(all_results.keys())
    names  = [ALARM_TYPES[t] for t in types]
    aucs   = [all_results[t]['auc'] for t in types]
    cv_aucs= [all_results[t]['cv_auc'] for t in types]
    errors = [all_results[t]['cv_std'] for t in types]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#111')
    ax.tick_params(colors='#888')
    for spine in ax.spines.values():
        spine.set_color('#333')

    x = np.arange(len(types))
    w = 0.35
    colors = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b']

    bars1 = ax.bar(x - w/2, aucs,   w, label='Test AUC',
                   color=colors, alpha=0.85)
    bars2 = ax.bar(x + w/2, cv_aucs, w, label='CV AUC',
                   color=colors, alpha=0.5,
                   yerr=errors, capsize=5, ecolor='white')

    ax.axhline(0.5, color='#444', linestyle='--', linewidth=1,
               label='Random baseline')
    ax.set_ylim(0.3, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names, color='#aaa', fontsize=10)
    ax.set_ylabel('AUC-ROC', color='#aaa')
    ax.set_title('XGBoost Performance by Alarm Type',
                 color='white', fontsize=13)
    ax.legend(facecolor='#1a1a1a', labelcolor='white')

    for bar, val in zip(bars1, aucs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom',
                color='white', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "auc_by_alarm_type.png", dpi=150,
                facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print("Saved: auc_by_alarm_type.png")


def train_per_type():
    X, y, feat_names = load_beat_features()
    indices, records = get_alarm_type_indices()

    print(f"Total dataset: {X.shape}")
    print(f"\nAlarm type distribution:")
    for t, name in ALARM_TYPES.items():
        idx = indices[t]
        y_t = y[idx]
        print(f"  {name:30s} n={len(idx):3d} | "
              f"{y_t.sum()} true / {(1-y_t).sum()} false")

    xgb_params = {
        'n_estimators':    [100, 200, 300],
        'max_depth':       [3, 4, 6],
        'learning_rate':   [0.01, 0.05, 0.1],
        'reg_alpha':       [0, 0.1, 1.0],
        'reg_lambda':      [1.0, 5.0],
        'subsample':       [0.8, 1.0],
        'min_child_weight':[1, 3],
    }

    all_results = {}
    summary     = {}

    for alarm_type, name in ALARM_TYPES.items():
        print(f"\n{'='*60}")
        print(f"Training: {name} ({alarm_type})")
        print(f"{'='*60}")

        idx = indices[alarm_type]
        X_t = X[idx]
        y_t = y[idx]

        n = len(X_t)
        if n < 20:
            print(f"  Skipping - too few samples ({n})")
            continue

        # Split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_t, y_t,
            test_size=0.2,
            random_state=42,
            stratify=y_t
        )
        print(f"  Train: {len(X_tr)} | Test: {len(X_te)}")

        # Scale
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # CV - use 3-fold for small datasets
        n_splits = min(5, min(np.bincount(y_tr)))
        n_splits = max(3, n_splits)
        cv = StratifiedKFold(n_splits=n_splits,
                             shuffle=True, random_state=42)

        # XGBoost grid search
        xgb_base = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            tree_method='hist',
            device='cpu'
        )

        search = GridSearchCV(
            xgb_base, xgb_params,
            scoring='roc_auc',
            cv=cv, n_jobs=-1,
            verbose=0, refit=True
        )
        search.fit(X_tr_s, y_tr)

        best = search.best_estimator_
        print(f"  Best params: {search.best_params_}")
        print(f"  Best CV AUC: {search.best_score_:.4f}")

        # Test evaluation
        probs = best.predict_proba(X_te_s)[:, 1]
        preds = best.predict(X_te_s)
        auc   = roc_auc_score(y_te, probs)
        acc   = accuracy_score(y_te, preds)

        # CV on full training set
        cv_aucs = cross_val_score(
            best, X_tr_s, y_tr,
            cv=cv, scoring='roc_auc', n_jobs=-1
        )

        print(f"\n  Test AUC: {auc:.4f} | Acc: {acc:.4f}")
        print(f"  CV  AUC:  {cv_aucs.mean():.4f} "
              f"± {cv_aucs.std():.4f}")
        print(classification_report(
            y_te, preds,
            target_names=["False Alarm", "True Alarm"]
        ))

        all_results[alarm_type] = {
            'auc':    auc,
            'acc':    acc,
            'cv_auc': float(cv_aucs.mean()),
            'cv_std': float(cv_aucs.std()),
            'labels': y_te.tolist(),
            'probs':  probs.tolist(),
            'name':   name,
            'n_train':len(X_tr),
            'n_test': len(X_te),
        }
        summary[name] = {
            'test_auc': auc,
            'test_acc': acc,
            'cv_auc':   float(cv_aucs.mean()),
            'cv_std':   float(cv_aucs.std()),
            'best_params': search.best_params_
        }

        # Save model
        best.save_model(
            str(MODELS_DIR / f"xgb_{alarm_type}.json")
        )

    #  Overall weighted AUC 
    if all_results:
        weights  = [all_results[t]['n_test']
                    for t in all_results]
        aucs     = [all_results[t]['auc']
                    for t in all_results]
        wtd_auc  = np.average(aucs, weights=weights)

        print(f"\n{'='*60}")
        print("FINAL RESULTS - Per-Type XGBoost")
        print(f"{'='*60}")
        for t, res in all_results.items():
            print(f"  {res['name']:30s} "
                  f"Test AUC {res['auc']:.4f} | "
                  f"CV AUC {res['cv_auc']:.4f}±{res['cv_std']:.4f}")
        print(f"\n  Weighted Average AUC: {wtd_auc:.4f}")

        summary['weighted_average_auc'] = wtd_auc

        # Plots
        plot_per_type_roc(all_results)
        plot_auc_summary(all_results)

        with open(RESULTS / "per_alarm_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return all_results


if __name__ == "__main__":
    np.random.seed(42)
    all_results = train_per_type()