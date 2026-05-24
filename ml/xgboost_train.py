"""
SigmaMedStat — XGBoost with Hyperparameter Tuning
Hand-crafted signal features + gradient boosting.
Full grid search with cross-validation.
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report,
    roc_curve, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

DATA_DIR   = Path("../backend/data/features")
RESULTS    = Path("results")
PLOTS_DIR  = RESULTS / "plots"
MODELS_DIR = RESULTS / "models"
for d in [RESULTS, PLOTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_data():
    X = np.load(DATA_DIR / "X_features.npy")
    y = np.load(DATA_DIR / "y_features.npy")

    with open(DATA_DIR / "feature_names.txt") as f:
        feat_names = [l.strip() for l in f.readlines()]

    print(f"Dataset: {X.shape} | {y.sum()} true / {(1-y).sum()} false")

    # Replace inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, feat_names


def plot_roc(results: dict, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#111')
    ax.tick_params(colors='#888')
    for spine in ax.spines.values():
        spine.set_color('#333')

    colors = ['#ef4444','#3b82f6','#10b981',
              '#f59e0b','#8b5cf6','#ec4899']
    for (name, res), color in zip(results.items(), colors):
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


def plot_feature_importance(model, feat_names: list, top_n: int = 20):
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#111')
    ax.tick_params(colors='#888')
    for spine in ax.spines.values():
        spine.set_color('#333')

    colors = ['#ef4444' if 'PLETH' in feat_names[i]
              else '#3b82f6' if 'II' in feat_names[i]
              else '#10b981' if 'RESP' in feat_names[i]
              else '#f59e0b'
              for i in idx]

    ax.barh([feat_names[i] for i in idx],
            importances[idx], color=colors, alpha=0.85)
    ax.set_title('Top 20 Most Important Features — XGBoost',
                 color='white', fontsize=13)
    ax.set_xlabel('Feature Importance', color='#aaa')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150,
                facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print("Saved: feature_importance.png")


def plot_confusion_matrix(y_true, y_pred, title: str, fname: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#111')

    im = ax.imshow(cm, cmap='Reds', alpha=0.8)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(['False Alarm', 'True Alarm'], color='#aaa')
    ax.set_yticklabels(['False Alarm', 'True Alarm'], color='#aaa')
    ax.set_xlabel('Predicted', color='#aaa')
    ax.set_ylabel('Actual', color='#aaa')
    ax.set_title(title, color='white', fontsize=12)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]),
                    ha='center', va='center',
                    color='white', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150,
                facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()

def load_beat_data():
    X = np.load(DATA_DIR / "X_beat.npy")
    y = np.load(DATA_DIR / "y_beat.npy")
    with open(DATA_DIR / "beat_feature_names.txt") as f:
        feat_names = [l.strip() for l in f.readlines()]
    print(f"Beat dataset: {X.shape} | {y.sum()} true / {(1-y).sum()} false")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, feat_names

def main():
    X, y, feat_names = load_beat_data()

    # Split — stratified to preserve class balance
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr
    )
    print(f"Train {len(X_tr)} | Val {len(X_vl)} | Test {len(X_te)}")

    # Robust scaler — handles outliers in feature range better than Standard
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)
    X_te_s = scaler.transform(X_te)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── XGBoost Hyperparameter Search ────────────────────────
    print("\n" + "="*60)
    print("XGBoost Hyperparameter Search")
    print("="*60)

    xgb_param_grid = {
        'n_estimators':    [100, 200, 300],
        'max_depth':       [3, 4, 6],
        'learning_rate':   [0.01, 0.05, 0.1],
        'subsample':       [0.8, 1.0],
        'colsample_bytree':[0.8, 1.0],
        'reg_alpha':       [0, 0.1, 1.0],   # L1
        'reg_lambda':      [1.0, 2.0, 5.0], # L2
        'min_child_weight':[1, 3, 5],
    }

    # Reduced grid for speed — most important params
    xgb_param_grid_fast = {
    'n_estimators':    [100, 200],
    'max_depth':       [3, 4],
    'learning_rate':   [0.05, 0.1],
    'reg_alpha':       [0, 0.1],
    'reg_lambda':      [1.0, 5.0],
}

    xgb_base = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    tree_method='hist',  # CPU histogram method — more stable
    device='cpu'         # force CPU for grid search
)

    print("Running grid search (this takes ~10-15 mins)...")
    xgb_search = GridSearchCV(
        xgb_base,
        xgb_param_grid_fast,
        scoring='roc_auc',
        cv=cv,
        n_jobs=1,
        verbose=1,
        refit=True
    )
    xgb_search.fit(X_tr_s, y_tr)

    print(f"\nBest XGBoost params: {xgb_search.best_params_}")
    print(f"Best CV AUC: {xgb_search.best_score_:.4f}")

    best_xgb = xgb_search.best_estimator_

    # ── Compare all models with best hyperparams ─────────────
    print("\n" + "="*60)
    print("Model Comparison on Test Set")
    print("="*60)

    models = {
        "XGBoost (tuned)": best_xgb,
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=6,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=42
        ),
        "LogisticRegression (L1)": LogisticRegression(
            penalty='l1', solver='liblinear', C=0.1,
            max_iter=1000, random_state=42
        ),
        "LogisticRegression (L2)": LogisticRegression(
            penalty='l2', C=1.0,
            max_iter=1000, random_state=42
        ),
        "SVM (RBF)": SVC(
            kernel='rbf', C=1.0, gamma='scale',
            probability=True, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        if name != "XGBoost (tuned)":
            model.fit(X_tr_s, y_tr)

        probs = model.predict_proba(X_te_s)[:, 1]
        preds = model.predict(X_te_s)
        auc   = roc_auc_score(y_te, probs)
        acc   = accuracy_score(y_te, preds)

        # Cross-validation AUC
        cv_aucs = cross_val_score(
            model, X_tr_s, y_tr,
            cv=cv, scoring='roc_auc', n_jobs=-1
        )

        results[name] = {
            'auc':    auc,
            'acc':    acc,
            'cv_auc': cv_aucs.mean(),
            'cv_std': cv_aucs.std(),
            'labels': y_te.tolist(),
            'probs':  probs.tolist()
        }

        print(f"\n{name}")
        print(f"  Test AUC: {auc:.4f} | Acc: {acc:.4f}")
        print(f"  CV  AUC:  {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")
        print(classification_report(
            y_te, preds,
            target_names=["False Alarm", "True Alarm"]
        ))

    # ── Feature importance ────────────────────────────────────
    plot_feature_importance(best_xgb, feat_names)

    # ── Confusion matrix for best model ──────────────────────
    best_preds = best_xgb.predict(X_te_s)
    plot_confusion_matrix(
        y_te, best_preds,
        "XGBoost — Confusion Matrix",
        "confusion_matrix.png"
    )

    # ── ROC curves ───────────────────────────────────────────
    plot_roc(results, "SigmaMedStat — ROC Curves (Signal Features)", "roc_features.png")

    # ── Leaderboard ───────────────────────────────────────────
    print("\n" + "="*60)
    print("LEADERBOARD")
    print("="*60)
    ranked = sorted(results.items(),
                    key=lambda x: x[1]['auc'], reverse=True)
    for name, r in ranked:
        print(f"  {name:35s} Test AUC {r['auc']:.4f} "
              f"| CV AUC {r['cv_auc']:.4f}±{r['cv_std']:.4f} "
              f"| Acc {r['acc']:.4f}")

    best_name, best_r = ranked[0]
    print(f"\nBEST MODEL: {best_name}")
    print(f"TEST AUC:   {best_r['auc']:.4f}")
    print(f"CV AUC:     {best_r['cv_auc']:.4f} ± {best_r['cv_std']:.4f}")

    # ── Save summary ──────────────────────────────────────────
    summary = {
        k: {
            'test_auc': v['auc'],
            'test_acc': v['acc'],
            'cv_auc':   v['cv_auc'],
            'cv_std':   v['cv_std']
        }
        for k, v in results.items()
    }
    summary['best_xgb_params'] = xgb_search.best_params_
    with open(RESULTS / "xgb_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Check results/plots/ for visualizations.")


if __name__ == "__main__":
    np.random.seed(42)
    main()