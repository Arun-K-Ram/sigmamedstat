"""
SigmaMedStat - Threshold Optimization
Uses out-of-fold predictions from 5-fold CV to find
the optimal classification threshold.

Generates:
  1. ROC curve with optimal threshold marked
  2. Precision-Recall curve with optimal F1 marked
  3. Threshold sweep table (0.1 to 0.9)
  4. Youden's J optimal threshold
  5. Clinical threshold (sensitivity >= 0.80)
"""

import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    auc, f1_score, confusion_matrix,
    roc_auc_score
)

#  Paths 
RESULTS_DIR = Path("results")
OUTPUT_DIR  = Path("results/paper_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WHITE    = "#ffffff"
CHARCOAL = "#2c3e50"
RED      = "#c0392b"
GREEN    = "#1e8449"
BLUE     = "#1a3a6b"
GREY     = "#7f8c8d"
LGREY    = "#e8e8e5"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.facecolor":    WHITE,
    "axes.edgecolor":    LGREY,
    "axes.grid":         True,
    "grid.color":        LGREY,
    "grid.linewidth":    0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.facecolor": WHITE,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})


def compute_metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1          = (2 * precision * sensitivity /
                   (precision + sensitivity)
                   if (precision + sensitivity) > 0 else 0)
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0
    return {
        "threshold":   round(threshold, 2),
        "sensitivity": round(sensitivity, 3),
        "specificity": round(specificity, 3),
        "precision":   round(precision, 3),
        "f1":          round(f1, 3),
        "npv":         round(npv, 3),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
    }


def main():
    # Load OOF predictions
    print("Loading OOF predictions...")
    with open(RESULTS_DIR / "experiment_04_kfold.json") as f:
        data = json.load(f)

    ea       = data["error_analysis"]
    y_prob   = np.array(ea["oof_probs"])
    y_true   = np.array(ea["oof_labels"], dtype=int)

    print(f"  Samples: {len(y_true)}")
    print(f"  True alarms: {y_true.sum()}")
    print(f"  False alarms: {(y_true==0).sum()}")
    print(f"  OOF AUC: {roc_auc_score(y_true, y_prob):.4f}\n")

    #  ROC Curve 
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Youden's J statistic - maximizes sensitivity + specificity
    youden_j    = tpr - fpr
    youden_idx  = np.argmax(youden_j)
    youden_thr  = roc_thresholds[youden_idx]
    youden_tpr  = tpr[youden_idx]
    youden_fpr  = fpr[youden_idx]

    print(f"Youden's J optimal threshold: {youden_thr:.3f}")
    print(f"  Sensitivity: {youden_tpr:.3f}")
    print(f"  Specificity: {1-youden_fpr:.3f}\n")

    #  Precision-Recall Curve 
    precision_curve, recall_curve, pr_thresholds = \
        precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)

    # Best F1 threshold
    f1_scores   = (2 * precision_curve * recall_curve /
                   (precision_curve + recall_curve + 1e-8))
    best_f1_idx = np.argmax(f1_scores)
    best_f1_thr = (pr_thresholds[best_f1_idx]
                   if best_f1_idx < len(pr_thresholds)
                   else pr_thresholds[-1])
    best_f1_val = f1_scores[best_f1_idx]

    print(f"Best F1 threshold: {best_f1_thr:.3f}")
    print(f"  F1: {best_f1_val:.3f}\n")

    #  Clinical threshold (sensitivity >= 0.80) 
    clinical_thr  = None
    clinical_spec = None
    for thr in np.arange(0.05, 0.95, 0.01):
        m = compute_metrics_at_threshold(y_true, y_prob, thr)
        if m["sensitivity"] >= 0.80:
            clinical_thr  = thr
            clinical_spec = m["specificity"]
            break

    if clinical_thr:
        print(f"Clinical threshold (sens>=0.80): {clinical_thr:.2f}")
        print(f"  Specificity at that point: {clinical_spec:.3f}\n")

    #  Threshold sweep table 
    print("Threshold sweep:")
    print(f"{'Thr':>5} {'Sens':>6} {'Spec':>6} "
          f"{'Prec':>6} {'F1':>6} {'FN':>4} {'FP':>4}")
    print("-" * 45)
    sweep_results = []
    for thr in np.arange(0.1, 0.95, 0.05):
        m = compute_metrics_at_threshold(y_true, y_prob, thr)
        sweep_results.append(m)
        marker = " ←" if abs(thr - youden_thr) < 0.03 else ""
        print(f"{thr:>5.2f} {m['sensitivity']:>6.3f} "
              f"{m['specificity']:>6.3f} {m['precision']:>6.3f} "
              f"{m['f1']:>6.3f} {m['fn']:>4} {m['fp']:>4}{marker}")

    #  Figure: ROC + PR curves 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # ROC curve
    ax1.plot(fpr, tpr, color=RED, lw=2,
             label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color=GREY, lw=1,
             linestyle="--", label="Random classifier")
    ax1.scatter([youden_fpr], [youden_tpr],
                color=RED, s=100, zorder=5,
                label=f"Youden's J optimum\n"
                      f"(threshold={youden_thr:.2f},\n"
                      f"sens={youden_tpr:.2f}, "
                      f"spec={1-youden_fpr:.2f})")
    if clinical_thr:
        cm = compute_metrics_at_threshold(
            y_true, y_prob, clinical_thr)
        ax1.scatter([1-cm["specificity"]], [cm["sensitivity"]],
                    color=BLUE, s=100, zorder=5, marker="^",
                    label=f"Clinical threshold (sens≥0.80)\n"
                          f"(threshold={clinical_thr:.2f},\n"
                          f"spec={cm['specificity']:.2f})")

    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Sensitivity)")
    ax1.set_title("(a) ROC Curve - Experiment 04\n"
                  "EfficientNet + LSTM (5-fold OOF)",
                  fontsize=10, color=CHARCOAL)
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    # PR curve
    ax2.plot(recall_curve, precision_curve,
             color=BLUE, lw=2,
             label=f"PR curve (AUC = {pr_auc:.3f})")
    ax2.axhline(y=y_true.mean(), color=GREY, lw=1,
                linestyle="--",
                label=f"Random classifier ({y_true.mean():.2f})")
    ax2.scatter([recall_curve[best_f1_idx]],
                [precision_curve[best_f1_idx]],
                color=BLUE, s=100, zorder=5,
                label=f"Best F1={best_f1_val:.3f}\n"
                      f"(threshold={best_f1_thr:.2f})")

    ax2.set_xlabel("Recall (Sensitivity)")
    ax2.set_ylabel("Precision")
    ax2.set_title("(b) Precision-Recall Curve - Experiment 04\n"
                  "EfficientNet + LSTM (5-fold OOF)",
                  fontsize=10, color=CHARCOAL)
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])

    fig.suptitle("Figure 9 - ROC and Precision-Recall Curves",
                 fontsize=12, color=CHARCOAL, y=1.02)
    plt.tight_layout()

    out = OUTPUT_DIR / "fig9_roc_pr_curves.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"\nFigure saved → {out}")

    #  Threshold sweep figure 
    fig2, ax = plt.subplots(figsize=(9, 5))
    thresholds = [r["threshold"] for r in sweep_results]
    sens_vals  = [r["sensitivity"] for r in sweep_results]
    spec_vals  = [r["specificity"] for r in sweep_results]
    f1_vals    = [r["f1"]          for r in sweep_results]

    ax.plot(thresholds, sens_vals, color=RED,
            lw=2, label="Sensitivity")
    ax.plot(thresholds, spec_vals, color=BLUE,
            lw=2, label="Specificity")
    ax.plot(thresholds, f1_vals,   color=GREEN,
            lw=2, linestyle="--", label="F1 Score")

    ax.axvline(x=round(youden_thr, 2), color=CHARCOAL,
               lw=1.5, linestyle=":",
               label=f"Youden's J optimum ({youden_thr:.2f})")
    if clinical_thr:
        ax.axvline(x=round(clinical_thr, 2), color=GREY,
                   lw=1.5, linestyle="-.",
                   label=f"Clinical threshold ({clinical_thr:.2f})")

    ax.axhline(y=0.80, color=RED, lw=0.8,
               linestyle="--", alpha=0.4,
               label="Sensitivity = 0.80 target")

    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_ylim([0, 1.05])
    ax.set_title("Figure 10 - Sensitivity, Specificity and F1 "
                 "vs Classification Threshold",
                 fontsize=11, color=CHARCOAL)
    ax.legend(fontsize=8.5, framealpha=0.9)
    plt.tight_layout()

    out2 = OUTPUT_DIR / "fig10_threshold_sweep.pdf"
    plt.savefig(out2)
    plt.savefig(str(out2).replace(".pdf", ".png"))
    plt.close()
    print(f"Figure saved → {out2}")

    #  Save results 
    threshold_results = {
        "roc_auc":       round(float(roc_auc), 4),
        "pr_auc":        round(float(pr_auc), 4),
        "youden_threshold":  round(float(youden_thr), 3),
        "youden_sensitivity": round(float(youden_tpr), 3),
        "youden_specificity": round(float(1-youden_fpr), 3),
        "best_f1_threshold":  round(float(best_f1_thr), 3),
        "best_f1_value":      round(float(best_f1_val), 3),
        "clinical_threshold": round(float(clinical_thr), 3)
                              if clinical_thr else None,
        "clinical_specificity": round(float(clinical_spec), 3)
                                if clinical_spec else None,
        "sweep": sweep_results,
    }

    out_json = RESULTS_DIR / "threshold_optimization.json"
    with open(out_json, "w") as f:
        json.dump(threshold_results, f, indent=2)
    print(f"Results saved → {out_json}")

    #  Summary 
    print("\n" + "=" * 55)
    print("THRESHOLD OPTIMIZATION SUMMARY")
    print("=" * 55)
    print(f"  Default threshold (0.50):")
    m_default = compute_metrics_at_threshold(
        y_true, y_prob, 0.50)
    print(f"    Sensitivity: {m_default['sensitivity']:.3f}")
    print(f"    Specificity: {m_default['specificity']:.3f}")
    print(f"    F1:          {m_default['f1']:.3f}")
    print(f"\n  Youden's J optimum ({youden_thr:.2f}):")
    m_youden = compute_metrics_at_threshold(
        y_true, y_prob, youden_thr)
    print(f"    Sensitivity: {m_youden['sensitivity']:.3f}")
    print(f"    Specificity: {m_youden['specificity']:.3f}")
    print(f"    F1:          {m_youden['f1']:.3f}")
    if clinical_thr:
        print(f"\n  Clinical threshold ({clinical_thr:.2f}):")
        m_clin = compute_metrics_at_threshold(
            y_true, y_prob, clinical_thr)
        print(f"    Sensitivity: {m_clin['sensitivity']:.3f}")
        print(f"    Specificity: {m_clin['specificity']:.3f}")
        print(f"    F1:          {m_clin['f1']:.3f}")
    print("=" * 55)


if __name__ == "__main__":
    main()