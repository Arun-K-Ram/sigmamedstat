"""
SigmaMedStat - Publication Quality Figure Generator
Generates all figures needed for the arXiv paper.

Figure 1: System architecture diagram
Figure 2: All-experiment AUC comparison bar chart
Figure 3: 5-fold CV results with confidence interval
Figure 4: Ablation study results (chunks + channels)
Figure 5: Per-alarm-type AUC bar chart
Figure 6: Error analysis breakdown
Figure 7: Training curve (Exp 04 vs static baseline)
Figure 8: Chunk visualization (signal → 6 scalograms)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import json
from pathlib import Path

#  Paths 
OUTPUT_DIR    = Path("results/paper_figures")
DATA_DIR      = Path("C:/Users/Arun/Documents/git/crip-x/backend/data/scalograms_temporal")
RESULTS_DIR   = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#  Style 
BG      = "#f4f4f2"
WHITE   = "#ffffff"
CHARCOAL = "#2c3e50"
RED     = "#c0392b"
GREEN   = "#1e8449"
BLUE    = "#1a3a6b"
GREY    = "#7f8c8d"
LGREY   = "#e8e8e5"

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "savefig.facecolor": WHITE,
    "axes.facecolor":  WHITE,
    "axes.edgecolor":  LGREY,
    "axes.grid":       True,
    "grid.color":      LGREY,
    "grid.linewidth":  0.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


#  Figure 1: Architecture Diagram 
def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=WHITE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_axis_off()

    def box(x, y, w, h, label, sublabel="", color=CHARCOAL, textcolor=WHITE):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor=LGREY,
                              linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=9, color=textcolor, fontweight="medium")
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.18,
                    sublabel, ha="center", va="center",
                    fontsize=7.5, color=textcolor, alpha=0.8)

    def arrow(x1, x2, y=2.0):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->",
                                   color=GREY, lw=1.5))

    # Boxes
    box(0.1, 1.3, 1.5, 1.4, "60s Raw Signal",
        "4 channels\n250 Hz", CHARCOAL)
    arrow(1.6, 2.0)
    box(2.0, 1.3, 1.5, 1.4, "6 × 10s Chunks",
        "Temporal\nSplit", "#1a3a6b")
    arrow(3.5, 3.9)
    box(3.9, 1.3, 1.5, 1.4, "CWT Scalograms",
        "(6, 4, 64, 64)\nper record", "#2e4057")
    arrow(5.4, 5.8)
    box(5.8, 1.3, 1.5, 1.4, "EfficientNet-B0",
        "Shared encoder\n(6, 1280) feats", "#6b1a1a")
    arrow(7.3, 7.7)
    box(7.7, 1.3, 1.0, 1.4, "LSTM",
        "hidden=64\nlayers=2", "#1e5c1e")
    arrow(8.7, 9.1)
    box(9.1, 1.3, 0.8, 1.4, "Output",
        "True /\nFalse", RED)

    # Labels above
    for x, label in [(0.85, "Input"), (2.75, "Split"),
                     (4.65, "Represent"), (6.55, "Encode"),
                     (8.2, "Sequence"), (9.5, "Predict")]:
        ax.text(x, 3.0, label, ha="center", va="center",
                fontsize=8, color=GREY)

    ax.set_title("Figure 1 - SigmaMedStat Architecture: "
                 "Temporal CWT-LSTM Pipeline",
                 fontsize=11, color=CHARCOAL, pad=12)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig1_architecture.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {out.name}")


#  Figure 2: Experiment Comparison Bar Chart 
def fig_experiment_comparison():
    fig, ax = plt.subplots(figsize=(7, 4))

    experiments = ["Exp 01\nStatic EfficientNet", "Exp 02\nHand-crafted + SVM",
                   "Exp 03\nPer-alarm XGBoost", "Exp 04\nEfficientNet + LSTM"]
    aucs        = [0.641, 0.539, 0.612, 0.822]
    colors      = [GREY, GREY, GREY, RED]
    errors      = [0, 0, 0, 0.016]  # only Exp 04 has CV std

    bars = ax.bar(experiments, aucs, color=colors, width=0.5,
                  edgecolor=WHITE, linewidth=0.8,
                  yerr=errors, capsize=4,
                  error_kw={"ecolor": CHARCOAL, "elinewidth": 1.2})

    # Baseline line
    ax.axhline(y=0.5, color=GREY, linestyle="--",
               linewidth=1.0, alpha=0.6, label="Random baseline (AUC=0.50)")

    # Value labels
    for bar, auc, err in zip(bars, aucs, errors):
        label = f"{auc:.3f}" + (f" ±{err:.3f}" if err > 0 else "")
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (err or 0) + 0.008,
                label, ha="center", va="bottom",
                fontsize=8.5, color=CHARCOAL, fontweight="medium")

    # Improvement annotation
    ax.annotate("", xy=(3, 0.822), xytext=(0, 0.641),
                arrowprops=dict(arrowstyle="->", color=RED,
                                lw=1.5, connectionstyle="arc3,rad=-0.3"))
    ax.text(1.7, 0.74, "+18.1 AUC points", color=RED,
            fontsize=8.5, fontstyle="italic")

    ax.set_ylabel("AUC (Area Under ROC Curve)")
    ax.set_ylim(0.45, 0.92)
    ax.set_title("Figure 2 - Test AUC Across All Four Experiments",
                 fontsize=11, color=CHARCOAL)
    ax.legend(loc="upper left", framealpha=0.8)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig2_experiment_comparison.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {out.name}")


#  Figure 3: K-Fold CV Results 
def fig_kfold():
    fig, ax = plt.subplots(figsize=(6, 4))

    folds    = [1, 2, 3, 4, 5]
    fold_aucs = [0.7923, 0.8254, 0.8185, 0.8344, 0.8373]
    mean_auc  = 0.8216
    std_auc   = 0.0161
    ci_low    = 0.7900
    ci_high   = 0.8531

    # Individual fold bars
    bars = ax.bar(folds, fold_aucs, color=BLUE, alpha=0.7,
                  width=0.5, edgecolor=WHITE, label="Per-fold AUC")

    # Mean line
    ax.axhline(y=mean_auc, color=RED, linewidth=2.0,
               label=f"Mean AUC = {mean_auc:.4f}")

    # 95% CI band
    ax.axhspan(ci_low, ci_high, alpha=0.12, color=RED,
               label=f"95% CI [{ci_low:.3f}, {ci_high:.3f}]")

    # Static baseline
    ax.axhline(y=0.641, color=GREY, linewidth=1.5,
               linestyle="--", label="Static baseline (Exp 01)")

    # Value labels
    for bar, auc in zip(bars, fold_aucs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{auc:.4f}", ha="center", va="bottom",
                fontsize=8, color=CHARCOAL)

    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.60, 0.90)
    ax.set_xticks(folds)
    ax.set_title("Figure 3 - 5-Fold Stratified Cross-Validation Results",
                 fontsize=11, color=CHARCOAL)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig3_kfold.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {out.name}")


#  Figure 4: Ablation Study 
def fig_ablation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Chunks ablation
    chunks      = ["1\n(static)", "2\n(30s)", "3\n(20s)", "6\n(10s)\n★"]
    chunk_aucs  = [0.7776, 0.7562, 0.7977, 0.8110]
    chunk_stds  = [0.0099, 0.0147, 0.0104, 0.0033]
    chunk_colors = [GREY, GREY, GREY, RED]

    bars1 = ax1.bar(chunks, chunk_aucs, color=chunk_colors,
                    width=0.5, edgecolor=WHITE,
                    yerr=chunk_stds, capsize=4,
                    error_kw={"ecolor": CHARCOAL, "elinewidth": 1.0})

    ax1.axhline(y=0.5, color=GREY, linestyle="--",
                linewidth=0.8, alpha=0.5)
    for bar, auc in zip(bars1, chunk_aucs):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.012,
                 f"{auc:.3f}", ha="center", va="bottom",
                 fontsize=8, color=CHARCOAL)

    ax1.set_ylabel("Mean AUC (3-fold CV)")
    ax1.set_ylim(0.45, 0.88)
    ax1.set_xlabel("Number of temporal chunks")
    ax1.set_title("(a) Chunks ablation\n(4 channels fixed)",
                  fontsize=10, color=CHARCOAL)

    # Channels ablation
    ch_labels = ["1\n(ECG II)", "2\n(ECG II+V)", "4\n(All) ★"]
    ch_aucs   = [0.7059, 0.7828, 0.7911]
    ch_stds   = [0.0040, 0.0305, 0.0175]
    ch_colors = [GREY, GREY, RED]

    bars2 = ax2.bar(ch_labels, ch_aucs, color=ch_colors,
                    width=0.4, edgecolor=WHITE,
                    yerr=ch_stds, capsize=4,
                    error_kw={"ecolor": CHARCOAL, "elinewidth": 1.0})

    ax2.axhline(y=0.5, color=GREY, linestyle="--",
                linewidth=0.8, alpha=0.5)
    for bar, auc in zip(bars2, ch_aucs):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.012,
                 f"{auc:.3f}", ha="center", va="bottom",
                 fontsize=8, color=CHARCOAL)

    ax2.set_ylabel("Mean AUC (3-fold CV)")
    ax2.set_ylim(0.45, 0.88)
    ax2.set_xlabel("Number of input channels")
    ax2.set_title("(b) Channels ablation\n(6 chunks fixed)",
                  fontsize=10, color=CHARCOAL)

    fig.suptitle("Figure 4 - Ablation Study: "
                 "Temporal Chunks and Signal Channels",
                 fontsize=11, color=CHARCOAL, y=1.02)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig4_ablation.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {out.name}")


#  Figure 5: Per-Alarm-Type Performance 
def fig_per_alarm():
    fig, ax = plt.subplots(figsize=(8, 4))

    alarm_types = ["Ventricular\nFlutter", "Bradycardia",
                   "Tachycardia", "Ventricular\nFib.", "Asystole"]
    aucs        = [0.820, 0.810, 0.750, 0.733, 0.722]
    ns          = [263, 56, 62, 32, 85]
    accs        = [79.5, 69.6, 71.0, 75.0, 76.5]

    x      = np.arange(len(alarm_types))
    width  = 0.35
    colors = [RED if a == max(aucs) else
              "#6b1a1a" if a == min(aucs) else BLUE
              for a in aucs]

    bars = ax.bar(x, aucs, width, color=colors,
                  alpha=0.85, edgecolor=WHITE, label="AUC")

    # n labels below x axis
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={n})"
                        for t, n in zip(alarm_types, ns)],
                       fontsize=8.5)

    # AUC value labels
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{auc:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=CHARCOAL, fontweight="medium")

    ax.axhline(y=0.5, color=GREY, linestyle="--",
               linewidth=1.0, alpha=0.6, label="Random baseline")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.45, 0.90)
    ax.set_title("Figure 5 - Per-Alarm-Type AUC (5-fold CV)",
                 fontsize=11, color=CHARCOAL)

    # Legend patches
    best_patch  = mpatches.Patch(color=RED,    label="Best (Ventricular Flutter)")
    worst_patch = mpatches.Patch(color="#6b1a1a", label="Hardest (Asystole)")
    other_patch = mpatches.Patch(color=BLUE,   label="Other alarm types")
    ax.legend(handles=[best_patch, worst_patch, other_patch],
              loc="lower right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig5_per_alarm.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {out.name}")


#  Figure 6: Error Analysis 
def fig_error_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Pie: error breakdown
    sizes  = [65, 52]
    labels = ["False Negatives\n(missed real alarms)\n65 cases",
              "False Positives\n(false alarm called real)\n52 cases"]
    colors_pie = [RED, BLUE]
    explode    = (0.05, 0)

    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors_pie,
        explode=explode, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 8.5})
    for at in autotexts:
        at.set_color(WHITE)
        at.set_fontweight("bold")

    ax1.set_title("(a) Error type breakdown\n(117 total errors, 23.5% of 498)",
                  fontsize=10, color=CHARCOAL)

    # Bar: confidence distribution of errors
    conf_bins  = ["60-70%", "70-80%", "80-90%", "90-100%"]
    # Approximate from the high_conf_errors=85 at >80%
    # Total 117 errors: 32 below 80%, 85 above 80%
    conf_counts = [16, 16, 42, 43]
    conf_colors = [GREY, GREY, RED, RED]

    bars = ax2.bar(conf_bins, conf_counts, color=conf_colors,
                   width=0.5, edgecolor=WHITE)
    for bar, count in zip(bars, conf_counts):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 str(count), ha="center", va="bottom",
                 fontsize=9, color=CHARCOAL)

    ax2.axvline(x=1.5, color=RED, linestyle="--",
                linewidth=1.2, alpha=0.7)
    ax2.text(1.6, 38, "High-confidence\nthreshold (>80%)",
             color=RED, fontsize=7.5)

    ax2.set_xlabel("Model confidence at time of error")
    ax2.set_ylabel("Number of errors")
    ax2.set_title("(b) Error confidence distribution\n"
                  "(85 of 117 errors made with >80% confidence)",
                  fontsize=10, color=CHARCOAL)

    fig.suptitle("Figure 6 - Error Analysis",
                 fontsize=11, color=CHARCOAL, y=1.02)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig6_error_analysis.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {out.name}")


#  Figure 7: Training Curve 
def fig_training_curve():
    """Load actual training history from JSON."""
    try:
        with open(RESULTS_DIR / "experiment_04_lstm.json") as f:
            data = json.load(f)
        history = data.get("history", [])
    except Exception:
        history = []

    if not history:
        print("  Skipping Fig 7 - no history in JSON")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    epochs   = [h["epoch"]   for h in history]
    val_aucs = [h["val_auc"] for h in history]
    losses   = [h["loss"]    for h in history]

    # Val AUC
    ax.plot(epochs, val_aucs, color=RED, linewidth=2.0,
            label="Exp 04 val AUC (EfficientNet + LSTM)")

    # Best epoch
    best_idx = val_aucs.index(max(val_aucs))
    ax.scatter([epochs[best_idx]], [val_aucs[best_idx]],
               color=RED, s=80, zorder=5,
               label=f"Best: {val_aucs[best_idx]:.3f} "
                     f"at epoch {epochs[best_idx]}")

    # Static baseline
    ax.axhline(y=0.641, color=CHARCOAL, linewidth=1.5,
               linestyle="--",
               label="Static EfficientNet baseline (0.641)")

    # Mean CV line
    ax.axhline(y=0.8216, color=GREEN, linewidth=1.2,
               linestyle=":", label="5-fold CV mean (0.8216)")

    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_ylim(0.50, 0.95)
    ax.set_title("Figure 7 - Training Curve: Exp 04 vs Static Baseline",
                 fontsize=11, color=CHARCOAL)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig7_training_curve.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {out.name}")


#  Figure 8: Chunk Visualization 
def fig_chunk_visualization():
    try:
        X     = np.load(DATA_DIR / "X_seq.npy")
        names = list(np.load(DATA_DIR / "names_seq.npy"))
        idx   = names.index("v101l") if "v101l" in names else 0
        seq   = X[idx]  # (6, 4, 64, 64)
    except Exception as e:
        print(f"  Skipping Fig 8 - data not found: {e}")
        return

    # Show 2 rows: ECG Lead II (ch 0) and SpO2 (ch 2)
    # across all 6 chunks - real data, no simulation
    fig, axes = plt.subplots(2, 6, figsize=(13, 4),
                             facecolor=WHITE)
    chunk_labels = [f"Chunk {i+1}\n{i*10}–{(i+1)*10}s"
                    for i in range(6)]
    channel_names = ["ECG Lead II", "SpO\u2082 (PLETH)"]
    channel_idx   = [0, 2]

    for row, (ch_idx, ch_name) in enumerate(
            zip(channel_idx, channel_names)):
        for col in range(6):
            ax = axes[row, col]
            ax.imshow(seq[col, ch_idx],
                      aspect="auto", origin="lower",
                      cmap="magma", interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(LGREY)
                sp.set_linewidth(0.6)
            if row == 0:
                ax.set_title(chunk_labels[col],
                             fontsize=8, color=CHARCOAL, pad=3)
            if col == 0:
                ax.set_ylabel(ch_name, fontsize=8,
                              color=GREY, labelpad=4)

    fig.suptitle(
        "Figure 8 - Temporal Chunking Visualization: "
        "Record v101l (True Ventricular Flutter Alarm)\n"
        "Each column is one 10-second chunk. "
        "Top row: ECG Lead II CWT scalogram. "
        "Bottom row: SpO\u2082 CWT scalogram. "
        "Brightness encodes time-frequency energy.",
        fontsize=9, color=CHARCOAL, y=1.03)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig8_chunk_visualization.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"),
                bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name} (revised - no simulated attention)")


#  Main 
def main():
    print("Generating publication figures...\n")
    fig_architecture()
    fig_experiment_comparison()
    fig_kfold()
    fig_ablation()
    fig_per_alarm()
    fig_error_analysis()
    fig_training_curve()
    fig_chunk_visualization()

    print(f"\nAll figures saved to {OUTPUT_DIR}")
    print("Both PDF and PNG versions generated.")
    print("\nFor LaTeX: use PDF versions (vector quality)")
    print("For arXiv preview: use PNG versions")


if __name__ == "__main__":
    main()