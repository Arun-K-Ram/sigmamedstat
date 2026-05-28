"""
SigmaMedStat - Calibration Analysis
Evaluates whether the model's confidence scores
are reliable - not just whether predictions are correct.

A well-calibrated model where it says 80% confidence
should be right 80% of the time.

Generates:
  1. Reliability diagram (calibration curve)
  2. Expected Calibration Error (ECE)
  3. Brier Score
  4. Confidence-stratified performance
  5. Calibration comparison across all 3 architectures
"""

import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

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


def compute_ece(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error.
    Measures average gap between confidence and accuracy.
    Perfect calibration = ECE of 0.
    """
    bins     = np.linspace(0, 1, n_bins + 1)
    ece      = 0.0
    bin_data = []

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            bin_data.append(None)
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        bin_n    = mask.sum()
        ece     += (bin_n / len(y_true)) * abs(bin_acc - bin_conf)
        bin_data.append({
            "bin_center": round(float((bins[i]+bins[i+1])/2), 2),
            "accuracy":   round(float(bin_acc), 3),
            "confidence": round(float(bin_conf), 3),
            "count":      int(bin_n),
            "gap":        round(float(abs(bin_acc - bin_conf)), 3),
        })

    return round(float(ece), 4), bin_data


def confidence_stratified(y_true, y_prob, n_quartiles=4):
    """
    Performance broken down by confidence quartile.
    Shows whether high-confidence predictions are more reliable.
    """
    quartiles   = np.percentile(
        np.maximum(y_prob, 1-y_prob),
        np.linspace(0, 100, n_quartiles+1))
    conf_scores = np.maximum(y_prob, 1-y_prob)
    results     = []

    for i in range(n_quartiles):
        mask  = (conf_scores >= quartiles[i]) & \
                (conf_scores < quartiles[i+1])
        if mask.sum() == 0:
            continue
        preds = (y_prob[mask] >= 0.5).astype(int)
        acc   = (preds == y_true[mask]).mean()
        results.append({
            "quartile":    i + 1,
            "conf_range":  f"{quartiles[i]:.2f}–{quartiles[i+1]:.2f}",
            "n":           int(mask.sum()),
            "accuracy":    round(float(acc), 3),
            "mean_conf":   round(float(conf_scores[mask].mean()), 3),
        })

    return results


def load_model_data(json_path, model_name):
    """Load OOF predictions from a results JSON."""
    with open(json_path) as f:
        data = json.load(f)
    ea = data.get("error_analysis", data)
    probs  = np.array(ea.get("oof_probs", []))
    labels = np.array(ea.get("oof_labels", []), dtype=int)
    if len(probs) == 0:
        probs  = np.array(data.get("oof_probs", []))
        labels = np.array(data.get("oof_labels", []), dtype=int)
    return probs, labels, model_name


def main():
    print("SigmaMedStat - Calibration Analysis")
    print("=" * 55)

    # Load all three models
    models_data = []

    # Exp 04 LSTM
    try:
        p, l, n = load_model_data(
            RESULTS_DIR / "experiment_04_kfold.json", "LSTM (Exp 04)")
        models_data.append((p, l, n))
        print(f"✓ Loaded Exp 04 LSTM: {len(p)} samples")
    except Exception as e:
        print(f"✗ Exp 04: {e}")

    # Exp 05 BiLSTM
    try:
        p, l, n = load_model_data(
            RESULTS_DIR / "experiment_05_bilstm.json", "BiLSTM (Exp 05)")
        models_data.append((p, l, n))
        print(f"✓ Loaded Exp 05 BiLSTM: {len(p)} samples")
    except Exception as e:
        print(f"✗ Exp 05: {e}")

    # Exp 06 Transformer
    try:
        p, l, n = load_model_data(
            RESULTS_DIR / "experiment_06_transformer.json",
            "Transformer (Exp 06)")
        models_data.append((p, l, n))
        print(f"✓ Loaded Exp 06 Transformer: {len(p)} samples")
    except Exception as e:
        print(f"✗ Exp 06: {e}")

    print()

    #  Compute calibration metrics 
    all_results = {}
    for y_prob, y_true, name in models_data:
        ece, bin_data = compute_ece(y_true, y_prob)
        brier         = brier_score_loss(y_true, y_prob)
        conf_strat    = confidence_stratified(y_true, y_prob)

        print(f"{name}:")
        print(f"  ECE:         {ece:.4f} "
              f"(0=perfect, lower is better)")
        print(f"  Brier Score: {brier:.4f} "
              f"(0=perfect, lower is better)")
        print(f"  Confidence-stratified accuracy:")
        for q in conf_strat:
            print(f"    Q{q['quartile']} "
                  f"conf={q['conf_range']} "
                  f"n={q['n']:>3}  "
                  f"acc={q['accuracy']:.3f}")
        print()

        all_results[name] = {
            "ece":          ece,
            "brier_score":  round(float(brier), 4),
            "bin_data":     bin_data,
            "conf_stratified": conf_strat,
        }

    #  Figure 1: Reliability diagrams 
    colors_plot = [RED, BLUE, GREEN]
    fig, axes   = plt.subplots(1, len(models_data),
                               figsize=(5*len(models_data), 5))
    if len(models_data) == 1:
        axes = [axes]

    for ax, (y_prob, y_true, name), color in zip(
            axes, models_data, colors_plot):

        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="uniform")

        ax.plot([0, 1], [0, 1], color=GREY, lw=1.5,
                linestyle="--", label="Perfect calibration")
        ax.plot(mean_pred, frac_pos, color=color,
                lw=2, marker="o", markersize=5,
                label=name.split("(")[0].strip())

        ece = all_results[name]["ece"]
        brier = all_results[name]["brier_score"]

        ax.fill_between(mean_pred, mean_pred, frac_pos,
                        alpha=0.15, color=color)

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"{name.split('(')[0].strip()}\n"
                     f"ECE={ece:.3f}  Brier={brier:.3f}",
                     fontsize=9, color=CHARCOAL)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

    fig.suptitle("Figure 12 - Reliability Diagrams\n"
                 "Calibration comparison across architectures.\n"
                 "Points above diagonal: underconfident. "
                 "Points below: overconfident.",
                 fontsize=10, color=CHARCOAL, y=1.05)
    plt.tight_layout()

    out = OUTPUT_DIR / "fig12_reliability_diagrams.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"Figure saved → {out}")

    #  Figure 2: Confidence-stratified performance 
    fig2, ax2 = plt.subplots(figsize=(9, 5))

    x_base  = np.arange(4)
    width   = 0.25
    offsets = [-0.25, 0, 0.25]

    for (y_prob, y_true, name), color, offset in zip(
            models_data, colors_plot, offsets):
        conf_strat = all_results[name]["conf_stratified"]
        accs = [q["accuracy"] for q in conf_strat]
        xs   = x_base[:len(accs)] + offset
        ax2.bar(xs, accs, width=width,
                color=color, alpha=0.8,
                edgecolor=WHITE, label=name.split("(")[0].strip())
        for x, acc in zip(xs, accs):
            ax2.text(x, acc + 0.005, f"{acc:.2f}",
                     ha="center", va="bottom",
                     fontsize=7.5, color=CHARCOAL)

    ax2.axhline(y=0.5, color=GREY, lw=1,
                linestyle="--", alpha=0.5,
                label="Random baseline")
    ax2.set_xticks(x_base)
    ax2.set_xticklabels([
        "Q1\n(lowest conf)", "Q2", "Q3",
        "Q4\n(highest conf)"])
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0.4, 1.0])
    ax2.set_title("Figure 13 - Confidence-Stratified Performance\n"
                  "Accuracy by confidence quartile across architectures",
                  fontsize=10, color=CHARCOAL)
    ax2.legend(fontsize=8.5, framealpha=0.9)
    plt.tight_layout()

    out2 = OUTPUT_DIR / "fig13_confidence_stratified.pdf"
    plt.savefig(out2)
    plt.savefig(str(out2).replace(".pdf", ".png"))
    plt.close()
    print(f"Figure saved → {out2}")

    #  Summary table
    print("\n" + "=" * 55)
    print("CALIBRATION SUMMARY")
    print("=" * 55)
    print(f"{'Model':<25} {'ECE':>6} {'Brier':>7}")
    print("-" * 40)
    for name, res in all_results.items():
        short = name.split("(")[0].strip()
        print(f"{short:<25} {res['ece']:>6.4f} "
              f"{res['brier_score']:>7.4f}")
    print("=" * 55)
    print("ECE: Expected Calibration Error "
          "(lower = better calibrated)")
    print("Brier: Brier Score "
          "(lower = more accurate probability estimates)")

    # Save JSON
    out_json = RESULTS_DIR / "calibration_analysis.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_json}")


if __name__ == "__main__":
    main()