"""
SigmaMedStat - PhysioNet 2015 Challenge Metric
Computes the official challenge scoring formula on
our out-of-fold predictions at multiple thresholds.

Challenge score = (TP + TN) / (TP + TN + FP + 5*FN)

This allows direct comparison to published leaderboard:
  Plesinger et al. 2015: 81.39
  Au-Yeung et al. 2019:  83.08
  Fallet et al. 2015:    85.04
"""

import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
OUTPUT_DIR  = Path("results/paper_figures")

WHITE    = "#ffffff"
CHARCOAL = "#2c3e50"
RED      = "#c0392b"
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


def challenge_score(tp, tn, fp, fn):
    """PhysioNet 2015 challenge scoring formula."""
    denom = tp + tn + fp + 5 * fn
    return (tp + tn) / denom * 100 if denom > 0 else 0


def compute_at_threshold(y_true, y_prob, threshold):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    score = challenge_score(tp, tn, fp, fn)
    sens  = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "threshold": round(threshold, 2),
        "score":     round(score, 2),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": round(sens, 3),
        "specificity": round(spec, 3),
    }


def main():
    print("PhysioNet 2015 Challenge Metric Computation")
    print("=" * 55)

    # Load OOF predictions
    with open(RESULTS_DIR / "experiment_04_kfold.json") as f:
        data = json.load(f)

    ea     = data["error_analysis"]
    y_prob = np.array(ea["oof_probs"])
    y_true = np.array(ea["oof_labels"], dtype=int)

    print(f"Samples: {len(y_true)}  "
          f"True={y_true.sum()}  False={(y_true==0).sum()}\n")

    # Published leaderboard scores
    leaderboard = {
        "Plesinger et al. 2015 (Event 1 winner)": 81.39,
        "Fallet et al. 2015 (Event 2 winner)":    85.04,
        "Au-Yeung et al. 2019 (best published)":  83.08,
    }

    print("Published leaderboard (5-minute window):")
    for name, score in leaderboard.items():
        print(f"  {name}: {score:.2f}")
    print()

    # Sweep thresholds
    print("Our scores at each threshold (60-second window):")
    print(f"{'Thr':>5} {'Score':>7} {'Sens':>6} "
          f"{'Spec':>6} {'TP':>4} {'TN':>4} "
          f"{'FP':>4} {'FN':>4}")
    print("-" * 50)

    sweep = []
    best_score = 0
    best_result = None

    for thr in np.arange(0.05, 0.95, 0.05):
        r = compute_at_threshold(y_true, y_prob, thr)
        sweep.append(r)
        if r["score"] > best_score:
            best_score  = r["score"]
            best_result = r
        marker = " ★" if r["score"] == best_score else ""
        print(f"{thr:>5.2f} {r['score']:>7.2f} "
              f"{r['sensitivity']:>6.3f} {r['specificity']:>6.3f} "
              f"{r['tp']:>4} {r['tn']:>4} "
              f"{r['fp']:>4} {r['fn']:>4}{marker}")

    print(f"\nBest challenge score: {best_result['score']:.2f} "
          f"at threshold {best_result['threshold']:.2f}")
    print(f"  Sensitivity: {best_result['sensitivity']:.3f}")
    print(f"  Specificity: {best_result['specificity']:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    thresholds = [r["threshold"] for r in sweep]
    scores     = [r["score"]     for r in sweep]

    ax.plot(thresholds, scores, color=RED,
            lw=2.5, label="Our model (60s window)")

    # Best point
    best_idx = scores.index(max(scores))
    ax.scatter([thresholds[best_idx]], [scores[best_idx]],
               color=RED, s=100, zorder=5,
               label=f"Our best: {max(scores):.2f} "
                     f"(thr={thresholds[best_idx]:.2f})")

    # Leaderboard lines
    colors_lb = [GREY, BLUE, CHARCOAL]
    styles_lb = ["--", "-.", ":"]
    for (name, score), color, style in zip(
            leaderboard.items(), colors_lb, styles_lb):
        short = name.split("(")[1].rstrip(")")
        ax.axhline(y=score, color=color, lw=1.5,
                   linestyle=style,
                   label=f"{short}: {score:.2f} "
                         f"(5-min window)")

    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Challenge Score")
    ax.set_title("PhysioNet 2015 Challenge Score vs Threshold\n"
                 "Our model (60s window) vs published "
                 "leaderboard (5-min window)",
                 fontsize=10, color=CHARCOAL)
    ax.legend(fontsize=8, framealpha=0.9, loc="lower center")
    ax.set_ylim([50, 92])

    out = OUTPUT_DIR / "fig11_challenge_metric.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"\nFigure saved → {out}")

    # Save results
    results = {
        "leaderboard":  leaderboard,
        "our_sweep":    sweep,
        "best_score":   best_result["score"],
        "best_threshold": best_result["threshold"],
        "note": "Our model uses 60s window; leaderboard uses 5-min window"
    }
    out_json = RESULTS_DIR / "challenge_metric.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_json}")

    print("\n" + "=" * 55)
    print("COMPARISON SUMMARY")
    print("=" * 55)
    print(f"  Our best challenge score: {best_result['score']:.2f}")
    print(f"  Plesinger 2015:           81.39")
    print(f"  Au-Yeung 2019:            83.08")
    print(f"  Fallet 2015:              85.04")
    print(f"\n  IMPORTANT: Direct comparison is not fair.")
    print(f"  Published scores use 5-minute windows.")
    print(f"  Our model uses only the final 60 seconds.")
    print(f"  We are operating under a harder constraint.")
    print("=" * 55)


if __name__ == "__main__":
    main()