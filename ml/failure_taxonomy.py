"""
SigmaMedStat - Failure Mode Taxonomy
Categorizes the false negatives and false positives
from 5-fold CV into clinically meaningful failure modes.

Categories:
  False Negatives (missed true alarms):
    - Asystole misclassified (flatline overlap)
    - Low signal amplitude (weak arrhythmia signal)
    - Short-duration event (arrhythmia brief, hard to detect)
    - Morphology overlap (rhythm looks like artifact)

  False Positives (false alarm called real):
    - Ventricular Flutter artifact (most common false alarm)
    - Motion artifact pattern
    - High confidence wrong (model certain but wrong)

Uses OOF predictions + record names to categorize.
"""

import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix

RESULTS_DIR   = Path("results")
OUTPUT_DIR    = Path("results/paper_figures")
PHYSIONET_DIR = Path("C:/Users/Arun/Documents/git/crip-x/backend/data/physionet/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WHITE    = "#ffffff"
CHARCOAL = "#2c3e50"
RED      = "#c0392b"
BLUE     = "#1a3a6b"
GREEN    = "#1e8449"
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

ALARM_CODES = {
    "v": "Ventricular Flutter",
    "a": "Asystole",
    "t": "Tachycardia",
    "b": "Bradycardia",
    "f": "Ventricular Fibrillation",
}


def get_signal_stats(record_id):
    """
    Load a record and compute basic signal statistics
    to characterize failure modes.
    """
    try:
        import wfdb
        rec = wfdb.rdrecord(
            str(PHYSIONET_DIR / record_id))
        sig = rec.p_signal
        sig = np.nan_to_num(sig)

        # Use ECG Lead II (channel 0)
        ecg = sig[:, 0]

        # Last 60 seconds
        fs      = 250
        ecg_60  = ecg[-fs*60:] if len(ecg) >= fs*60 else ecg

        stats = {
            "amplitude":  round(float(np.ptp(ecg_60)), 4),
            "std":        round(float(np.std(ecg_60)), 4),
            "flatline":   bool(np.std(ecg_60) < 0.05),
            "low_amp":    bool(np.ptp(ecg_60) < 0.5),
        }
        return stats
    except Exception:
        return None


def categorize_fn(record_id, confidence, alarm_type,
                  sig_stats):
    """Categorize a false negative by likely failure mode."""
    if alarm_type == "Asystole":
        return "Asystole-artifact overlap"
    if sig_stats and sig_stats["flatline"]:
        return "Flatline pattern (artifact-like)"
    if sig_stats and sig_stats["low_amp"]:
        return "Low signal amplitude"
    if confidence > 0.95:
        return "High-confidence morphology overlap"
    return "Temporal pattern not captured"


def categorize_fp(record_id, confidence, alarm_type,
                  sig_stats):
    """Categorize a false positive by likely failure mode."""
    if alarm_type == "Ventricular Flutter":
        return "Ventricular Flutter artifact pattern"
    if alarm_type == "Tachycardia":
        return "Tachycardia rate pattern mimics true alarm"
    if confidence > 0.90:
        return "High-confidence misclassification"
    return "Borderline signal pattern"


def main():
    print("SigmaMedStat - Failure Mode Taxonomy")
    print("=" * 55)

    # Load OOF predictions
    with open(RESULTS_DIR / "experiment_04_kfold.json") as f:
        data = json.load(f)

    ea       = data["error_analysis"]
    y_prob   = np.array(ea["oof_probs"])
    y_true   = np.array(ea["oof_labels"], dtype=int)

    # Load names
    from pathlib import Path as P
    names_path = P("C:/Users/Arun/Documents/git/crip-x"
                   "/backend/data/scalograms_temporal/names_seq.npy")
    names = list(np.load(names_path))

    y_pred = (y_prob >= 0.5).astype(int)

    # Find errors
    fn_idx = np.where((y_pred == 0) & (y_true == 1))[0]
    fp_idx = np.where((y_pred == 1) & (y_true == 0))[0]

    print(f"False negatives: {len(fn_idx)}")
    print(f"False positives: {len(fp_idx)}\n")

    #  Categorize false negatives 
    print("FALSE NEGATIVE TAXONOMY:")
    print("-" * 50)
    fn_categories = {}
    fn_details    = []

    for idx in fn_idx:
        rid        = names[idx]
        alarm_type = ALARM_CODES.get(rid[0], "Unknown")
        confidence = max(y_prob[idx], 1 - y_prob[idx])
        sig_stats  = get_signal_stats(rid)
        category   = categorize_fn(
            rid, confidence, alarm_type, sig_stats)

        fn_categories[category] = \
            fn_categories.get(category, 0) + 1
        fn_details.append({
            "record":     rid,
            "alarm_type": alarm_type,
            "confidence": round(float(confidence), 4),
            "category":   category,
        })

    for cat, count in sorted(fn_categories.items(),
                              key=lambda x: -x[1]):
        pct = count / len(fn_idx) * 100
        print(f"  {cat:<45} {count:>3} ({pct:.1f}%)")

    #  By alarm type 
    print("\nFalse negatives by alarm type:")
    fn_by_type = {}
    for d in fn_details:
        t = d["alarm_type"]
        fn_by_type[t] = fn_by_type.get(t, 0) + 1
    for t, c in sorted(fn_by_type.items(),
                        key=lambda x: -x[1]):
        print(f"  {t:<30} {c:>3}")

    #  Categorize false positives 
    print("\nFALSE POSITIVE TAXONOMY:")
    print("-" * 50)
    fp_categories = {}
    fp_details    = []

    for idx in fp_idx:
        rid        = names[idx]
        alarm_type = ALARM_CODES.get(rid[0], "Unknown")
        confidence = max(y_prob[idx], 1 - y_prob[idx])
        sig_stats  = get_signal_stats(rid)
        category   = categorize_fp(
            rid, confidence, alarm_type, sig_stats)

        fp_categories[category] = \
            fp_categories.get(category, 0) + 1
        fp_details.append({
            "record":     rid,
            "alarm_type": alarm_type,
            "confidence": round(float(confidence), 4),
            "category":   category,
        })

    for cat, count in sorted(fp_categories.items(),
                              key=lambda x: -x[1]):
        pct = count / len(fp_idx) * 100
        print(f"  {cat:<45} {count:>3} ({pct:.1f}%)")

    print("\nFalse positives by alarm type:")
    fp_by_type = {}
    for d in fp_details:
        t = d["alarm_type"]
        fp_by_type[t] = fp_by_type.get(t, 0) + 1
    for t, c in sorted(fp_by_type.items(),
                        key=lambda x: -x[1]):
        print(f"  {t:<30} {c:>3}")

    #  Confidence distribution of errors 
    print("\nHigh-confidence errors (>0.90):")
    hc_fn = [d for d in fn_details if d["confidence"] > 0.90]
    hc_fp = [d for d in fp_details if d["confidence"] > 0.90]
    print(f"  High-conf FN: {len(hc_fn)} "
          f"({len(hc_fn)/len(fn_idx)*100:.1f}% of FN)")
    print(f"  High-conf FP: {len(hc_fp)} "
          f"({len(hc_fp)/len(fp_idx)*100:.1f}% of FP)")

    #  Figure 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # FN taxonomy pie
    fn_labels = list(fn_categories.keys())
    fn_sizes  = list(fn_categories.values())
    fn_colors = [RED, "#e74c3c", "#c0392b",
                 "#a93226", "#922b21"][:len(fn_labels)]
    axes[0].pie(fn_sizes, labels=None,
                colors=fn_colors,
                autopct="%1.0f%%",
                startangle=90,
                pctdistance=0.75,
                textprops={"fontsize": 8})
    axes[0].legend(fn_labels,
                   loc="lower center",
                   bbox_to_anchor=(0.5, -0.35),
                   fontsize=7.5,
                   framealpha=0.9)
    axes[0].set_title(f"(a) False Negative Taxonomy\n"
                      f"({len(fn_idx)} missed true alarms)",
                      fontsize=10, color=CHARCOAL)

    # FP taxonomy pie
    fp_labels = list(fp_categories.keys())
    fp_sizes  = list(fp_categories.values())
    fp_colors = [BLUE, "#2980b9", "#1a5276",
                 "#154360"][:len(fp_labels)]
    axes[1].pie(fp_sizes, labels=None,
                colors=fp_colors,
                autopct="%1.0f%%",
                startangle=90,
                pctdistance=0.75,
                textprops={"fontsize": 8})
    axes[1].legend(fp_labels,
                   loc="lower center",
                   bbox_to_anchor=(0.5, -0.35),
                   fontsize=7.5,
                   framealpha=0.9)
    axes[1].set_title(f"(b) False Positive Taxonomy\n"
                      f"({len(fp_idx)} unnecessary responses)",
                      fontsize=10, color=CHARCOAL)

    fig.suptitle("Figure 15 - Failure Mode Taxonomy\n"
                 "Categorization of classification errors "
                 "by likely clinical cause",
                 fontsize=11, color=CHARCOAL, y=1.05)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig15_failure_taxonomy.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"),
                bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved → {out}")

    # Save results
    results = {
        "fn_total":       int(len(fn_idx)),
        "fp_total":       int(len(fp_idx)),
        "fn_categories":  fn_categories,
        "fp_categories":  fp_categories,
        "fn_by_type":     fn_by_type,
        "fp_by_type":     fp_by_type,
        "fn_details":     fn_details,
        "fp_details":     fp_details,
        "high_conf_fn":   len(hc_fn),
        "high_conf_fp":   len(hc_fp),
    }

    out_json = RESULTS_DIR / "failure_taxonomy.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_json}")

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"  Total errors:     {len(fn_idx)+len(fp_idx)}")
    print(f"  False negatives:  {len(fn_idx)} "
          f"(missed true alarms - dangerous)")
    print(f"  False positives:  {len(fp_idx)} "
          f"(false alarms called real - wasteful)")
    print(f"  High-conf FN:     {len(hc_fn)} "
          f"({len(hc_fn)/len(fn_idx)*100:.1f}%)")
    print(f"  High-conf FP:     {len(hc_fp)} "
          f"({len(hc_fp)/len(fp_idx)*100:.1f}%)")
    print("=" * 55)


if __name__ == "__main__":
    main()