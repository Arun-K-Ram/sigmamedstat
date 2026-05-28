"""
SigmaMedStat - Statistical Significance Between Architectures
DeLong test comparing:
  Exp 04 LSTM vs Exp 05 BiLSTM
  Exp 04 LSTM vs Exp 06 Transformer
  Exp 05 BiLSTM vs Exp 06 Transformer
"""

import numpy as np
import json
import scipy.stats as stats
from pathlib import Path
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path("results")


def delong_compare(y_true, probs_a, probs_b):
    """DeLong test comparing two AUC values."""
    def placement(y_true, probs):
        pos = probs[y_true == 1]
        neg = probs[y_true == 0]
        V10 = np.array([np.mean(p > neg) for p in pos])
        V01 = np.array([np.mean(p < pos) for p in neg])
        auc = roc_auc_score(y_true, probs)
        S10 = np.var(V10, ddof=1) / len(pos)
        S01 = np.var(V01, ddof=1) / len(neg)
        return auc, S10 + S01

    auc_a, var_a = placement(y_true, probs_a)
    auc_b, var_b = placement(y_true, probs_b)
    se   = np.sqrt(var_a + var_b)
    z    = (auc_a - auc_b) / se if se > 0 else 0
    p    = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, z, p


def load_probs(json_path):
    with open(json_path) as f:
        d = json.load(f)
    ea = d.get("error_analysis", d)
    probs  = np.array(ea.get("oof_probs",  d.get("oof_probs", [])))
    labels = np.array(ea.get("oof_labels", d.get("oof_labels", [])),
                      dtype=int)
    return probs, labels


def main():
    print("Architecture Statistical Significance Testing")
    print("=" * 55)

    lstm_p,  lstm_y  = load_probs(
        RESULTS_DIR / "experiment_04_kfold.json")
    bilstm_p, bilstm_y = load_probs(
        RESULTS_DIR / "experiment_05_bilstm.json")
    trans_p,  trans_y  = load_probs(
        RESULTS_DIR / "experiment_06_transformer.json")

    comparisons = [
        ("LSTM vs BiLSTM",      lstm_p,   bilstm_p, lstm_y),
        ("LSTM vs Transformer", lstm_p,   trans_p,  lstm_y),
        ("BiLSTM vs Transformer", bilstm_p, trans_p, bilstm_y),
    ]

    results = {}
    print(f"\n{'Comparison':<28} {'AUC_A':>7} {'AUC_B':>7} "
          f"{'z':>7} {'p':>8} {'Sig':>5}")
    print("-" * 65)

    for name, pa, pb, y in comparisons:
        auc_a, auc_b, z, p = delong_compare(y, pa, pb)
        sig = "YES" if p < 0.05 else "NO"
        print(f"{name:<28} {auc_a:>7.4f} {auc_b:>7.4f} "
              f"{z:>7.3f} {p:>8.4f} {sig:>5}")
        results[name] = {
            "auc_a": round(float(auc_a), 4),
            "auc_b": round(float(auc_b), 4),
            "z":     round(float(z), 4),
            "p":     round(float(p), 6),
            "significant": bool(p < 0.05),
        }

    # Bootstrap CI on differences
    print("\nBootstrap 95% CI on AUC differences (1000 iter):")
    rng = np.random.default_rng(42)
    n   = len(lstm_y)

    for name, pa, pb, y in comparisons:
        diffs = []
        for _ in range(1000):
            idx   = rng.integers(0, n, size=n)
            auc_a = roc_auc_score(y[idx], pa[idx])
            auc_b = roc_auc_score(y[idx], pb[idx])
            diffs.append(auc_a - auc_b)
        ci_low  = np.percentile(diffs, 2.5)
        ci_high = np.percentile(diffs, 97.5)
        print(f"  {name:<28} CI=[{ci_low:+.4f}, {ci_high:+.4f}]  "
              f"Excl zero: {ci_low > 0 or ci_high < 0}")
        results[name]["bootstrap_ci"] = [
            round(float(ci_low), 4),
            round(float(ci_high), 4)]
        results[name]["ci_excludes_zero"] = bool(
            ci_low > 0 or ci_high < 0)

    out = RESULTS_DIR / "architecture_significance.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved → {out}")
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    for name, res in results.items():
        sig_str = "SIGNIFICANT" if res["significant"] \
                  else "not significant"
        print(f"  {name}: p={res['p']:.4f} - {sig_str}")
    print("=" * 55)


if __name__ == "__main__":
    main()