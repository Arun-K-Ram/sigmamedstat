"""
Hand-crafted signal feature extraction for ICU alarm classification.

Features are computed from raw waveforms - not scalograms.
This is the approach that works best on small medical datasets
and is used in most published work on PhysioNet Challenge 2015.

Feature categories:
1. Time-domain: amplitude, energy, zero crossings, entropy
2. Frequency-domain: dominant frequency, spectral entropy, band power
3. Cross-channel: correlation between ECG, SpO2, RESP
4. Non-linear: sample entropy, Hjorth parameters
"""

import numpy as np
import wfdb
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("../backend/data/physionet/training")
FS = 250
WINDOW_SEC = 60
WINDOW_SAMPLES = FS * WINDOW_SEC  # 15000 samples


def safe_divide(a, b, default=0.0):
    return a / b if b != 0 else default


def time_domain_features(sig: np.ndarray) -> dict:
    """Extract time-domain features from a signal window."""
    sig = np.nan_to_num(sig, nan=np.nanmean(sig) if not np.all(np.isnan(sig)) else 0)

    mean     = np.mean(sig)
    std      = np.std(sig)
    rms      = np.sqrt(np.mean(sig**2))
    peak     = np.max(np.abs(sig))
    p2p      = np.ptp(sig)  # peak to peak
    skew     = float(np.mean(((sig - mean) / (std + 1e-8))**3))
    kurt     = float(np.mean(((sig - mean) / (std + 1e-8))**4))

    # Zero crossing rate
    zcr = np.sum(np.diff(np.sign(sig - mean)) != 0) / len(sig)

    # Signal energy
    energy = np.sum(sig**2) / len(sig)

    # Line length (sum of absolute differences - captures complexity)
    line_length = np.sum(np.abs(np.diff(sig))) / len(sig)

    # Hjorth parameters
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    activity   = np.var(sig)
    mobility   = safe_divide(np.std(d1), np.std(sig))
    complexity = safe_divide(
        safe_divide(np.std(d2), np.std(d1)),
        mobility
    )

    return {
        'mean': mean, 'std': std, 'rms': rms,
        'peak': peak, 'p2p': p2p, 'skew': skew, 'kurt': kurt,
        'zcr': zcr, 'energy': energy, 'line_length': line_length,
        'hjorth_activity': activity,
        'hjorth_mobility': mobility,
        'hjorth_complexity': complexity,
    }


def frequency_domain_features(sig: np.ndarray, fs: int = FS) -> dict:
    """Extract frequency-domain features."""
    sig = np.nan_to_num(sig, nan=0)

    # Power spectral density using Welch method
    freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=min(256, len(sig)//4))

    total_power = np.sum(psd) + 1e-10

    # Dominant frequency
    dom_freq = freqs[np.argmax(psd)]

    # Spectral entropy
    psd_norm = psd / total_power
    spec_entropy = scipy_entropy(psd_norm + 1e-10)

    # Band power (clinically meaningful bands)
    def band_power(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return np.sum(psd[mask]) / total_power

    vlf_power  = band_power(0.0,  0.5)   # Very low frequency
    lf_power   = band_power(0.5,  5.0)   # Low frequency (respiration range)
    hf_power   = band_power(5.0,  15.0)  # High frequency (cardiac range)
    vhf_power  = band_power(15.0, 50.0)  # Very high frequency (noise)

    # Spectral edge frequency (95% of power)
    cumsum = np.cumsum(psd)
    sef95_idx = np.searchsorted(cumsum, 0.95 * cumsum[-1])
    sef95 = freqs[min(sef95_idx, len(freqs)-1)]

    return {
        'dom_freq': dom_freq,
        'spec_entropy': spec_entropy,
        'vlf_power': vlf_power,
        'lf_power': lf_power,
        'hf_power': hf_power,
        'vhf_power': vhf_power,
        'sef95': sef95,
        'total_power': total_power,
    }


def nonlinear_features(sig: np.ndarray) -> dict:
    """Extract non-linear complexity features."""
    sig = np.nan_to_num(sig, nan=0)

    # Sample entropy (measures signal complexity/regularity)
    # Lower = more regular (flatline), Higher = more complex (noise)
    def sample_entropy(x, m=2, r_factor=0.2):
        r = r_factor * np.std(x)
        if r == 0:
            return 0.0
        n = len(x)
        # Use subset for speed
        x = x[:500]
        n = len(x)
        def count_matches(template_len):
            count = 0
            for i in range(n - template_len):
                template = x[i:i+template_len]
                for j in range(i+1, n - template_len):
                    if np.max(np.abs(x[j:j+template_len] - template)) < r:
                        count += 1
            return count
        try:
            cm  = count_matches(m)
            cm1 = count_matches(m+1)
            return -np.log(safe_divide(cm1, cm, default=1e-10) + 1e-10)
        except:
            return 0.0

    samp_en = 0.0  # disabled for speed - too slow on full dataset

    # Coefficient of variation
    cv = safe_divide(np.std(sig), np.abs(np.mean(sig)) + 1e-8)

    # Approximate flatness (std / range)
    rng = np.ptp(sig)
    flatness = 1.0 - safe_divide(np.std(sig), rng + 1e-8)

    return {
        'sample_entropy': samp_en,
        'coeff_variation': cv,
        'flatness': flatness,
    }

def cross_channel_features(signals: np.ndarray) -> dict:
    """
    Extract cross-channel correlation features.
    Key innovation: correlation between SpO2 and ECG tells us
    if a bad reading is isolated (sensor issue) or global (real event).
    """
    # Pad signals to always have 4 channels
    if signals.shape[1] < 4:
        pad = np.zeros((signals.shape[0], 4 - signals.shape[1]))
        signals = np.hstack([signals, pad])
    
    signals = np.nan_to_num(signals, nan=0)
    feats = {}
    ch_names = ['II', 'V', 'PLETH', 'RESP']
    
    for i in range(4):
        for j in range(i+1, 4):
            try:
                corr = np.corrcoef(signals[:, i], signals[:, j])[0, 1]
            except:
                corr = 0.0
            feats[f'corr_{ch_names[i]}_{ch_names[j]}'] = \
                corr if not np.isnan(corr) else 0.0

    stds = [np.std(signals[:, i]) for i in range(4)]
    mean_std = np.mean(stds)
    std_std  = np.std(stds)
    feats['simultaneous_degradation'] = safe_divide(std_std, mean_std + 1e-8)
    
    return feats

def extract_all_features(record_stem: str) -> tuple:
    """
    Extract all features from a PhysioNet record.
    Returns (feature_vector, label, feature_names).
    """
    path   = str(DATA_DIR / record_stem)
    record = wfdb.rdrecord(path)
    label  = 1 if record_stem.endswith('l') else 0

    # Get alarm window (last 60 seconds)
    signals = record.p_signal[-WINDOW_SAMPLES:, :]

    # Pad signals to 4 channels if needed
    if signals.shape[1] < 4:
        pad = np.zeros((signals.shape[0], 4 - signals.shape[1]))
        signals = np.hstack([signals, pad])

    # Pad if needed
    if signals.shape[0] < WINDOW_SAMPLES:
        pad = np.zeros((WINDOW_SAMPLES - signals.shape[0], signals.shape[1]))
        signals = np.vstack([pad, signals])

    ch_names = record.sig_name

    all_feats = {}

    # Always process exactly 4 channels, pad with zeros if missing
    EXPECTED_CHANNELS = ['II', 'V', 'PLETH', 'RESP']

    for ch_idx, ch in enumerate(EXPECTED_CHANNELS):
        if ch_idx < signals.shape[1]:
            sig = signals[:, ch_idx]
        else:
            sig = np.zeros(WINDOW_SAMPLES)

        td = time_domain_features(sig)
        fd = frequency_domain_features(sig)
        nl = nonlinear_features(sig)

        for k, v in {**td, **fd, **nl}.items():
            all_feats[f'{ch}_{k}'] = float(v) if np.isfinite(v) else 0.0

    # Cross-channel features - OUTSIDE the loop
    cc = cross_channel_features(signals)
    all_feats.update(cc)

    feat_names = sorted(all_feats.keys())
    feat_vec   = np.array([all_feats[k] for k in feat_names], dtype=np.float32)

    return feat_vec, label, feat_names


def build_feature_dataset(output_dir: Path):
    """Build full feature dataset from all PhysioNet records."""
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [f.stem for f in DATA_DIR.glob("*.hea")]
    print(f"Processing {len(records)} records...")

    X, y, failed = [], [], []
    feat_names = None

    expected_len = None

    for rec in tqdm(records):
        try:
            feats, label, names = extract_all_features(rec)
            if feat_names is None:
                feat_names = names
                expected_len = len(feats)
            if len(feats) != expected_len:
                failed.append((rec, f"wrong feature length: {len(feats)} vs {expected_len}"))
                continue
            X.append(feats)
            y.append(label)
        except Exception as e:
            failed.append((rec, str(e)))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"\nDataset: {X.shape}")
    print(f"Labels: {y.sum()} true / {(1-y).sum()} false")
    print(f"Failed: {len(failed)}")
    print(f"Features per sample: {X.shape[1]}")

    np.save(output_dir / "X_features.npy", X)
    np.save(output_dir / "y_features.npy", y)

    with open(output_dir / "feature_names.txt", "w") as f:
        for name in feat_names:
            f.write(name + "\n")

    print(f"Saved to {output_dir}")
    return X, y, feat_names


if __name__ == "__main__":
    output_dir = Path("../backend/data/features")
    X, y, names = build_feature_dataset(output_dir)
    print(f"\nSample feature names: {names[:10]}")
    print(f"Feature range: {X.min():.3f} to {X.max():.3f}")