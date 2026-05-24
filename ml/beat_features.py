"""
SigmaMedStat - Beat-Level Feature Extraction
Pan-Tompkins QRS detector + per-beat morphology features.

This is the approach used in published PhysioNet Challenge 2015 papers.
Beat-level analysis captures arrhythmia patterns that window statistics miss.

Key insight:
- Ventricular fibrillation → chaotic, irregular beats, no clear QRS
- False alarm → normal beats + sensor artifact on another channel
- Window stats can't distinguish these
- Beat morphology can
"""

import numpy as np
import wfdb
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DATA_DIR     = Path("../backend/data/physionet/training")
FS           = 250
WINDOW_SEC   = 60
WINDOW_SAMP  = FS * WINDOW_SEC   # 15000 samples
BEAT_WIN_MS  = 300               # ms around each beat to analyze
BEAT_WIN_S   = int(BEAT_WIN_MS / 1000 * FS)  # samples


# ── Pan-Tompkins QRS Detector ─────────────────────────────────
def pan_tompkins(ecg: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Simplified Pan-Tompkins QRS detection.
    Returns array of R-peak sample indices.

    Steps:
    1. Bandpass filter (5-15 Hz) - isolates QRS complex
    2. Differentiate - emphasizes steep slopes of QRS
    3. Square - makes all values positive, emphasizes large values
    4. Moving window integration - smooths
    5. Adaptive thresholding - finds peaks
    """
    # 1. Bandpass filter 5-15 Hz
    nyq = fs / 2
    b, a = scipy_signal.butter(2, [5/nyq, 15/nyq], btype='band')
    filtered = scipy_signal.filtfilt(b, a, ecg)

    # 2. Differentiate
    diff = np.diff(filtered, prepend=filtered[0])

    # 3. Square
    squared = diff ** 2

    # 4. Moving window integration (150ms window)
    win = int(0.15 * fs)
    integrated = np.convolve(squared, np.ones(win)/win, mode='same')

    # 5. Adaptive peak detection
    # Use 60% of max as initial threshold
    threshold = 0.6 * np.max(integrated)
    min_distance = int(0.2 * fs)  # min 200ms between beats

    # Find peaks above threshold
    peaks = []
    i = 0
    while i < len(integrated):
        if integrated[i] > threshold:
            # Find local max in neighborhood
            start = max(0, i - min_distance//2)
            end   = min(len(integrated), i + min_distance//2)
            local_max = start + np.argmax(integrated[start:end])
            if not peaks or local_max - peaks[-1] > min_distance:
                peaks.append(local_max)
            i = end
        else:
            i += 1

    return np.array(peaks)


# ── Per-Beat Features ─────────────────────────────────────────
def beat_morphology_features(ecg: np.ndarray,
                              r_peaks: np.ndarray) -> dict:
    """
    Extract morphology features from detected beats.

    Features:
    - RR intervals (heart rate variability)
    - QRS width and amplitude
    - Beat-to-beat variability
    - Morphology consistency across beats
    """
    if len(r_peaks) < 3:
        # Too few beats - return zeros
        return {k: 0.0 for k in [
            'beat_count', 'mean_rr', 'std_rr', 'cv_rr',
            'rmssd', 'pnn50', 'mean_qrs_amp', 'std_qrs_amp',
            'mean_qrs_width', 'beat_corr', 'beat_corr_std',
            'irregularity_score', 'hr_mean', 'hr_std'
        ]}

    # ── RR intervals (time between beats) ───────────────────
    rr = np.diff(r_peaks) / FS * 1000  # convert to ms

    mean_rr  = np.mean(rr)
    std_rr   = np.std(rr)
    cv_rr    = std_rr / (mean_rr + 1e-8)  # coefficient of variation

    # RMSSD - root mean square of successive differences
    # High in AFib and VFib, low in normal rhythm
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))

    # pNN50 - proportion of successive RR intervals differing > 50ms
    pnn50 = np.mean(np.abs(np.diff(rr)) > 50)

    # Heart rate
    hr = 60000 / (rr + 1e-8)  # BPM
    hr_mean = np.mean(hr)
    hr_std  = np.std(hr)

    # ── QRS morphology ───────────────────────────────────────
    half_win = BEAT_WIN_S // 2
    beats = []

    for r in r_peaks:
        start = r - half_win
        end   = r + half_win
        if start >= 0 and end < len(ecg):
            beat = ecg[start:end]
            beats.append(beat)

    if len(beats) < 2:
        mean_qrs_amp   = 0.0
        std_qrs_amp    = 0.0
        mean_qrs_width = 0.0
        beat_corr      = 0.0
        beat_corr_std  = 0.0
    else:
        beats = np.array(beats)

        # QRS amplitude (peak to peak per beat)
        amps  = np.ptp(beats, axis=1)
        mean_qrs_amp = np.mean(amps)
        std_qrs_amp  = np.std(amps)

        # QRS width - width at 50% of amplitude
        widths = []
        for beat in beats:
            b_norm = beat - np.min(beat)
            threshold = 0.5 * np.max(b_norm)
            above = np.where(b_norm > threshold)[0]
            widths.append(len(above) / FS * 1000 if len(above) > 0 else 0)
        mean_qrs_width = np.mean(widths)

        # Beat-to-beat correlation - consistency of morphology
        # Low correlation = irregular/chaotic beats (VFib signature)
        corrs = []
        template = beats[0]
        for beat in beats[1:]:
            if np.std(template) > 0 and np.std(beat) > 0:
                c = np.corrcoef(template, beat)[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
        beat_corr     = np.mean(corrs) if corrs else 0.0
        beat_corr_std = np.std(corrs)  if corrs else 0.0

    # Irregularity score - combines RR variability and morphology
    irregularity_score = cv_rr * (1 - beat_corr + 1e-8)

    return {
        'beat_count':         float(len(r_peaks)),
        'mean_rr':            float(mean_rr),
        'std_rr':             float(std_rr),
        'cv_rr':              float(cv_rr),
        'rmssd':              float(rmssd),
        'pnn50':              float(pnn50),
        'mean_qrs_amp':       float(mean_qrs_amp),
        'std_qrs_amp':        float(std_qrs_amp),
        'mean_qrs_width':     float(mean_qrs_width),
        'beat_corr':          float(beat_corr),
        'beat_corr_std':      float(beat_corr_std),
        'irregularity_score': float(irregularity_score),
        'hr_mean':            float(hr_mean),
        'hr_std':             float(hr_std),
    }


def signal_quality_features(sig: np.ndarray) -> dict:
    """
    Signal quality features - detect artifacts, flatlines, noise.
    These are the same features as V1 but computed per-channel.
    """
    sig = np.nan_to_num(sig, nan=0.0)

    std = np.std(sig)
    flatline = 1.0 if std < 0.05 else 0.0

    # Spike detection via z-score
    if std > 0:
        z = np.abs((sig - np.mean(sig)) / std)
        spike_ratio = float(np.mean(z > 5))
    else:
        spike_ratio = 0.0

    # Dropout detection
    dropout_ratio = float(np.mean(sig == 0))

    # Signal-to-noise ratio estimate
    freqs, psd = scipy_signal.welch(sig, fs=FS,
                                     nperseg=min(256, len(sig)//4))
    signal_band = (freqs >= 0.5) & (freqs <= 40)
    noise_band  = freqs > 40
    snr = float(np.sum(psd[signal_band]) /
                (np.sum(psd[noise_band]) + 1e-10))

    return {
        'std':          float(std),
        'flatline':     flatline,
        'spike_ratio':  spike_ratio,
        'dropout_ratio':dropout_ratio,
        'snr':          snr,
    }


def extract_beat_features(record_stem: str) -> tuple:
    """
    Extract full feature set from one PhysioNet record.
    Combines:
    1. Beat morphology features from ECG channels
    2. Signal quality features from all channels
    3. Cross-channel consistency features
    """
    path   = str(DATA_DIR / record_stem)
    record = wfdb.rdrecord(path)
    label  = 1 if record_stem.endswith('l') else 0

    # Get alarm window (last 60 seconds)
    signals = record.p_signal[-WINDOW_SAMP:, :]

    # Pad to 4 channels
    if signals.shape[1] < 4:
        pad = np.zeros((signals.shape[0], 4 - signals.shape[1]))
        signals = np.hstack([signals, pad])

    # Pad rows if needed
    if signals.shape[0] < WINDOW_SAMP:
        pad = np.zeros((WINDOW_SAMP - signals.shape[0], 4))
        signals = np.vstack([pad, signals])

    signals = np.nan_to_num(signals, nan=0.0)

    ch_names = ['ECG_II', 'ECG_V', 'PLETH', 'RESP']
    all_feats = {}

    # ── Beat features from ECG channels (II and V) ──────────
    for ch_idx, ch_name in enumerate(['ECG_II', 'ECG_V']):
        ecg = signals[:, ch_idx]

        # Detect R-peaks
        r_peaks = pan_tompkins(ecg, fs=FS)

        # Beat morphology features
        beat_feats = beat_morphology_features(ecg, r_peaks)
        for k, v in beat_feats.items():
            all_feats[f'{ch_name}_{k}'] = v if np.isfinite(v) else 0.0

    # ── Signal quality features for all channels ─────────────
    for ch_idx, ch_name in enumerate(ch_names):
        sig = signals[:, ch_idx]
        sq  = signal_quality_features(sig)
        for k, v in sq.items():
            all_feats[f'{ch_name}_{k}'] = v if np.isfinite(v) else 0.0

    # ── Cross-channel consistency ─────────────────────────────
    # Key feature: if ECG looks normal but SpO2 is bad → likely false alarm
    ecg_std   = np.std(signals[:, 0])
    pleth_std = np.std(signals[:, 2])
    resp_std  = np.std(signals[:, 3])

    # Isolation score - how isolated is the bad channel?
    stds = np.array([np.std(signals[:, i]) for i in range(4)])
    all_feats['channel_std_cv'] = float(
        np.std(stds) / (np.mean(stds) + 1e-8)
    )

    # ECG-PLETH coherence (should be correlated in normal rhythm)
    try:
        corr = np.corrcoef(signals[:, 0], signals[:, 2])[0, 1]
        all_feats['ecg_pleth_corr'] = float(corr) if np.isfinite(corr) else 0.0
    except:
        all_feats['ecg_pleth_corr'] = 0.0

    # Alarm type from filename (one-hot encode)
    alarm_type = record_stem[0]  # a, b, t, v
    for t in ['a', 'b', 't', 'v']:
        all_feats[f'alarm_type_{t}'] = 1.0 if alarm_type == t else 0.0

    feat_names = sorted(all_feats.keys())
    feat_vec   = np.array([all_feats[k] for k in feat_names],
                          dtype=np.float32)

    return feat_vec, label, feat_names


def build_beat_dataset(output_dir: Path):
    """Build beat-level feature dataset from all PhysioNet records."""
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [f.stem for f in DATA_DIR.glob("*.hea")]
    print(f"Processing {len(records)} records...")

    X, y, failed = [], [], []
    feat_names   = None
    expected_len = None

    for rec in tqdm(records):
        try:
            feats, label, names = extract_beat_features(rec)
            if feat_names is None:
                feat_names   = names
                expected_len = len(feats)
            if len(feats) != expected_len:
                failed.append((rec, f"wrong length: {len(feats)}"))
                continue
            X.append(feats)
            y.append(label)
        except Exception as e:
            failed.append((rec, str(e)))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nDataset: {X.shape}")
    print(f"Labels: {y.sum()} true / {(1-y).sum()} false")
    print(f"Failed: {len(failed)}")
    if failed[:3]:
        for r, e in failed[:3]:
            print(f"  {r}: {e}")

    np.save(output_dir / "X_beat.npy", X)
    np.save(output_dir / "y_beat.npy", y)
    with open(output_dir / "beat_feature_names.txt", "w") as f:
        for name in feat_names:
            f.write(name + "\n")

    print(f"\nFeatures per sample: {X.shape[1]}")
    print(f"Saved to {output_dir}")
    return X, y, feat_names


if __name__ == "__main__":
    # Debug single record
    record_stem = "v100s"  # known false alarm
    path = str(DATA_DIR / record_stem)
    record = wfdb.rdrecord(path)
    ecg = record.p_signal[-WINDOW_SAMP:, 0]
    
    r_peaks = pan_tompkins(ecg)
    print(f"Record: {record_stem}")
    print(f"Signal length: {len(ecg)} samples = {len(ecg)/FS:.0f} seconds")
    print(f"R-peaks detected: {len(r_peaks)}")
    print(f"Expected ~60-100 beats in 60 seconds")
    print(f"Mean RR interval: {np.mean(np.diff(r_peaks))/FS*1000:.0f}ms")
    print(f"Expected ~600-1000ms for normal HR")