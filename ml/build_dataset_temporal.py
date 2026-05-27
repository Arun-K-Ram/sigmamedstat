"""
SigmaMedStat - Temporal Dataset Builder
Splits each 60-second record into 6 x 10-second chunks.
Generates CWT scalogram for each chunk.
Output shape: (N, 6, 4, 64, 64)

Each record becomes a sequence of 6 scalograms.
The LSTM will learn patterns across these chunks over time.
"""

import numpy as np
import pywt
import wfdb
from pathlib import Path
from tqdm import tqdm

#  Paths 
PHYSIONET_DIR = Path("C:/Users/Arun/Documents/git/crip-x/backend/data/physionet/training")
RECORDS_FILE  = PHYSIONET_DIR / "RECORDS"
OUTPUT_DIR    = Path("C:/Users/Arun/Documents/git/crip-x/backend/data/scalograms_temporal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#  Config 
FS          = 250          # sampling rate Hz
TOTAL_SEC   = 60           # total window seconds
CHUNK_SEC   = 10           # each chunk seconds
N_CHUNKS    = TOTAL_SEC // CHUNK_SEC   # 6 chunks
SAMPLES     = FS * TOTAL_SEC           # 15000 total
CHUNK_SAMP  = FS * CHUNK_SEC           # 2500 per chunk
N_CHANNELS  = 4
SCALE_SIZE  = 64
WAVELET     = "morl"

# CWT scales - same as Experiment 01
SCALES = np.geomspace(1, 128, num=SCALE_SIZE)


def cwt_scalogram(signal: np.ndarray) -> np.ndarray:
    """Convert 1D signal to (64, 64) CWT scalogram."""
    # Ensure signal is exactly CHUNK_SAMP length
    if len(signal) < CHUNK_SAMP:
        signal = np.pad(signal, (0, CHUNK_SAMP - len(signal)))
    else:
        signal = signal[:CHUNK_SAMP]

    # Downsample to 64 points for the time axis
    indices = np.linspace(0, len(signal) - 1, SCALE_SIZE, dtype=int)
    signal_ds = signal[indices]

    # CWT
    coeffs, _ = pywt.cwt(signal_ds, SCALES, WAVELET)
    scalogram = np.abs(coeffs)  # (64, 64)

    # Normalize to [0, 1]
    s_min, s_max = scalogram.min(), scalogram.max()
    if s_max > s_min:
        scalogram = (scalogram - s_min) / (s_max - s_min)

    return scalogram.astype(np.float32)


def load_record(record_id: str):
    """
    Load raw signal from .mat file.
    Returns (15000, n_channels) or None if can't load.
    """
    path = str(PHYSIONET_DIR / record_id)
    try:
        rec = wfdb.rdrecord(path)
        sig = rec.p_signal  # (samples, channels)
        sig = np.nan_to_num(sig, nan=0.0)
        return sig, rec.sig_name
    except Exception as e:
        return None, None


def get_label(record_id: str, labels: dict) -> int:
    """Return 1 for true alarm, 0 for false alarm."""
    return labels.get(record_id, None)


def load_labels() -> dict:
    """
    Load ground truth labels from .hea files.
    #True alarm  → 1
    #False alarm → 0
    """
    labels = {}
    record_ids = []
    with open(RECORDS_FILE) as f:
        for line in f:
            rid = line.strip()
            if rid:
                record_ids.append(rid)

    for rid in record_ids:
        hea_path = PHYSIONET_DIR / f"{rid}.hea"
        try:
            with open(hea_path) as f:
                content = f.read()
            if "#True alarm" in content:
                labels[rid] = 1
            elif "#False alarm" in content:
                labels[rid] = 0
            # skip if neither found
        except Exception:
            continue

    return labels

def process_record(record_id: str, sig: np.ndarray, sig_names: list):
    """
    Split signal into 6 chunks, compute CWT per chunk per channel.
    Returns (6, 4, 64, 64) or None if not enough channels.
    """
    n_samples, n_ch = sig.shape

    # Need at least 4 channels
    if n_ch < N_CHANNELS:
        return None

    # Take last SAMPLES samples (60 seconds before alarm)
    if n_samples >= SAMPLES:
        sig = sig[-SAMPLES:, :N_CHANNELS]
    else:
        # Pad at the start if shorter
        pad = np.zeros((SAMPLES - n_samples, N_CHANNELS), dtype=np.float32)
        sig = np.vstack([pad, sig[:, :N_CHANNELS]])

    # Split into 6 chunks: (6, CHUNK_SAMP, 4)
    chunks = np.array_split(sig, N_CHUNKS, axis=0)

    sequence = []
    for chunk in chunks:
        # chunk shape: (CHUNK_SAMP, 4)
        chunk_scalograms = []
        for ch in range(N_CHANNELS):
            scalo = cwt_scalogram(chunk[:, ch])  # (64, 64)
            chunk_scalograms.append(scalo)
        # Stack channels: (4, 64, 64)
        chunk_arr = np.stack(chunk_scalograms, axis=0)
        sequence.append(chunk_arr)

    # Stack chunks: (6, 4, 64, 64)
    return np.stack(sequence, axis=0)


def main():
    print("SigmaMedStat - Temporal Dataset Builder")
    print(f"Config: {N_CHUNKS} chunks × {CHUNK_SEC}s each · {SCALE_SIZE}×{SCALE_SIZE} scalograms")
    print(f"Output shape per record: ({N_CHUNKS}, {N_CHANNELS}, {SCALE_SIZE}, {SCALE_SIZE})")
    print()

    # Load labels
    labels = load_labels()
    print(f"Loaded {len(labels)} record labels from RECORDS file")

    # Load record list
    record_ids = list(labels.keys())
    print(f"Processing {len(record_ids)} records...\n")

    X_list     = []
    y_list     = []
    names_list = []
    skipped    = 0

    for record_id in tqdm(record_ids, desc="Building temporal dataset"):
        label = labels[record_id]

        # Load raw signal
        sig, sig_names = load_record(record_id)
        if sig is None:
            skipped += 1
            continue

        # Process into sequence of scalograms
        sequence = process_record(record_id, sig, sig_names)
        if sequence is None:
            skipped += 1
            continue

        X_list.append(sequence)
        y_list.append(label)
        names_list.append(record_id)

    print(f"\nDone. {len(X_list)} records processed, {skipped} skipped.")

    # Stack and save
    X     = np.stack(X_list, axis=0).astype(np.float32)
    y     = np.array(y_list, dtype=np.int64)
    names = np.array(names_list)

    print(f"\nFinal shapes:")
    print(f"  X:     {X.shape}  (records, chunks, channels, H, W)")
    print(f"  y:     {y.shape}")
    print(f"  names: {names.shape}")
    print(f"\nClass balance: {y.sum()} true alarms, {(y==0).sum()} false alarms")

    np.save(OUTPUT_DIR / "X_seq.npy",     X)
    np.save(OUTPUT_DIR / "y_seq.npy",     y)
    np.save(OUTPUT_DIR / "names_seq.npy", names)

    print(f"\nSaved to {OUTPUT_DIR}")
    print("  X_seq.npy")
    print("  y_seq.npy")
    print("  names_seq.npy")


if __name__ == "__main__":
    main()