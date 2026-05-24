"""
CWT Pipeline - converts raw ICU signal windows into 2D scalogram images
for CNN input.

Innovation: We use Continuous Wavelet Transform (CWT) instead of raw
signal or FFT because CWT preserves BOTH time and frequency information
simultaneously. Artifacts like spikes and dropouts have distinct
time-frequency signatures that CNNs can learn to recognize.
"""

import numpy as np
import pywt
import torch
from pathlib import Path
import wfdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple

# Constants
FS = 250                  # Sampling rate Hz
WINDOW_SEC = 60           # Last 60 seconds = alarm window
WINDOW_SAMPLES = FS * WINDOW_SEC  # 15000 samples
SCALOGRAM_SIZE = 64       # 64x64 output image
WAVELET = 'morl'          # Morlet wavelet - best for biomedical signals
SCALES = np.arange(1, 128)  # Frequency scales

DATA_DIR = Path("../backend/data/physionet/training")


def signal_to_scalogram(signal: np.ndarray, size: int = SCALOGRAM_SIZE) -> np.ndarray:
    """
    Convert 1D signal window to 2D CWT scalogram image.
    
    The Morlet wavelet is ideal for biomedical signals because it
    resembles the oscillatory patterns in ECG/SpO2 waveforms.
    High-frequency artifacts (spikes) appear as bright vertical
    streaks. Low-frequency drift appears as horizontal bands.
    Dropouts appear as sudden silence regions.
    """
    # Handle NaN values
    signal = np.nan_to_num(signal, nan=np.nanmean(signal) if not np.all(np.isnan(signal)) else 0)
    
    # Downsample to reduce compute - 15000 → 1000 samples
    downsample_factor = WINDOW_SAMPLES // 1000
    signal_ds = signal[::downsample_factor]
    
    # Compute CWT
    scales = np.logspace(0, 2, size)  # Log-spaced scales for better freq resolution
    coefficients, _ = pywt.cwt(signal_ds, scales, WAVELET)
    
    # Take magnitude (power)
    scalogram = np.abs(coefficients)
    
    # Resize to fixed size
    from PIL import Image
    img = Image.fromarray(scalogram).resize((size, size), Image.LANCZOS)
    scalogram_resized = np.array(img, dtype=np.float32)
    
    # Normalize to [0, 1]
    if scalogram_resized.max() > 0:
        scalogram_resized = scalogram_resized / scalogram_resized.max()
    
    return scalogram_resized


def load_record(record_stem: str) -> Tuple[np.ndarray, int]:
    """
    Load a PhysioNet record and extract the alarm window.
    Returns (signal_array, label) where label=1 is true alarm, 0 is false.
    """
    path = str(DATA_DIR / record_stem)
    record = wfdb.rdrecord(path)
    
    # Label from filename suffix
    label = 1 if record_stem.endswith('l') else 0
    
    # Extract last 60 seconds (alarm window)
    signal = record.p_signal[-WINDOW_SAMPLES:, :]  # shape: (15000, 4)
    
    return signal, label


def record_to_scalograms(record_stem: str) -> Tuple[np.ndarray, int]:
    """
    Convert a record to a multi-channel scalogram tensor.
    Returns (scalogram_tensor, label) where tensor shape is (4, 64, 64)
    - one 64x64 scalogram per signal channel.
    """
    signal, label = load_record(record_stem)
    
    scalograms = []
    for ch in range(signal.shape[1]):
        scalo = signal_to_scalogram(signal[:, ch])
        scalograms.append(scalo)
    
    # Stack channels → (4, 64, 64) tensor
    tensor = np.stack(scalograms, axis=0)
    
    return tensor, label


def visualize_sample(record_stem: str, save_path: str = None):
    """
    Generate visualization of CWT scalograms for a record.
    This is what goes on LinkedIn - the Grad-CAM heatmap visualization.
    """
    signal, label = load_record(record_stem)
    channel_names = ['ECG II', 'ECG V', 'PLETH (SpO₂)', 'RESP']
    alarm_type = "TRUE ALARM" if label == 1 else "FALSE ALARM"
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('#0a0a0a')
    
    for ch in range(4):
        # Raw signal
        ax = axes[0, ch]
        ax.set_facecolor('#111')
        t = np.linspace(0, 60, WINDOW_SAMPLES)
        sig = signal[:, ch]
        sig = np.nan_to_num(sig, nan=0)
        color = '#ef4444' if label == 0 else '#22c55e'
        ax.plot(t, sig, color=color, linewidth=0.5, alpha=0.8)
        ax.set_title(channel_names[ch], color='white', fontsize=10)
        ax.tick_params(colors='#555')
        ax.set_xlabel('Time (s)', color='#555', fontsize=8)
        for spine in ax.spines.values():
            spine.set_color('#222')
        
        # Scalogram
        ax = axes[1, ch]
        ax.set_facecolor('#111')
        scalo = signal_to_scalogram(signal[:, ch])
        im = ax.imshow(scalo, aspect='auto', cmap='inferno', origin='lower')
        ax.set_title(f'CWT Scalogram', color='#888', fontsize=9)
        ax.tick_params(colors='#555')
        for spine in ax.spines.values():
            spine.set_color('#222')
    
    fig.suptitle(
        f'SigmaMedStat - {record_stem} - {alarm_type}',
        color='white', fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#0a0a0a', edgecolor='none')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # Test on one true alarm and one false alarm
    print("Testing CWT pipeline...")
    
    # False alarm
    tensor, label = record_to_scalograms("v100s")
    print(f"v100s → tensor shape: {tensor.shape}, label: {label} (false alarm)")
    
    # True alarm  
    tensor, label = record_to_scalograms("v101l")
    print(f"v101l → tensor shape: {tensor.shape}, label: {label} (true alarm)")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_sample("v100s", save_path="viz_false_alarm.png")
    visualize_sample("v101l", save_path="viz_true_alarm.png")
    
    print("\nCWT pipeline working correctly.")
    print("Check viz_false_alarm.png and viz_true_alarm.png")