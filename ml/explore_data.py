import wfdb
import numpy as np
from pathlib import Path

DATA_DIR = Path("../backend/data/physionet/training")

# Read one record
record_name = str(DATA_DIR / "v100s")
record = wfdb.rdrecord(record_name)

print(f"Record: v100s (ventricular alarm - FALSE alarm)")
print(f"Signals: {record.sig_name}")
print(f"Sampling rate: {record.fs} Hz")
print(f"Duration: {record.sig_len / record.fs:.1f} seconds")
print(f"Shape: {record.p_signal.shape}")
print(f"\nSignal channels:")
for i, name in enumerate(record.sig_name):
    sig = record.p_signal[:, i]
    print(f"  {name}: mean={np.nanmean(sig):.2f}, std={np.nanstd(sig):.2f}")


DATA_DIR = Path("../backend/data/physionet/training")
files = list(DATA_DIR.glob("*.hea"))

true_alarms = [f for f in files if f.stem.endswith('l')]
false_alarms = [f for f in files if f.stem.endswith('s')]

# By type
for prefix in ['a', 'b', 't', 'v']:
    true_count = len([f for f in true_alarms if f.stem.startswith(prefix)])
    false_count = len([f for f in false_alarms if f.stem.startswith(prefix)])
    print(f"{prefix}: {true_count} true alarms, {false_count} false alarms")

print(f"\nTotal: {len(true_alarms)} true, {len(false_alarms)} false")
print(f"False alarm rate: {len(false_alarms)/(len(true_alarms)+len(false_alarms))*100:.1f}%")