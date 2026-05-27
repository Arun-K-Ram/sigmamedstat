import numpy as np
import wfdb
from pathlib import Path

DATA_DIR = Path("C:/Users/Arun/Documents/git/crip-x/backend/data/physionet/training")
RECORDS  = DATA_DIR / "RECORDS"

alarm_types = {
    "v": "Ventricular Flutter",
    "a": "Asystole",
    "t": "Tachycardia",
    "b": "Bradycardia",
    "f": "Ventricular Fibrillation",
}

counts  = {k: {"true": 0, "false": 0} for k in alarm_types}
skipped = 0

with open(RECORDS) as f:
    record_ids = [l.strip() for l in f if l.strip()]

for rid in record_ids:
    hea_path = DATA_DIR / f"{rid}.hea"
    try:
        content = hea_path.read_text()
        prefix  = rid[0].lower()
        if prefix not in alarm_types:
            skipped += 1
            continue

        # Only count records with 4 channels
        rec = wfdb.rdrecord(str(DATA_DIR / rid))
        if rec.p_signal.shape[1] < 4:
            skipped += 1
            continue

        if "#True alarm" in content:
            counts[prefix]["true"] += 1
        elif "#False alarm" in content:
            counts[prefix]["false"] += 1
        else:
            skipped += 1
    except Exception:
        skipped += 1

print("Per-alarm-type true/false breakdown (498 filtered records)")
print("=" * 55)
print(f"{'Alarm Type':<25} {'Total':>6} {'True':>6} {'False':>6}")
print("-" * 55)
total_t, total_f = 0, 0
for code, name in alarm_types.items():
    t = counts[code]["true"]
    f = counts[code]["false"]
    total_t += t
    total_f += f
    print(f"{name:<25} {t+f:>6} {t:>6} {f:>6}")
print("-" * 55)
print(f"{'Total':<25} {total_t+total_f:>6} {total_t:>6} {total_f:>6}")
print(f"\nFiltered out: {skipped}")
print()
print("LaTeX table rows:")
print()
for code, name in alarm_types.items():
    t = counts[code]["true"]
    f = counts[code]["false"]
    short = {
        "v": "Ventricular Flutter",
        "a": "Asystole",
        "t": "Tachycardia",
        "b": "Bradycardia",
        "f": "Ventricular Fib.",
    }[code]
    print(f"{short:<20} & {t+f:>3} & {t:>3} & {f:>3} \\\\")