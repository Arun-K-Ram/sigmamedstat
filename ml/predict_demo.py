"""
SigmaMedStat - Demo Prediction Script
Runs trained EfficientNet+NeuralNet model on 6 PhysioNet records.
Saves predictions as JSON for demo page hardcoding.

Records selected:
  v100s → Ventricular false alarm
  v101l → Ventricular true alarm
  a100s → Asystole false alarm
  t100s → Tachycardia false alarm
  t100l → Tachycardia true alarm
  b100s → Bradycardia false alarm
"""

import torch
import torch.nn as nn
import numpy as np
import json
import wfdb
import pywt
from pathlib import Path
from PIL import Image
from torchvision import models
from sklearn.preprocessing import StandardScaler

DATA_DIR    = Path("../backend/data/physionet/training")
MODELS_DIR  = Path("results/models")
RESULTS_DIR = Path("results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

FS           = 250
WINDOW_SEC   = 60
WINDOW_SAMP  = FS * WINDOW_SEC

ALARM_NAMES = {
    'a': 'Asystole',
    'b': 'Bradycardia',
    't': 'Tachycardia',
    'v': 'Ventricular Flutter/Fibrillation'
}

SCENARIO_META = {
    'v100s': {
        'label':        'Ventricular Flutter - False Alarm',
        'alarm_type':   'Ventricular Flutter/Fibrillation',
        'bed':          'Bed 7 · ICU West',
        'description':  'Monitor triggered a ventricular flutter alarm. Patient appears stable. SigmaMedStat analyzes whether this alarm should be acted on.',
        'ground_truth': False,
    },
    'v101l': {
        'label':        'Ventricular Flutter - True Alarm',
        'alarm_type':   'Ventricular Flutter/Fibrillation',
        'bed':          'Bed 12 · ICU East',
        'description':  'Monitor triggered a ventricular flutter alarm. SigmaMedStat confirms this is a genuine cardiac event requiring immediate response.',
        'ground_truth': True,
    },
    'a103l': {
        'label':        'Asystole - True Alarm',
        'alarm_type':   'Asystole (Cardiac Flatline)',
        'bed':          'Bed 3 · ICU North',
        'description':  'Monitor shows apparent cardiac arrest. SigmaMedStat evaluates whether this is a genuine emergency or a sensor artifact.',
        'ground_truth': True,
    },
    'a104s': {
        'label':        'Asystole - False Alarm',
        'alarm_type':   'Asystole (Cardiac Flatline)',
        'bed':          'Bed 8 · ICU North',
        'description':  'Flatline alarm triggered. SigmaMedStat determines this is likely a sensor issue, not a true cardiac arrest.',
        'ground_truth': False,
    },
    't107l': {
        'label':        'Tachycardia - True Alarm',
        'alarm_type':   'Tachycardia (Rapid Heart Rate)',
        'bed':          'Bed 2 · Step-Down Unit',
        'description':  'Rapid heart rate alarm triggered. SigmaMedStat confirms genuine tachycardia requiring clinical assessment.',
        'ground_truth': True,
    },
    'b124s': {
        'label':        'Bradycardia - False Alarm',
        'alarm_type':   'Bradycardia (Slow Heart Rate)',
        'bed':          'Bed 5 · Cardiac Unit',
        'description':  'Low heart rate alarm triggered. SigmaMedStat evaluates whether this reflects true patient deterioration.',
        'ground_truth': False,
    },
}


# ── CWT Scalogram ─────────────────────────────────────────────
def signal_to_scalogram(sig: np.ndarray, size: int = 64) -> np.ndarray:
    sig = np.nan_to_num(sig, nan=0.0)
    downsample = len(sig) // 1000
    sig_ds = sig[::downsample] if downsample > 0 else sig
    scales = np.logspace(0, 2, size)
    coeffs, _ = pywt.cwt(sig_ds, scales, 'morl')
    scalogram = np.abs(coeffs)
    img = Image.fromarray(scalogram).resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def record_to_tensor(record_stem: str) -> torch.Tensor:
    path   = str(DATA_DIR / record_stem)
    record = wfdb.rdrecord(path)
    signals = record.p_signal[-WINDOW_SAMP:, :]

    # Pad to 4 channels
    if signals.shape[1] < 4:
        pad = np.zeros((signals.shape[0], 4 - signals.shape[1]))
        signals = np.hstack([signals, pad])

    scalograms = []
    for ch in range(4):
        s = signal_to_scalogram(signals[:, ch])
        scalograms.append(s)

    tensor = np.stack(scalograms, axis=0)  # (4, 64, 64)
    return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)  # (1, 4, 64, 64)


def get_raw_signal(record_stem: str) -> dict:
    """Get raw signal values for visualization."""
    path   = str(DATA_DIR / record_stem)
    record = wfdb.rdrecord(path)
    signals = record.p_signal[-WINDOW_SAMP:, :]

    # Downsample to 100 points for frontend
    n = signals.shape[0]
    indices = np.linspace(0, n-1, 100, dtype=int)

    ch_names = record.sig_name
    result = {}
    for i, name in enumerate(ch_names[:4]):
        sig = signals[indices, i]
        sig = np.nan_to_num(sig, nan=float(np.nanmean(signals[:, i])))
        result[name] = [round(float(v), 3) for v in sig]

    return result


# ── Feature Extractor ─────────────────────────────────────────
def build_extractor() -> nn.Module:
    m = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT)
    orig = m.features[0][0]
    new = nn.Conv2d(4, orig.out_channels, orig.kernel_size,
                    orig.stride, orig.padding, bias=False)
    with torch.no_grad():
        new.weight[:, :3] = orig.weight
        new.weight[:, 3]  = orig.weight.mean(dim=1)
    m.features[0][0] = new
    extractor = nn.Sequential(m.features, m.avgpool, nn.Flatten())
    for p in extractor.parameters():
        p.requires_grad = False
    return extractor.eval().to(DEVICE)


# ── Neural Classifier ─────────────────────────────────────────
class NeuralClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 4, 16)),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(max(hidden_dim // 4, 16), 2)
        )

    def forward(self, x):
        return self.net(x)


def load_classifier(feat_dim: int) -> nn.Module:
    model = NeuralClassifier(feat_dim, hidden_dim=256, dropout=0.5)
    model_path = MODELS_DIR / "efficientnet_neural_best.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded model from {model_path}")
    else:
        print(f"WARNING: No saved model found at {model_path}")
        print("Using untrained model - predictions will be random")
    return model.eval().to(DEVICE)


# ── Main Prediction ───────────────────────────────────────────
def predict_record(record_stem: str, extractor: nn.Module,
                   classifier: nn.Module, scaler: StandardScaler) -> dict:
    print(f"\nProcessing: {record_stem}")

    # Get scalogram tensor
    tensor = record_to_tensor(record_stem).to(DEVICE)

    # Extract features
    with torch.no_grad():
        features = extractor(tensor)  # (1, 1280)
        features_np = features.cpu().numpy()

        # Scale
        features_scaled = scaler.transform(features_np)
        features_t = torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE)

        # Classify
        logits = classifier(features_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    false_alarm_prob = float(probs[0])
    true_alarm_prob  = float(probs[1])
    predicted_false  = false_alarm_prob > 0.5

    trust_score = int((false_alarm_prob if predicted_false
                       else true_alarm_prob) * 100)

    meta = SCENARIO_META[record_stem]
    ground_truth = meta['ground_truth']
    correct = (predicted_false == (not ground_truth))

    print(f"  False alarm prob: {false_alarm_prob:.3f}")
    print(f"  True alarm prob:  {true_alarm_prob:.3f}")
    print(f"  Predicted:        {'FALSE ALARM' if predicted_false else 'TRUE ALARM'}")
    print(f"  Ground truth:     {'FALSE ALARM' if not ground_truth else 'TRUE ALARM'}")
    print(f"  Correct:          {correct}")

    # Get raw signals
    raw_signals = get_raw_signal(record_stem)

    # Build result
    alarm_type = record_stem[0]

    return {
        "record":           record_stem,
        "label":            meta['label'],
        "alarm_type":       meta['alarm_type'],
        "bed":              meta['bed'],
        "description":      meta['description'],
        "ground_truth":     ground_truth,
        "raw_signals":      raw_signals,
        "prediction": {
            "is_false_alarm":      predicted_false,
            "false_alarm_prob":    round(false_alarm_prob, 4),
            "true_alarm_prob":     round(true_alarm_prob, 4),
            "trust_score":         trust_score,
            "correct":             correct,
            "grade": (
                "CRITICAL" if trust_score < 25 else
                "POOR"     if trust_score < 50 else
                "GOOD"     if trust_score < 75 else
                "EXCELLENT"
            ),
        },
        "pipeline": {
            "step1": "Continuous Wavelet Transform applied to 60-second signal window",
            "step2": "EfficientNet-B0 extracts 1,280 time-frequency features",
            "step3": "Neural classifier (256→64→2) produces alarm probability",
            "model": "EfficientNet-B0 + NeuralNet (AUC 0.641 on PhysioNet Challenge 2015)",
        },
        "clinical": {
            "alarm_type_full":   meta['alarm_type'],
            "action": (
                "Do NOT act on this alarm - high probability of false alarm. Verify sensor placement."
                if predicted_false else
                "Act immediately - model indicates genuine clinical event."
            ),
            "without_sigmamedstat": (
                "Without SigmaMedStat, this alarm would be one of 350+ daily alerts - statistically likely to be ignored."
                if not ground_truth else
                "Without SigmaMedStat, this alarm could be dismissed as another false positive."
            ),
        }
    }


def main():
    records = list(SCENARIO_META.keys())

    # First extract features from training data to fit scaler
    print("Fitting scaler on training data...")
    X = np.load("../backend/data/scalograms/X.npy")
    y = np.load("../backend/data/scalograms/y.npy")

    np.random.seed(42)
    idx    = np.random.permutation(len(X))
    tr_end = int(0.70 * len(X))
    X_tr   = X[idx[:tr_end]]

    extractor = build_extractor()

    # Extract training features for scaler
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32))
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)
    feats   = []
    with torch.no_grad():
        for (batch,) in tqdm(loader, desc="Fitting scaler"):
            f = extractor(batch.to(DEVICE))
            feats.append(f.cpu().numpy())
    feat_tr = np.concatenate(feats, axis=0)

    scaler = StandardScaler()
    scaler.fit(feat_tr)
    print(f"Scaler fitted on {feat_tr.shape[0]} training samples")

    # Load classifier
    classifier = load_classifier(feat_tr.shape[1])

    # Predict all records
    all_results = {}
    for record in records:
        # Check record exists
        hea_path = DATA_DIR / f"{record}.hea"
        if not hea_path.exists():
            print(f"Skipping {record} - not found in dataset")
            continue
        result = predict_record(record, extractor, classifier, scaler)
        all_results[record] = result

    # Save
    output_path = RESULTS_DIR / "demo_predictions.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved predictions to {output_path}")
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for rec, res in all_results.items():
        p = res['prediction']
        gt = "FALSE" if not res['ground_truth'] else "TRUE"
        pred = "FALSE" if p['is_false_alarm'] else "TRUE"
        correct = "✓" if p['correct'] else "✗"
        print(f"  {rec:10s} GT:{gt:5s} Pred:{pred:5s} "
              f"Conf:{p['false_alarm_prob']:.2f} {correct}")


if __name__ == "__main__":
    main()