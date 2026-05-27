"""
SigmaMedStat - Grad-CAM Visualization Script (v3)
Clean, readable layout designed for non-technical audiences.

Layout per record:
  - Header: alarm info + model verdict
  - Row 1: Raw ECG waveform (familiar heartbeat line)
  - Row 2 left: CWT Scalogram (ECG Lead II only, explained simply)
  - Row 2 right: Grad-CAM overlay (where the model looked)
  - Footer: plain-English explanation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from torchvision import models
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import cv2
import json
import wfdb

#  Paths 
SCALOGRAM_DIR = Path("C:/Users/Arun/Documents/git/crip-x/backend/data/scalograms")
PHYSIONET_DIR = Path("C:/Users/Arun/Documents/git/crip-x/backend/data/physionet/training")
MODELS_DIR    = Path("results/models")
OUTPUT_DIR    = Path("C:/Users/Arun/Documents/git/crip-x/frontend/public/gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

RECORDS = {
    "v100s": {"label":"Irregular heartbeat",  "alarm_type":"Ventricular Flutter",       "ground_truth":False},
    "v101l": {"label":"Irregular heartbeat",  "alarm_type":"Ventricular Flutter",       "ground_truth":True},
    "a109l": {"label":"Cardiac arrest",       "alarm_type":"Asystole",                  "ground_truth":True},
    "b187l": {"label":"Slow heart rate",      "alarm_type":"Bradycardia",               "ground_truth":True},
    "t116s": {"label":"Rapid heart rate",     "alarm_type":"Tachycardia",               "ground_truth":False},
    "f120s": {"label":"Irregular heartbeat",  "alarm_type":"Ventricular Fibrillation",  "ground_truth":False},
}

BG   = "#f4f4f2"
TEXT = "#2c3e50"
MID  = "#e8e8e5"
RED  = "#c0392b"
GRN  = "#1e8449"
GREY = "#7f8c8d"

#  Model 
def build_extractor():
    m    = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    orig = m.features[0][0]
    new  = nn.Conv2d(4, orig.out_channels, orig.kernel_size,
                     orig.stride, orig.padding, bias=False)
    with torch.no_grad():
        new.weight[:, :3] = orig.weight
        new.weight[:, 3]  = orig.weight.mean(dim=1)
    m.features[0][0] = new
    return m.to(DEVICE)

class NeuralClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim//4, 16)),
            nn.ReLU(),
            nn.Dropout(dropout*0.5),
            nn.Linear(max(hidden_dim//4, 16), 2)
        )
    def forward(self, x): return self.net(x)

#  Grad-CAM 
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        model.features[4].register_forward_hook(self._save_act)
        model.features[4].register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):  self.activations = o.detach().clone()
    def _save_grad(self, m, gi, go): self.gradients  = go[0].detach().clone()

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        t = tensor.clone().requires_grad_(True)
        out = self.model.features(t)
        out = self.model.avgpool(out).flatten(1)
        out[0, class_idx % out.shape[1]].backward()

        w   = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = torch.relu((w * self.activations).sum(dim=1)).squeeze().cpu().numpy()
        cam = cv2.GaussianBlur(cam, (5,5), 0)
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

#  Fit scaler 
def fit_scaler(extractor):
    print("Fitting scaler...")
    X    = np.load(SCALOGRAM_DIR / "X.npy")
    idx  = np.random.permutation(len(X))
    X_tr = X[idx[:int(0.70*len(X))]]
    ldr  = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32)), batch_size=32)
    feats = []
    extractor.eval()
    with torch.no_grad():
        for (b,) in ldr:
            o = extractor.features(b.to(DEVICE))
            feats.append(extractor.avgpool(o).flatten(1).cpu().numpy())
    ft = np.concatenate(feats)
    print(f"  {ft.shape[0]} samples, {ft.shape[1]} features")
    return StandardScaler().fit(ft), ft.shape[1]

#  Load raw ECG waveform 
def load_raw_ecg(record_id, n_points=500):
    """Load raw ECG Lead II signal, downsample to n_points for display."""
    path = str(PHYSIONET_DIR / record_id)
    try:
        rec = wfdb.rdrecord(path)
        sig = rec.p_signal[-15000:, 0]   # ECG Lead II, last 60s
        sig = np.nan_to_num(sig)
        # Downsample to n_points
        indices = np.linspace(0, len(sig)-1, n_points, dtype=int)
        return sig[indices]
    except Exception as e:
        print(f"  Could not load raw signal for {record_id}: {e}")
        return None

#  Generate one clean image per record 
def generate(record_id, scalogram, meta, extractor, classifier, scaler, gradcam):
    extractor.eval(); classifier.eval()

    # Prediction
    # Prediction
    tensor = torch.tensor(scalogram[np.newaxis], dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        features_out = extractor.features(tensor)
        pooled       = extractor.avgpool(features_out)
        feat         = pooled.flatten(1).detach().cpu().numpy()

    feat_scaled = scaler.transform(feat)
    feat_t      = torch.tensor(feat_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = classifier(feat_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    false_prob = float(probs[0])
    true_prob  = float(probs[1])
    is_false   = false_prob > 0.5
    confidence = max(false_prob, true_prob)
    correct    = is_false == (not meta["ground_truth"])

    # Grad-CAM on ECG Lead II channel (ch 0)
    cam    = gradcam.generate(tensor, class_idx=0 if is_false else 1)
    cam_64 = cv2.resize(cam, (64, 64))

    # Raw ECG
    raw_ecg = load_raw_ecg(record_id)

    #  Labels 
    gt_label   = "This was a FALSE alarm - no real emergency" \
                 if not meta["ground_truth"] \
                 else "This was a REAL emergency"
    pred_label = "Device error - stand down" \
                 if is_false \
                 else "Real emergency - act now"
    gt_color   = GRN if not meta["ground_truth"] else RED
    pred_color = GRN if is_false                 else RED
    tick       = "✓  Model got this right" if correct else "✗  Model got this wrong"
    tick_color = GRN if correct else RED

    #  Figure 
    fig = plt.figure(figsize=(16, 10), facecolor=BG)

    #  Header strip 
    header = fig.add_axes([0, 0.91, 1, 0.09], facecolor=TEXT)
    header.set_axis_off()
    header.text(0.02, 0.65, f"{meta['label']}  ·  {meta['alarm_type']}",
                color="white", fontsize=13, va="center", fontweight="medium",
                transform=header.transAxes)
    header.text(0.02, 0.20, f"Record: {record_id}",
                color="#95a5a6", fontsize=9, va="center",
                transform=header.transAxes)
    # Ground truth pill
    header.text(0.38, 0.50,  "What actually happened:",
                color="#95a5a6", fontsize=9, va="center",
                transform=header.transAxes)
    header.text(0.55, 0.50, gt_label,
                color=gt_color, fontsize=10, va="center", fontweight="medium",
                transform=header.transAxes)
    # Correct tick
    header.text(0.82, 0.50, tick,
                color=tick_color, fontsize=10, va="center", fontweight="medium",
                transform=header.transAxes)

    #  Main grid: 2 rows, 2 cols 
    # Row 0: Raw ECG waveform (full width)
    # Row 1 left: Scalogram
    # Row 1 right: Grad-CAM
    gs = GridSpec(2, 2, figure=fig,
                  top=0.89, bottom=0.16,
                  left=0.06, right=0.97,
                  hspace=0.45, wspace=0.18,
                  height_ratios=[1, 1.4])

    #  Row 0: Raw ECG (spans both columns) 
    ax_ecg = fig.add_subplot(gs[0, :])
    ax_ecg.set_facecolor(BG)

    if raw_ecg is not None:
        t = np.linspace(0, 60, len(raw_ecg))
        ax_ecg.plot(t, raw_ecg, color=RED, linewidth=1.0, alpha=0.9)
        ax_ecg.fill_between(t, raw_ecg, alpha=0.08, color=RED)
    else:
        ax_ecg.text(30, 0, "Raw signal unavailable", ha="center",
                    color=GREY, fontsize=10)

    ax_ecg.set_xlim(0, 60)
    ax_ecg.set_xlabel("Time (seconds)", fontsize=9, color=GREY, labelpad=4)
    ax_ecg.set_ylabel("Electrical\namplitude (mV)", fontsize=9,
                       color=GREY, labelpad=4, linespacing=1.3)
    ax_ecg.set_title("Raw heart rhythm signal  ·  ECG Lead II  ·  60 seconds before alarm",
                      fontsize=10, color=TEXT, pad=8, loc="left")
    ax_ecg.tick_params(colors=GREY, labelsize=8)
    ax_ecg.spines["top"].set_visible(False)
    ax_ecg.spines["right"].set_visible(False)
    for sp in ["bottom","left"]:
        ax_ecg.spines[sp].set_edgecolor(MID)

    # Alarm marker at end
    ax_ecg.axvline(x=59.5, color=RED, linestyle="--", linewidth=1.0, alpha=0.7)
    ax_ecg.text(59, ax_ecg.get_ylim()[1]*0.85, "Alarm\nfired",
                color=RED, fontsize=7, ha="right", linespacing=1.3)

    #  Row 1 left: Scalogram 
    ax_sc = fig.add_subplot(gs[1, 0])
    ax_sc.set_facecolor(BG)
    scalo = scalogram[0]  # ECG Lead II scalogram
    im_sc = ax_sc.imshow(scalo, aspect="auto", origin="lower",
                          cmap="magma", interpolation="bilinear")
    ax_sc.set_title("Step 1 - Signal converted to heat map\n"
                    "(same data, different view: bright = strong signal activity)",
                    fontsize=9, color=TEXT, pad=6, linespacing=1.4, loc="left")
    ax_sc.set_xlabel("Time  →  (0 to 60 seconds)", fontsize=8, color=GREY, labelpad=4)
    ax_sc.set_ylabel("Signal frequency  →\n(low at bottom, high at top)",
                      fontsize=8, color=GREY, labelpad=4, linespacing=1.3)
    ax_sc.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax_sc.spines.values(): sp.set_edgecolor(MID)

    cb1 = fig.colorbar(im_sc, ax=ax_sc, fraction=0.04, pad=0.02, shrink=0.85)
    cb1.ax.tick_params(labelsize=7, colors=GREY)
    cb1.set_label("Signal intensity", fontsize=7, color=GREY)

    # Frequency labels
    ylim = ax_sc.get_ylim()
    ax_sc.text(-3, ylim[0] + (ylim[1]-ylim[0])*0.05,  "Low",
               fontsize=7, color=GREY, ha="right", va="center")
    ax_sc.text(-3, ylim[0] + (ylim[1]-ylim[0])*0.5,   "Mid",
               fontsize=7, color=GREY, ha="right", va="center")
    ax_sc.text(-3, ylim[0] + (ylim[1]-ylim[0])*0.92,  "High",
               fontsize=7, color=GREY, ha="right", va="center")

    #  Row 1 right: Grad-CAM 
    ax_gc = fig.add_subplot(gs[1, 1])
    ax_gc.set_facecolor(BG)

    # Greyscale base
    ax_gc.imshow(scalo, aspect="auto", origin="lower",
                 cmap="Greys", alpha=0.65, interpolation="bilinear")
    # Grad-CAM overlay
    im_gc = ax_gc.imshow(cam_64, aspect="auto", origin="lower",
                          cmap="Reds", alpha=0.70, interpolation="bilinear",
                          vmin=0, vmax=1)
    # Contour to show hotspot boundary
    ax_gc.contour(cam_64, levels=[0.55, 0.75],
                  colors=[RED], linewidths=[0.7, 1.2], alpha=0.85)

    ax_gc.set_title("Step 2 - Where the model focused its attention\n"
                    "(red areas had the most influence on the prediction)",
                    fontsize=9, color=TEXT, pad=6, linespacing=1.4, loc="left")
    ax_gc.set_xlabel("Time  →  (0 to 60 seconds)", fontsize=8, color=GREY, labelpad=4)
    ax_gc.set_ylabel("Signal frequency  →", fontsize=8, color=GREY, labelpad=4)
    ax_gc.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax_gc.spines.values(): sp.set_edgecolor(MID)

    cb2 = fig.colorbar(im_gc, ax=ax_gc, fraction=0.04, pad=0.02, shrink=0.85)
    cb2.ax.tick_params(labelsize=7, colors=GREY)
    cb2.set_label("Attention strength", fontsize=7, color=GREY)

    #  Prediction box at bottom 
    pred_ax = fig.add_axes([0.06, 0.03, 0.91, 0.10],
                            facecolor="white")
    pred_ax.set_xlim(0, 1)
    pred_ax.set_ylim(0, 1)
    pred_ax.set_axis_off()

    # Border
    for sp_pos in [(0,0,1,0),(1,0,1,1),(1,1,0,1),(0,1,0,0)]:
        pred_ax.plot([sp_pos[0],sp_pos[2]], [sp_pos[1],sp_pos[3]],
                     color=MID, linewidth=0.8, transform=pred_ax.transAxes)

    pred_ax.text(0.01, 0.75, "Model verdict:",
                 fontsize=9, color=GREY, va="center")
    pred_ax.text(0.13, 0.75, pred_label,
                 fontsize=11, color=pred_color, va="center", fontweight="medium")
    pred_ax.text(0.01, 0.28, f"Confidence: {confidence*100:.0f}%",
                 fontsize=9, color=GREY, va="center")

    # Confidence bar
    bar_x, bar_y, bar_w, bar_h = 0.28, 0.28, 0.40, 0.18
    pred_ax.add_patch(mpatches.FancyBboxPatch(
        (bar_x, bar_y - bar_h/2 - 0.05), bar_w, bar_h + 0.05,
        boxstyle="round,pad=0.01", facecolor=MID, edgecolor="none"))
    pred_ax.add_patch(mpatches.FancyBboxPatch(
        (bar_x, bar_y - bar_h/2 - 0.05), bar_w * confidence, bar_h + 0.05,
        boxstyle="round,pad=0.01", facecolor=pred_color, edgecolor="none",
        alpha=0.85))
    pred_ax.text(bar_x + bar_w + 0.01, bar_y, f"{confidence*100:.0f}%",
                 fontsize=9, color=pred_color, va="center")

    pred_ax.text(0.72, 0.75, "Model trained on:",
                 fontsize=8, color=GREY, va="center")
    pred_ax.text(0.72, 0.28,
                 "750 real ICU alarm recordings  ·  EfficientNet-B0  ·  PhysioNet Challenge 2015",
                 fontsize=8, color=GREY, va="center")

    # Save
    out = OUTPUT_DIR / f"{record_id}_gradcam.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")

    return {
        "record":             record_id,
        "label":              meta["label"],
        "alarm_type":         meta["alarm_type"],
        "ground_truth":       meta["ground_truth"],
        "ground_truth_plain": gt_label,
        "predicted":          pred_label,
        "is_false_alarm":     is_false,
        "false_alarm_prob":   round(false_prob, 4),
        "true_alarm_prob":    round(true_prob, 4),
        "confidence":         round(confidence, 4),
        "correct":            correct,
        "image":              f"/gradcam/{record_id}_gradcam.png",
    }


#  Main 
def main():
    np.random.seed(42)
    X     = np.load(SCALOGRAM_DIR / "X.npy")
    names = list(np.load(SCALOGRAM_DIR / "names.npy"))

    extractor = build_extractor()
    scaler, feat_dim = fit_scaler(extractor)

    classifier = NeuralClassifier(feat_dim)
    mp = MODELS_DIR / "efficientnet_neural_best.pt"
    if mp.exists():
        classifier.load_state_dict(torch.load(mp, map_location=DEVICE))
        print(f"Loaded model from {mp.name}")
    else:
        print("WARNING: No saved model found")
    classifier.to(DEVICE)

    gradcam = GradCAM(extractor)

    results = {}
    print("\nGenerating visualizations...\n")
    for rid, meta in RECORDS.items():
        if rid not in names:
            print(f"SKIP {rid}"); continue
        idx = names.index(rid)
        print(f"Processing {rid}...")
        r = generate(rid, X[idx], meta, extractor, classifier, scaler, gradcam)
        results[rid] = r
        gt   = "FALSE" if not meta["ground_truth"] else "TRUE"
        pred = "FALSE" if r["is_false_alarm"] else "TRUE"
        print(f"  GT:{gt}  Pred:{pred}  Conf:{r['confidence']*100:.0f}%  "
              f"{'✓' if r['correct'] else '✗'}\n")

    with open("results/gradcam_results.json", "w") as f:
        json.dump(results, f, indent=2)

    correct = sum(1 for r in results.values() if r["correct"])
    print("="*50)
    print(f"DONE - {correct}/{len(results)} correct")
    print("="*50)

if __name__ == "__main__":
    main()