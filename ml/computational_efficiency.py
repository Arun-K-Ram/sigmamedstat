"""
SigmaMedStat - Computational Efficiency Analysis
Measures inference time, parameter count, and memory
for all three sequence architectures.

Reports:
  1. Parameter count per model
  2. Inference time per sample (CPU and GPU)
  3. Batch inference throughput
  4. Memory footprint
  5. Real-time feasibility assessment
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
from pathlib import Path
from torchvision import models
import math

RESULTS_DIR = Path("results")
DEVICE_GPU  = torch.device("cuda" if torch.cuda.is_available()
                           else "cpu")
DEVICE_CPU  = torch.device("cpu")

# ── Model definitions ──────────────────────────────────────────
def build_encoder():
    m    = models.efficientnet_b0(
               weights=models.EfficientNet_B0_Weights.DEFAULT)
    orig = m.features[0][0]
    new  = nn.Conv2d(4, orig.out_channels, orig.kernel_size,
                     orig.stride, orig.padding, bias=False)
    with torch.no_grad():
        new.weight[:, :3] = orig.weight
        new.weight[:, 3]  = orig.weight.mean(dim=1)
    m.features[0][0] = new
    return nn.Sequential(m.features, m.avgpool, nn.Flatten())


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = build_encoder()
        self.lstm    = nn.LSTM(1280, 64, 2,
                               batch_first=True, dropout=0.3)
        self.head    = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.3),   nn.Linear(64, 2))

    def forward(self, x):
        b, c, ch, H, W = x.shape
        f = self.encoder(x.view(b*c, ch, H, W))
        f = f.view(b, c, -1)
        _, (hn, _) = self.lstm(f)
        return self.head(hn[-1])


class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = build_encoder()
        self.lstm    = nn.LSTM(1280, 128, 2,
                               batch_first=True,
                               dropout=0.4,
                               bidirectional=True)
        self.head    = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Dropout(0.4),    nn.Linear(64, 2))

    def forward(self, x):
        b, c, ch, H, W = x.shape
        f = self.encoder(x.view(b*c, ch, H, W))
        f = f.view(b, c, -1)
        _, (hn, _) = self.lstm(f)
        combined = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.head(combined)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=10, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = build_encoder()
        self.proj    = nn.Linear(1280, 256)
        self.pos_enc = PositionalEncoding()
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=256, nhead=4,
            dim_feedforward=512,
            dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Dropout(0.3),    nn.Linear(64, 2))

    def forward(self, x):
        b, c, ch, H, W = x.shape
        f = self.encoder(x.view(b*c, ch, H, W))
        f = f.view(b, c, -1)
        f = self.proj(f)
        f = self.pos_enc(f)
        f = self.transformer(f)
        return self.head(f.mean(dim=1))


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    return total, trainable


def measure_inference(model, dummy_input, device,
                      n_warmup=10, n_runs=100):
    model = model.to(device).eval()
    x     = dummy_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _  = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms":   round(float(np.mean(times)), 3),
        "std_ms":    round(float(np.std(times)), 3),
        "min_ms":    round(float(np.min(times)), 3),
        "max_ms":    round(float(np.max(times)), 3),
        "p95_ms":    round(float(np.percentile(times, 95)), 3),
    }


def main():
    print("SigmaMedStat - Computational Efficiency Analysis")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print("=" * 60)

    models_list = [
        ("LSTM (Exp 04)",        LSTMModel()),
        ("BiLSTM (Exp 05)",      BiLSTMModel()),
        ("Transformer (Exp 06)", TransformerModel()),
    ]

    # Dummy inputs
    single_input = torch.randn(1, 6, 4, 64, 64)
    batch_input  = torch.randn(16, 6, 4, 64, 64)

    results = {}

    for name, model in models_list:
        print(f"\n{name}")
        print("-" * 50)

        # Parameter count
        total, trainable = count_parameters(model)
        print(f"  Total parameters:     {total:>12,}")
        print(f"  Trainable parameters: {trainable:>12,}")
        print(f"  Model size (MB):      "
              f"{total*4/1024/1024:>10.2f}")

        # CPU inference - single sample
        cpu_single = measure_inference(
            model, single_input, DEVICE_CPU,
            n_warmup=5, n_runs=50)
        print(f"  CPU inference (1 sample): "
              f"{cpu_single['mean_ms']:.2f} ± "
              f"{cpu_single['std_ms']:.2f} ms")

        # GPU inference if available
        gpu_single = None
        gpu_batch  = None
        if torch.cuda.is_available():
            gpu_single = measure_inference(
                model, single_input, DEVICE_GPU,
                n_warmup=10, n_runs=100)
            gpu_batch  = measure_inference(
                model, batch_input, DEVICE_GPU,
                n_warmup=10, n_runs=100)
            print(f"  GPU inference (1 sample): "
                  f"{gpu_single['mean_ms']:.2f} ± "
                  f"{gpu_single['std_ms']:.2f} ms")
            print(f"  GPU inference (batch=16): "
                  f"{gpu_batch['mean_ms']:.2f} ± "
                  f"{gpu_batch['std_ms']:.2f} ms")
            throughput = 16 / (gpu_batch["mean_ms"] / 1000)
            print(f"  GPU throughput:           "
                  f"{throughput:.0f} samples/sec")

        # Real-time feasibility
        # ICU alarm fires → model must respond before
        # nurse reaches bedside (~30 seconds)
        # We use 60s window so processing must be << 60s
        feasible_cpu = cpu_single["mean_ms"] < 60000
        feasible_gpu = (gpu_single["mean_ms"] < 1000
                        if gpu_single else None)
        print(f"  Real-time feasible (CPU): {feasible_cpu}")
        if feasible_gpu is not None:
            print(f"  Real-time feasible (GPU): {feasible_gpu}")

        results[name] = {
            "total_params":    total,
            "trainable_params": trainable,
            "model_size_mb":   round(total*4/1024/1024, 2),
            "cpu_inference":   cpu_single,
            "gpu_inference":   gpu_single,
            "gpu_batch":       gpu_batch,
            "realtime_cpu":    feasible_cpu,
            "realtime_gpu":    feasible_gpu,
        }

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Model':<25} {'Params':>10} {'Size(MB)':>9} "
          f"{'CPU(ms)':>8} {'GPU(ms)':>8}")
    print("-" * 65)
    for name, res in results.items():
        short  = name.split("(")[0].strip()
        cpu_ms = res["cpu_inference"]["mean_ms"]
        gpu_ms = (res["gpu_inference"]["mean_ms"]
                  if res["gpu_inference"] else "N/A")
        gpu_str = f"{gpu_ms:.2f}" if isinstance(gpu_ms, float) \
                  else gpu_ms
        print(f"  {short:<23} {res['total_params']:>10,} "
              f"{res['model_size_mb']:>9.1f} "
              f"{cpu_ms:>8.1f} {gpu_str:>8}")

    # Save
    out = RESULTS_DIR / "computational_efficiency.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()