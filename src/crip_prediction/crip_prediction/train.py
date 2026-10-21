#!/usr/bin/env python3
"""
crip_prediction.train

Offline training for the LSTM trajectory forecaster (Module 6).

Like the RL trainer, this is an OFFLINE step and is not part of the ROS build
or the test suite. It trains a small LSTM to predict the next HORIZON
displacement steps from the past INPUT_LEN displacement steps, on synthetic
pedestrian-like tracks, and exports to ONNX for the runtime node.

Usage:
    python3 -m crip_prediction.train --epochs 20 --out models/forecaster.onnx

Deps (training env only): pip install torch onnx onnxscript numpy
"""
import argparse
import os

import numpy as np

from crip_prediction.forecaster import (make_synthetic_tracks, INPUT_LEN,
                                         HORIZON)


def build_dataset(n_tracks=2000, steps=40, seed=0):
    rng = np.random.default_rng(seed)
    tracks = make_synthetic_tracks(n_tracks, steps, rng)
    X, Y = [], []
    for tr in tracks:
        deltas = np.diff(tr, axis=0)  # (steps-1, 2)
        for i in range(len(deltas) - INPUT_LEN - HORIZON):
            X.append(deltas[i:i + INPUT_LEN])
            Y.append(deltas[i + INPUT_LEN:i + INPUT_LEN + HORIZON])
    return np.array(X, np.float32), np.array(Y, np.float32)


def train(epochs, out_path, seed=0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    X, Y = build_dataset(seed=seed)
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y).reshape(len(Y), -1)  # flatten horizon*2

    class LSTMForecastNet(nn.Module):
        def __init__(self, hidden=64):
            super().__init__()
            self.lstm = nn.LSTM(input_size=2, hidden_size=hidden,
                                batch_first=True)
            self.head = nn.Linear(hidden, HORIZON * 2)

        def forward(self, x):           # x: (B, INPUT_LEN, 2)
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])   # (B, HORIZON*2)

    net = LSTMForecastNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    bs = 256
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred = net(Xt[idx])
            loss = loss_fn(pred, Yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {ep+1}/{epochs}  mse={total/n:.5f}")

    export_onnx(net, out_path)
    print(f"exported ONNX forecaster -> {out_path}")


def export_onnx(net, out_path):
    import torch
    net.eval()
    dummy = torch.zeros((1, INPUT_LEN, 2), dtype=torch.float32)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.onnx.export(
        net, dummy, out_path,
        input_names=["history"], output_names=["future"],
        dynamic_axes={"history": {0: "batch"}, "future": {0: "batch"}},
        opset_version=17, dynamo=False)


def main():
    ap = argparse.ArgumentParser(description="Train the LSTM forecaster.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="models/forecaster.onnx")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args.epochs, args.out, args.seed)


if __name__ == "__main__":
    main()
