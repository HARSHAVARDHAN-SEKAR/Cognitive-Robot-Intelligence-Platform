#!/usr/bin/env python3
"""
crip_learning.pipeline

The self-learning loop (Module 8): Logs -> Dataset -> Training -> Evaluation
-> Deploy. Designed to run on a schedule (e.g. nightly cron / timer).

Steps:
  1. Read the experience log and build a forecast dataset from observed tracks.
     (If too little real data, augment with synthetic tracks so the loop still
     produces a candidate -- the loop is about the *mechanism*.)
  2. Split train/val.
  3. Train a candidate forecaster (the M7 LSTM; CPU-trainable in minutes).
  4. Evaluate the candidate AND the currently-deployed model on the SAME
     validation split (val MSE).
  5. Promote the candidate only if it beats the current model by a margin
     (or if there is no current model yet).

Training deps (torch, onnx, onnxscript) are only needed when actually running
the pipeline; the dataset/registry/evaluation logic is import-safe without
them. Run:

    python3 -m crip_learning.pipeline --log <log.jsonl> --registry <dir> \
        --epochs 15 --min-improvement 0.02
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

from crip_learning.dataset import (ExperienceLog, build_forecast_dataset,
                                    dataset_stats, INPUT_LEN, HORIZON)
from crip_learning.registry import ModelRegistry


def _train_test_split(X, Y, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val, tr = idx[:n_val], idx[n_val:]
    return X[tr], Y[tr], X[val], Y[val]


def _ensure_min_data(X, Y, min_windows=200, seed=0):
    """If logged data is sparse, augment with synthetic tracks so the loop can
    still produce a candidate model. Returns possibly-augmented (X, Y)."""
    if len(X) >= min_windows:
        return X, Y
    from crip_prediction.forecaster import make_synthetic_tracks
    rng = np.random.default_rng(seed)
    tracks = make_synthetic_tracks(300, 40, rng)
    sx, sy = [], []
    for tr in tracks:
        d = np.diff(tr, axis=0)
        for i in range(len(d) - INPUT_LEN - HORIZON):
            sx.append(d[i:i + INPUT_LEN])
            sy.append(d[i + INPUT_LEN:i + INPUT_LEN + HORIZON])
    sx = np.array(sx, np.float32)
    sy = np.array(sy, np.float32)
    if len(X):
        return np.concatenate([X, sx]), np.concatenate([Y, sy])
    return sx, sy


def _onnx_val_mse(model_path, Xval, Yval):
    """Evaluate an ONNX forecaster on the validation split (mean MSE of
    predicted vs actual displacement windows)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    preds = sess.run(None, {iname: Xval.astype(np.float32)})[0]
    preds = np.asarray(preds).reshape(len(Xval), HORIZON, 2)
    return float(np.mean((preds - Yval) ** 2))


def _train_candidate(Xtr, Ytr, epochs, out_path, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class LSTMForecastNet(nn.Module):
        def __init__(self, hidden=64):
            super().__init__()
            self.lstm = nn.LSTM(2, hidden, batch_first=True)
            self.head = nn.Linear(hidden, HORIZON * 2)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    net = LSTMForecastNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(Xtr)
    Yt = torch.from_numpy(Ytr).reshape(len(Ytr), -1)
    bs, n = 256, len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(net(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
    net.eval()
    dummy = torch.zeros((1, INPUT_LEN, 2), dtype=torch.float32)
    torch.onnx.export(net, dummy, out_path, input_names=["history"],
                      output_names=["future"],
                      dynamic_axes={"history": {0: "batch"},
                                    "future": {0: "batch"}},
                      opset_version=17, dynamo=False)


def run_pipeline(log_path, registry_dir, epochs=15, min_improvement=0.02,
                 seed=0, logger=print):
    """Returns a result dict describing what the loop did."""
    log = ExperienceLog(log_path)
    records = log.read()
    stats = dataset_stats(records)
    logger(f"[pipeline] log stats: {stats}")

    X, Y = build_forecast_dataset(records)
    logger(f"[pipeline] real forecast windows: {len(X)}")
    X, Y = _ensure_min_data(X, Y, seed=seed)
    logger(f"[pipeline] training windows (after augmentation if needed): {len(X)}")

    Xtr, Ytr, Xval, Yval = _train_test_split(X, Y, seed=seed)
    reg = ModelRegistry(registry_dir)

    with tempfile.TemporaryDirectory() as d:
        cand = os.path.join(d, "candidate.onnx")
        _train_candidate(Xtr, Ytr, epochs, cand, seed=seed)
        cand_mse = _onnx_val_mse(cand, Xval, Yval)
        logger(f"[pipeline] candidate val MSE: {cand_mse:.5f}")

        cur_path = reg.current_model_path()
        cur_mse = None
        if cur_path:
            try:
                cur_mse = _onnx_val_mse(cur_path, Xval, Yval)
                logger(f"[pipeline] current  val MSE: {cur_mse:.5f}")
            except Exception as exc:  # noqa: BLE001
                logger(f"[pipeline] could not eval current model: {exc}")

        meta = {"val_mse": cand_mse, "n_train": int(len(Xtr)),
                "n_val": int(len(Xval)), "epochs": epochs,
                "log_stats": stats}
        version = reg.add_version(cand, meta)

        promoted = False
        reason = ""
        if cur_mse is None:
            promoted = True
            reason = "no current model; promoting first candidate"
        elif cand_mse <= cur_mse * (1.0 - min_improvement):
            promoted = True
            reason = (f"candidate improves val MSE by "
                      f">= {min_improvement*100:.0f}% "
                      f"({cur_mse:.5f} -> {cand_mse:.5f})")
        else:
            reason = (f"candidate did not beat current by margin "
                      f"({cur_mse:.5f} vs {cand_mse:.5f}); keeping current")

        if promoted:
            reg.promote(version)
        logger(f"[pipeline] {version}: {'PROMOTED' if promoted else 'kept'} "
               f"-- {reason}")

        return {"version": version, "promoted": promoted, "reason": reason,
                "candidate_val_mse": cand_mse, "current_val_mse": cur_mse,
                "stats": stats}


def main():
    ap = argparse.ArgumentParser(description="CRIP self-learning pipeline.")
    ap.add_argument("--log", required=True, help="experience JSONL path")
    ap.add_argument("--registry", required=True, help="model registry dir")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--min-improvement", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_pipeline(args.log, args.registry, args.epochs,
                 args.min_improvement, args.seed)


if __name__ == "__main__":
    main()
