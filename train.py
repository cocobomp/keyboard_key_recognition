#!/usr/bin/env python3
"""train.py — small CNN over the keystroke mel-spectrograms.

The split is *per session*, always.  Train sessions and validation sessions are
disjoint recordings, and the test sessions passed here are only used to assert
they never leak into training.  There is no random window-level split anywhere
in this file, by design: two presses of the same key inside one session share
the microphone, the room and the hand position, so a random split would score
memorisation instead of recognition.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import kkr_common as kc


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_sessions(processed_dir: str, session_ids):
    """Load the given processed sessions; returns X, y(str), sessions, modes, cfg."""
    Xs, ys, sids, modes = [], [], [], []
    cfg = None
    for sid in session_ids:
        path = os.path.join(processed_dir, f"{sid}.npz")
        if not os.path.isfile(path):
            raise SystemExit(f"missing processed session: {path} (run preprocess.py first)")
        with np.load(path, allow_pickle=False) as z:
            X = z["X"].astype(np.float32)
            y = z["y"].astype(str)
            mode = str(z["mode"])
            c = json.loads(str(z["config"]))
        if cfg is None:
            cfg = c
        elif {k: c[k] for k in c if k != "key_set"} != {k: cfg[k] for k in cfg if k != "key_set"}:
            raise SystemExit(
                f"session {sid} was preprocessed with a different feature config; "
                "re-run preprocess.py over every session with the same options."
            )
        Xs.append(X)
        ys.append(y)
        sids.extend([sid] * len(y))
        modes.extend([mode] * len(y))
    if not Xs:
        raise SystemExit("no session to load")
    return (
        np.concatenate(Xs),
        np.concatenate(ys),
        np.array(sids),
        np.array(modes),
        cfg,
    )


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False, seed: int = 0):
        self.X = torch.from_numpy(X)  # (n, n_mels, n_frames)
        self.y = torch.from_numpy(y)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        n_mels, n_frames = x.shape
        shift = int(self.rng.integers(-6, 7))  # +-16 ms of timing jitter
        if shift:
            x = torch.roll(x, shift, dims=1)
        x = x + float(self.rng.normal(0.0, 0.15))  # session-gain-like offset
        x = x + torch.from_numpy(self.rng.normal(0, 0.05, size=x.shape).astype(np.float32))
        if self.rng.random() < 0.5:  # SpecAugment-style masking
            w = int(self.rng.integers(1, max(2, n_mels // 8)))
            f0 = int(self.rng.integers(0, n_mels - w))
            x[f0 : f0 + w, :] = 0.0
        if self.rng.random() < 0.5:
            w = int(self.rng.integers(1, max(2, n_frames // 8)))
            t0 = int(self.rng.integers(0, n_frames - w))
            x[:, t0 : t0 + w] = 0.0
        return x

    def __getitem__(self, i):
        x = self.X[i].clone() if self.augment else self.X[i]
        if self.augment:
            x = self._augment(x)
        return x.unsqueeze(0), self.y[i]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class KeyCNN(nn.Module):
    """3 conv blocks + dense head — small enough for a few thousand windows."""

    def __init__(self, n_classes: int, widths=(32, 64, 128), dropout: float = 0.3):
        super().__init__()
        blocks, in_ch = [], 1
        for w in widths:
            blocks += [
                nn.Conv2d(in_ch, w, 3, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
                nn.Conv2d(w, w, 3, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    k = min(k, logits.shape[1])
    pred = logits.topk(k, dim=1).indices
    return int((pred == targets.unsqueeze(1)).any(dim=1).sum().item())


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum = n = c1 = c5 = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += float(criterion(logits, yb).item()) * len(yb)
        c1 += topk_correct(logits, yb, 1)
        c5 += topk_correct(logits, yb, 5)
        n += len(yb)
    return loss_sum / max(n, 1), c1 / max(n, 1), c5 / max(n, 1)


def resolve_splits(args) -> tuple[list[str], list[str], list[str]]:
    train, val, test = args.train_sessions or [], args.val_sessions or [], args.test_sessions or []
    if args.split_file:
        spec = kc.read_json(args.split_file)
        train = train or list(spec.get("train", []))
        val = val or list(spec.get("val", []))
        test = test or list(spec.get("test", []))
    if not train:
        raise SystemExit("--train-sessions (or --split-file) is required: the split must be explicit.")
    if not val:
        if len(train) < 2:
            raise SystemExit(
                "need at least 2 training sessions so one can be held out for validation, "
                "or pass --val-sessions explicitly."
            )
        val = [train[-1]]
        train = train[:-1]
        print(
            f"! no --val-sessions given: holding out session '{val[0]}' from the training "
            "set for model selection (still a per-session split, never a random one)."
        )
    kc.check_disjoint(train=train, val=val, test=test)
    return train, val, test


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--train-sessions", nargs="*", default=None)
    p.add_argument("--val-sessions", nargs="*", default=None, help="held-out session(s) for model selection")
    p.add_argument("--test-sessions", nargs="*", default=None, help="declared here only to assert no leakage")
    p.add_argument("--split-file", default=None, help='JSON: {"train": [...], "val": [...], "test": [...]}')
    p.add_argument("--out", default="models/keycnn.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--widths", type=int, nargs=3, default=(32, 64, 128))
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--balance-classes", action="store_true", help="inverse-frequency loss weighting")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cpu | mps | cuda (default: best available)")
    return p.parse_args(argv)


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ids, val_ids, test_ids = resolve_splits(args)
    print(f"train sessions: {train_ids}\nval sessions  : {val_ids}\ntest sessions : {test_ids or '(none declared)'}")

    Xtr, ytr, _, mtr, cfg = load_sessions(args.processed_dir, train_ids)
    Xva, yva, _, _, _ = load_sessions(args.processed_dir, val_ids)

    vocab = sorted(set(ytr.tolist()))
    idx = {k: i for i, k in enumerate(vocab)}
    unseen = sorted(set(yva.tolist()) - set(vocab))
    if unseen:
        print(f"! validation keys never seen in training, dropped from val: {unseen}")
        keep = np.array([k in idx for k in yva])
        Xva, yva = Xva[keep], yva[keep]
    ytr_i = np.array([idx[k] for k in ytr], dtype=np.int64)
    yva_i = np.array([idx[k] for k in yva], dtype=np.int64)

    # Per-mel-bin standardisation, statistics from the TRAIN sessions only.
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-5
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd

    counts = Counter(ytr.tolist())
    print(
        f"\ntrain windows: {len(ytr_i)} ({dict(Counter(mtr.tolist()))})  "
        f"val windows: {len(yva_i)}\nclasses: {len(vocab)}  "
        f"min/median/max per class: {min(counts.values())}/"
        f"{int(np.median(list(counts.values())))}/{max(counts.values())}"
    )
    print(f"feature shape: {Xtr.shape[1:]}  (mel bins x frames)")

    device = pick_device(args.device)
    print(f"device: {device}\n")

    train_ds = WindowDataset(Xtr, ytr_i, augment=args.augment, seed=args.seed)
    val_ds = WindowDataset(Xva, yva_i, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
                          num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=args.num_workers)

    model = KeyCNN(len(vocab), tuple(args.widths), args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: KeyCNN widths={tuple(args.widths)}  {n_params / 1e3:.0f}k params")

    weight = None
    if args.balance_classes:
        w = np.array([1.0 / counts[k] for k in vocab], dtype=np.float32)
        weight = torch.tensor(w / w.mean(), device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    kc.ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    log_path = os.path.splitext(args.out)[0] + "_trainlog.csv"
    log_rows = []
    best_acc, best_epoch = -1.0, -1
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = n = c1 = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(yb)
            c1 += topk_correct(logits.detach(), yb, 1)
            n += len(yb)
        sched.step()
        tr_loss, tr_acc = loss_sum / max(n, 1), c1 / max(n, 1)
        va_loss, va_acc, va_top5 = evaluate(model, val_dl, device, criterion)
        log_rows.append(
            {"epoch": epoch, "train_loss": tr_loss, "train_top1": tr_acc,
             "val_loss": va_loss, "val_top1": va_acc, "val_top5": va_top5,
             "lr": sched.get_last_lr()[0]}
        )
        flag = ""
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "vocab": vocab,
                    "norm": {"mu": mu, "sd": sd},
                    "feature_config": cfg,
                    "model": {"widths": list(args.widths), "dropout": args.dropout,
                              "input_shape": list(Xtr.shape[1:])},
                    "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
                    "args": vars(args),
                    "epoch": epoch,
                    "val_top1": va_acc,
                    "val_top5": va_top5,
                },
                args.out,
            )
            flag = "  <- best"
        print(
            f"epoch {epoch:3d}/{args.epochs}  train loss {tr_loss:.3f} top1 {kc.fmt_pct(tr_acc)}  |  "
            f"val loss {va_loss:.3f} top1 {kc.fmt_pct(va_acc)} top5 {kc.fmt_pct(va_top5)}{flag}"
        )

    with open(log_path, "w", newline="") as fh:
        import csv

        w = csv.DictWriter(fh, fieldnames=list(log_rows[0]))
        w.writeheader()
        w.writerows(log_rows)

    chance = 1.0 / len(vocab)
    print(
        f"\nbest val top-1 {kc.fmt_pct(best_acc)} at epoch {best_epoch} "
        f"(chance {kc.fmt_pct(chance)}) in {time.time() - t_start:.0f}s"
    )
    print(f"checkpoint: {args.out}\ntrain log : {log_path}")
    print("\nNext: eval.py on the held-out sessions — validation numbers above are for "
          "model selection only and are NOT the experiment's result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
