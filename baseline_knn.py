#!/usr/bin/env python3
"""baseline_knn.py — k-NN on MFCC summary features, as a baseline for the CNN.

A deliberately simple, non-parametric reference point: turn each keystroke window
into a fixed MFCC descriptor (mean + std of the MFCCs and their deltas over time)
and classify by nearest neighbours. If the CNN does not clearly beat this, the CNN
is not buying much.

The MFCCs are the DCT of the log-mel spectrograms already computed by
preprocess.py, so this baseline sees exactly the same windows, feature config and
per-session split as train.py/eval.py — the comparison is apples to apples. There
is, as everywhere in this repo, no random window-level split: train/val/test are
whole, disjoint recording sessions.

    python baseline_knn.py --split-file splits.json --lm-corpus corpus/lm_train.txt
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
from scipy.fftpack import dct

import kkr_common as kc


# --------------------------------------------------------------------------- #
# Data (torch-free loader, same .npz files as train.py)
# --------------------------------------------------------------------------- #


def load_sessions(processed_dir: str, session_ids):
    Xs, ys, sids, modes, cfg = [], [], [], [], None
    for sid in session_ids:
        path = os.path.join(processed_dir, f"{sid}.npz")
        if not os.path.isfile(path):
            raise SystemExit(f"missing processed session: {path} (run preprocess.py first)")
        with np.load(path, allow_pickle=False) as z:
            X = z["X"].astype(np.float32)
            y = z["y"].astype(str)
            mode = str(z["mode"])
            c = json.loads(str(z["config"]))
        cmp = lambda d: {k: d[k] for k in d if k != "key_set"}  # noqa: E731
        if cfg is None:
            cfg = c
        elif cmp(c) != cmp(cfg):
            raise SystemExit(f"session {sid} was preprocessed with a different feature config.")
        Xs.append(X)
        ys.append(y)
        sids.extend([sid] * len(y))
        modes.extend([mode] * len(y))
    if not Xs:
        raise SystemExit("no session to load")
    return np.concatenate(Xs), np.concatenate(ys), np.array(sids), np.array(modes), cfg


# --------------------------------------------------------------------------- #
# Features: MFCC summary from the stored log-mel spectrograms
# --------------------------------------------------------------------------- #


def mfcc_features(X: np.ndarray, n_mfcc: int, drop_c0: bool = True) -> np.ndarray:
    """X: (n, n_mels, n_frames) log-mel -> (n, D) MFCC mean/std + delta mean/std."""
    # MFCC = DCT-II along the mel axis, keep the low-order coefficients.
    coeffs = dct(X, type=2, axis=1, norm="ortho")
    lo = 1 if drop_c0 else 0
    m = coeffs[:, lo : lo + n_mfcc, :]                       # (n, n_mfcc, T)
    dm = np.diff(m, axis=2) if m.shape[2] > 1 else np.zeros_like(m)
    feats = np.concatenate(
        [m.mean(2), m.std(2), dm.mean(2), dm.std(2)], axis=1  # (n, 4*n_mfcc)
    )
    return feats.astype(np.float32)


# --------------------------------------------------------------------------- #
# k-NN (numpy, distance-weighted class scores so we get top-1 and top-5)
# --------------------------------------------------------------------------- #


def knn_scores(Xtr, ytr, Xte, n_classes, k, weighting="distance", metric="euclidean", chunk=512):
    if metric == "cosine":
        Xtr = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8)
        Xte = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    tr_sq = (Xtr**2).sum(1)
    scores = np.zeros((len(Xte), n_classes), dtype=np.float64)
    for i in range(0, len(Xte), chunk):
        xb = Xte[i : i + chunk]
        d2 = (xb**2).sum(1, keepdims=True) - 2 * xb @ Xtr.T + tr_sq[None, :]
        np.maximum(d2, 0, out=d2)
        kk = min(k, Xtr.shape[0])
        nn = np.argpartition(d2, kk - 1, axis=1)[:, :kk]           # (b, k)
        rows = np.repeat(np.arange(xb.shape[0]), kk)
        cols = nn.ravel()
        if weighting == "distance":
            w = 1.0 / (np.sqrt(d2[rows, cols]) + 1e-6)
        else:
            w = np.ones_like(rows, dtype=np.float64)
        np.add.at(scores[i : i + chunk], (rows, ytr[cols]), w)
    return scores


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def topk_metrics(scores: np.ndarray, y: np.ndarray) -> dict:
    if len(y) == 0:
        return {"n": 0, "top1": float("nan"), "top5": float("nan")}
    order = np.argsort(-scores, axis=1)
    top1 = float((order[:, 0] == y).mean())
    k = min(5, scores.shape[1])
    top5 = float((order[:, :k] == y[:, None]).any(axis=1).mean())
    return {"n": int(len(y)), "top1": top1, "top5": top5}


def resolve_splits(args):
    train, val, test = args.train_sessions or [], args.val_sessions or [], args.test_sessions or []
    if args.split_file:
        spec = kc.read_json(args.split_file)
        train = train or list(spec.get("train", []))
        val = val or list(spec.get("val", []))
        test = test or list(spec.get("test", []))
    if not train or not test:
        raise SystemExit("baseline needs explicit --train-sessions and --test-sessions (or --split-file).")
    if not val and len(train) >= 2:
        val = [train[-1]]
        train = train[:-1]
        print(f"! no --val-sessions: holding out '{val[0]}' from train for tuning k (per-session, never random).")
    kc.check_disjoint(train=train, val=val, test=test)
    return train, val, test


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--train-sessions", nargs="*", default=None)
    p.add_argument("--val-sessions", nargs="*", default=None)
    p.add_argument("--test-sessions", nargs="*", default=None)
    p.add_argument("--split-file", default=None)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--n-mfcc", type=int, default=20)
    p.add_argument("--keep-c0", action="store_true", help="keep the 0th cepstral coeff (overall energy)")
    p.add_argument("--k", type=int, default=None, help="neighbours; default: tuned on the validation session")
    p.add_argument("--metric", choices=("euclidean", "cosine"), default="cosine")
    p.add_argument("--weighting", choices=("distance", "uniform"), default="distance")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    train_ids, val_ids, test_ids = resolve_splits(args)
    print(f"train sessions: {train_ids}\nval sessions  : {val_ids}\ntest sessions : {test_ids}")

    Xtr_raw, ytr, _, _, cfg = load_sessions(args.processed_dir, train_ids)
    vocab = sorted(set(ytr.tolist()))
    idx = {k: i for i, k in enumerate(vocab)}
    ytr_i = np.array([idx[k] for k in ytr], dtype=np.int64)

    def featurize(ids):
        X, y, sids, modes, _ = load_sessions(args.processed_dir, ids)
        keep = np.array([k in idx for k in y])
        F = mfcc_features(X[keep], args.n_mfcc, drop_c0=not args.keep_c0)
        yi = np.array([idx[k] for k in y[keep]], dtype=np.int64)
        return F, yi, sids[keep], modes[keep]

    Ftr = mfcc_features(Xtr_raw, args.n_mfcc, drop_c0=not args.keep_c0)
    # Standardise features on the TRAIN sessions only.
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr = (Ftr - mu) / sd
    print(f"\nMFCC descriptor: {Ftr.shape[1]} dims (n_mfcc={args.n_mfcc}, drop_c0={not args.keep_c0})")
    print(f"train windows: {len(ytr_i)}  classes: {len(vocab)}  (chance top-1 {kc.fmt_pct(1 / len(vocab))})")

    # ---- choose k on the validation session (never on the test set) -------- #
    k = args.k
    if k is None:
        if not val_ids:
            k = 5
            print("no validation session: using k = 5")
        else:
            Fva, yva, _, _ = featurize(val_ids)
            Fva = (Fva - mu) / sd
            best = (-1.0, 5)
            print(f"tuning k on validation session(s) {val_ids}:")
            for kk in (1, 3, 5, 9, 15, 25, 41):
                acc = topk_metrics(knn_scores(Ftr, ytr_i, Fva, len(vocab), kk, args.weighting, args.metric), yva)["top1"]
                print(f"  k={kk:>3}: val top-1 {kc.fmt_pct(acc)}")
                if acc > best[0]:
                    best = (acc, kk)
            k = best[1]
            print(f"chosen k = {k}")

    # ---- evaluate on the held-out test sessions ---------------------------- #
    Fte, yte, sids, modes = featurize(test_ids)
    Fte = (Fte - mu) / sd
    scores = knn_scores(Ftr, ytr_i, Fte, len(vocab), k, args.weighting, args.metric)
    pred = scores.argmax(1)

    chance = 1.0 / len(vocab)
    results = {
        "baseline": "knn-mfcc",
        "k": k, "n_mfcc": args.n_mfcc, "metric": args.metric, "weighting": args.weighting,
        "drop_c0": not args.keep_c0, "feature_dims": int(Ftr.shape[1]),
        "vocab": vocab, "chance_top1": chance,
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        "overall": topk_metrics(scores, yte), "by_mode": {}, "by_session": {},
    }

    print("\n" + "=" * 72)
    print(f"k-NN / MFCC BASELINE — HELD-OUT SESSIONS (k={k}, {args.metric}, {args.weighting})")
    print("=" * 72)
    print(f"{'regime':<10}{'n':>8}{'top-1':>10}{'top-5':>10}   (chance top-1 {kc.fmt_pct(chance)})")
    for mode in ("prose", "random"):
        m = modes == mode
        met = topk_metrics(scores[m], yte[m])
        results["by_mode"][mode] = met
        if met["n"]:
            print(f"{mode:<10}{met['n']:>8}{kc.fmt_pct(met['top1']):>10}{kc.fmt_pct(met['top5']):>10}")
    for other in sorted(set(modes.tolist()) - {"prose", "random"}):
        results["by_mode"][other] = topk_metrics(scores[modes == other], yte[modes == other])

    p_, r_ = results["by_mode"].get("prose", {}), results["by_mode"].get("random", {})
    if p_.get("n") and r_.get("n"):
        results["prose_minus_random"] = {"top1": p_["top1"] - r_["top1"], "top5": p_["top5"] - r_["top5"]}
        print(f"\n>>> prose - random = {(p_['top1'] - r_['top1']) * 100:+.1f} pts top-1 "
              f"(random {kc.fmt_pct(r_['top1'])} = {r_['top1'] / chance:.1f}x chance)")

    for sid in test_ids:
        m = sids == sid
        met = topk_metrics(scores[m], yte[m])
        met["mode"] = str(modes[m][0]) if m.any() else "?"
        results["by_session"][sid] = met

    # compare against the CNN if its eval ran into the same results dir
    cnn_path = os.path.join(args.results_dir, "eval_results.json")
    if os.path.isfile(cnn_path):
        try:
            cnn = kc.read_json(cnn_path)
            results["cnn_comparison"] = compare_to_cnn(cnn, results)
        except Exception:
            pass

    kc.ensure_dir(args.results_dir)
    out = os.path.join(args.results_dir, "baseline_knn.json")
    kc.write_json(out, results)
    print(f"\nwrote {out}")
    if "cnn_comparison" in results:
        print_comparison(results["cnn_comparison"])
    else:
        print("(run eval.py into the same --results-dir to print a CNN-vs-baseline table)")
    return 0


def compare_to_cnn(cnn: dict, knn: dict) -> dict:
    out = {"chance_top1": knn["chance_top1"], "by_mode": {}}
    for mode in ("prose", "random"):
        c = (cnn.get("by_mode") or {}).get(mode)
        k = knn["by_mode"].get(mode)
        if c and k and c.get("n") and k.get("n"):
            out["by_mode"][mode] = {"cnn_top1": c["top1"], "knn_top1": k["top1"],
                                    "cnn_top5": c["top5"], "knn_top5": k["top5"]}
    return out


def print_comparison(cmp: dict) -> None:
    print("\n" + "=" * 72)
    print("CNN vs k-NN/MFCC baseline (acoustic model, held-out sessions)")
    print("=" * 72)
    print(f"{'regime':<10}{'CNN top-1':>12}{'kNN top-1':>12}{'CNN top-5':>12}{'kNN top-5':>12}")
    for mode, m in cmp["by_mode"].items():
        print(f"{mode:<10}{kc.fmt_pct(m['cnn_top1']):>12}{kc.fmt_pct(m['knn_top1']):>12}"
              f"{kc.fmt_pct(m['cnn_top5']):>12}{kc.fmt_pct(m['knn_top5']):>12}")
    print(f"(chance top-1 {kc.fmt_pct(cmp['chance_top1'])})")


if __name__ == "__main__":
    raise SystemExit(main())
