#!/usr/bin/env python3
"""eval.py — the actual experiment, on sessions the model has never seen.

Everything here is reported *separately for prose and for random strings*.
That contrast is the headline result:

    top-1(prose) - top-1(random)  ==  the share of the score that comes from
                                      linguistic redundancy, not from acoustics.

The random-string number is the honest measure of what the microphone can
discriminate.  The optional n-gram decoding stage makes the same point from the
other side: it inflates prose accuracy and collapses on random strings.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import torch

import kkr_common as kc
from train import KeyCNN, load_sessions


# --------------------------------------------------------------------------- #
# Character n-gram language model (interpolated, order-N)
# --------------------------------------------------------------------------- #


class CharNgramLM:
    """Jelinek-Mercer interpolated character n-gram over a fixed alphabet."""

    def __init__(self, alphabet: list[str], order: int = 5, lam: float = 0.75):
        self.alphabet = list(alphabet)
        self.index = {c: i for i, c in enumerate(self.alphabet)}
        self.order = order
        self.lam = lam
        self.counts: list[dict[str, np.ndarray]] = [dict() for _ in range(order)]
        self.totals: list[dict[str, float]] = [dict() for _ in range(order)]
        self._cache: dict[str, np.ndarray] = {}
        self._uniform = np.full(len(self.alphabet), 1.0 / len(self.alphabet))

    def fit(self, text: str) -> "CharNgramLM":
        keep = set(self.alphabet)
        cleaned = "".join(c if c in keep else " " for c in text.lower())
        cleaned = " ".join(cleaned.split())
        V = len(self.alphabet)
        for n in range(self.order):  # n = context length
            ctx_counts, ctx_tot = self.counts[n], self.totals[n]
            for i in range(n, len(cleaned)):
                ctx = cleaned[i - n : i]
                arr = ctx_counts.get(ctx)
                if arr is None:
                    arr = ctx_counts[ctx] = np.zeros(V)
                    ctx_tot[ctx] = 0.0
                arr[self.index[cleaned[i]]] += 1.0
                ctx_tot[ctx] += 1.0
        self._cache.clear()
        return self

    def _dist(self, ctx: str) -> np.ndarray:
        n = len(ctx)
        if n == 0:
            arr = self.counts[0].get("")
            tot = self.totals[0].get("", 0.0)
            if arr is None or tot == 0:
                return self._uniform
            return 0.9 * (arr / tot) + 0.1 * self._uniform
        lower = self._dist(ctx[1:])
        arr = self.counts[n].get(ctx)
        tot = self.totals[n].get(ctx, 0.0)
        if arr is None or tot == 0:
            return lower
        return self.lam * (arr / tot) + (1 - self.lam) * lower

    def logp(self, ctx: str) -> np.ndarray:
        ctx = ctx[-(self.order - 1) :] if self.order > 1 else ""
        out = self._cache.get(ctx)
        if out is None:
            out = self._cache[ctx] = np.log(np.maximum(self._dist(ctx), 1e-12))
        return out


def beam_decode(logp_acoustic: np.ndarray, lm: CharNgramLM, weight: float, beam: int = 32) -> np.ndarray:
    """Beam search over one contiguous character sequence. Returns class indices."""
    T, V = logp_acoustic.shape
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    beams = [("", 0.0)]
    back = []
    for t in range(T):
        lm_rows = np.stack([lm.logp(ctx) for ctx, _ in beams])            # (B, V)
        scores = lm_rows * weight + logp_acoustic[t][None, :]
        scores = scores + np.array([s for _, s in beams])[:, None]
        flat = scores.ravel()
        k = min(beam, flat.size)
        top = np.argpartition(-flat, k - 1)[:k]
        top = top[np.argsort(-flat[top])]
        parents, chars = np.divmod(top, V)
        back.append((parents, chars))
        beams = [
            (beams[p][0] + lm.alphabet[c], float(flat[p * V + c]))
            for p, c in zip(parents, chars)
        ]
    out = np.zeros(T, dtype=np.int64)
    b = 0
    for t in range(T - 1, -1, -1):
        parents, chars = back[t]
        out[t] = chars[b]
        b = parents[b]
    return out


def split_sequences(times: np.ndarray, max_gap_s: float) -> list[np.ndarray]:
    """Contiguous typing runs: a pause longer than `max_gap_s` starts a new one."""
    if times.size == 0:
        return []
    order = np.argsort(times)
    runs, cur = [], [int(order[0])]
    for prev, cur_i in zip(order[:-1], order[1:]):
        if times[cur_i] - times[prev] > max_gap_s:
            runs.append(np.array(cur))
            cur = []
        cur.append(cur_i)
    if len(cur):
        runs.append(np.array(cur))
    return runs


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def topk_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    if len(y) == 0:
        return {"n": 0, "top1": float("nan"), "top5": float("nan")}
    order = np.argsort(-logits, axis=1)
    top1 = float((order[:, 0] == y).mean())
    k = min(5, logits.shape[1])
    top5 = float((order[:, :k] == y[:, None]).any(axis=1).mean())
    return {"n": int(len(y)), "top1": top1, "top5": top5}


def neighbour_analysis(y_true: np.ndarray, y_pred: np.ndarray, vocab: list[str]) -> dict:
    """Are the mistakes physically adjacent keys, or are they arbitrary?"""
    dists, neigh = [], 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            continue
        d = kc.key_distance(vocab[t], vocab[p])
        if d is None:
            continue
        dists.append(d)
        neigh += d <= kc.NEIGHBOUR_RADIUS
    if not dists:
        return {"n_errors": 0}
    # Baseline: a wrong key drawn from the test label distribution.
    freq = np.bincount(y_true, minlength=len(vocab)).astype(float)
    base_d, base_n, wsum = 0.0, 0.0, 0.0
    for i, ci in enumerate(vocab):
        if freq[i] == 0:
            continue
        for j, cj in enumerate(vocab):
            if i == j or freq[j] == 0:
                continue
            d = kc.key_distance(ci, cj)
            if d is None:
                continue
            w = freq[i] * freq[j]
            base_d += w * d
            base_n += w * (d <= kc.NEIGHBOUR_RADIUS)
            wsum += w
    return {
        "n_errors": len(dists),
        "mean_error_distance": float(np.mean(dists)),
        "chance_error_distance": float(base_d / wsum) if wsum else float("nan"),
        "pct_errors_on_physical_neighbour": float(neigh / len(dists)),
        "chance_pct_neighbour": float(base_n / wsum) if wsum else float("nan"),
        "neighbour_radius_key_units": kc.NEIGHBOUR_RADIUS,
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    M = np.zeros((n, n), dtype=np.int64)
    np.add.at(M, (y_true, y_pred), 1)
    return M


def save_confusion(M: np.ndarray, vocab: list[str], path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = M.sum(axis=1, keepdims=True)
    N = M / np.maximum(row, 1)
    fig, ax = plt.subplots(figsize=(0.32 * len(vocab) + 3, 0.32 * len(vocab) + 2.5))
    im = ax.imshow(N, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(vocab)), vocab, fontsize=7, rotation=90)
    ax.set_yticks(range(len(vocab)), vocab, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confusion_csv(M: np.ndarray, vocab: list[str], path: str) -> None:
    import csv

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred"] + vocab)
        for i, k in enumerate(vocab):
            w.writerow([k] + M[i].tolist())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


@torch.no_grad()
def run_model(model, X: np.ndarray, device, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i : i + batch]).unsqueeze(1).to(device)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)


def load_eval_set(processed_dir: str, ids, vocab, norm):
    X, y, sids, modes, cfg = load_sessions(processed_dir, ids)
    idx = {k: i for i, k in enumerate(vocab)}
    keep = np.array([k in idx for k in y])
    dropped = Counter(y[~keep].tolist())
    if dropped:
        print(f"! dropping {int((~keep).sum())} test windows whose key is not in the model vocabulary: {dict(dropped)}")
    X, y, sids, modes = X[keep], y[keep], sids[keep], modes[keep]
    X = (X - norm["mu"]) / norm["sd"]
    yi = np.array([idx[k] for k in y], dtype=np.int64)
    return X, yi, sids, modes, cfg


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="models/keycnn.pt")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--test-sessions", nargs="*", default=None, help="default: the test list stored in the checkpoint")
    p.add_argument("--val-sessions", nargs="*", default=None, help="default: the val list stored in the checkpoint")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--raw-dir", default="data/raw", help="only used to check the LM corpus for prompt leakage")
    p.add_argument("--device", default=None)
    # language-model decoding
    p.add_argument("--lm-corpus", default=None, help="text file for the n-gram LM (enables LM decoding)")
    p.add_argument("--lm-order", type=int, default=5)
    p.add_argument("--lm-weight", default="auto", help="float, or 'auto' to tune on the validation sessions")
    p.add_argument("--lm-beam", type=int, default=32)
    p.add_argument("--lm-lambda", type=float, default=0.75)
    p.add_argument("--seq-gap-s", type=float, default=2.0, help="pause that separates two typing runs")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = list(ckpt["vocab"])
    norm = ckpt["norm"]
    splits = ckpt.get("splits", {})
    test_ids = args.test_sessions or splits.get("test") or []
    val_ids = args.val_sessions or splits.get("val") or []
    if not test_ids:
        raise SystemExit("no test session: pass --test-sessions (they must be sessions held out at training time)")
    kc.check_disjoint(train=splits.get("train", []), test=test_ids)

    device = torch.device(args.device) if args.device else torch.device("cpu")
    model = KeyCNN(len(vocab), tuple(ckpt["model"]["widths"]), ckpt["model"]["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    print(f"checkpoint : {args.checkpoint} (epoch {ckpt.get('epoch')}, val top-1 {kc.fmt_pct(ckpt.get('val_top1', float('nan')))})")
    print(f"train sess : {splits.get('train')}")
    print(f"test sess  : {test_ids}\n")

    X, y, sids, modes, _ = load_eval_set(args.processed_dir, test_ids, vocab, norm)
    logits = run_model(model, X, device)
    logp = torch.log_softmax(torch.from_numpy(logits), dim=1).numpy()
    pred = logits.argmax(axis=1)

    kc.ensure_dir(args.results_dir)
    chance = 1.0 / len(vocab)
    results = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "vocab": vocab,
        "chance_top1": chance,
        "splits": {"train": splits.get("train"), "val": val_ids, "test": test_ids},
        "n_test_windows": int(len(y)),
        "by_mode": {},
        "by_session": {},
        "overall": topk_metrics(logits, y),
    }

    # ---- headline: prose vs random ---------------------------------------- #
    print("=" * 72)
    print("PER-CHARACTER ACCURACY ON HELD-OUT SESSIONS (acoustic model alone)")
    print("=" * 72)
    print(f"{'regime':<10}{'n':>8}{'top-1':>10}{'top-5':>10}   (chance top-1 {kc.fmt_pct(chance)})")
    for mode in ("prose", "random"):
        m = modes == mode
        met = topk_metrics(logits[m], y[m])
        results["by_mode"][mode] = met
        if met["n"]:
            print(f"{mode:<10}{met['n']:>8}{kc.fmt_pct(met['top1']):>10}{kc.fmt_pct(met['top5']):>10}")
        else:
            print(f"{mode:<10}{'-':>8}   (no held-out session in this regime)")
    for other in sorted(set(modes.tolist()) - {"prose", "random"}):
        m = modes == other
        results["by_mode"][other] = topk_metrics(logits[m], y[m])

    p_, r_ = results["by_mode"].get("prose", {}), results["by_mode"].get("random", {})
    gap = None
    if p_.get("n") and r_.get("n"):
        gap = {
            "top1": p_["top1"] - r_["top1"],
            "top5": p_["top5"] - r_["top5"],
            "random_over_chance": r_["top1"] / chance,
        }
        results["prose_minus_random"] = gap
        print(
            f"\n>>> prose - random  =  {gap['top1'] * 100:+.1f} pts top-1, "
            f"{gap['top5'] * 100:+.1f} pts top-5"
        )
        print(
            "    That gap is the part of the score contributed by linguistic\n"
            "    redundancy, not by acoustics. The random-string number\n"
            f"    ({kc.fmt_pct(r_['top1'])} top-1, {gap['random_over_chance']:.1f}x chance) is the\n"
            "    honest measure of what the microphone actually discriminates."
        )
    else:
        print(
            "\n! only one regime is present in the held-out sessions — record at least "
            "one prose and one random session for the comparison that matters."
        )

    for sid in test_ids:
        m = sids == sid
        met = topk_metrics(logits[m], y[m])
        met["mode"] = str(modes[m][0]) if m.any() else "?"
        results["by_session"][sid] = met
    print("\nper session:")
    for sid, met in results["by_session"].items():
        print(f"  {sid:<38}{met['mode']:<8}{met['n']:>6}  top1 {kc.fmt_pct(met['top1'])}  top5 {kc.fmt_pct(met['top5'])}")

    # ---- confusion + physical-neighbour structure -------------------------- #
    print("\nconfusion matrices:")
    for name, mask in [("all", np.ones(len(y), bool)), ("prose", modes == "prose"), ("random", modes == "random")]:
        if not mask.any():
            continue
        M = confusion(y[mask], pred[mask], len(vocab))
        csv_path = os.path.join(args.results_dir, f"confusion_{name}.csv")
        png_path = os.path.join(args.results_dir, f"confusion_{name}.png")
        save_confusion_csv(M, vocab, csv_path)
        save_confusion(M, vocab, png_path, f"held-out sessions — {name} (row-normalised)")
        print(f"  {csv_path}\n  {png_path}")
        na = neighbour_analysis(y[mask], pred[mask], vocab)
        results.setdefault("neighbour_analysis", {})[name] = na
        if na.get("n_errors"):
            print(
                f"    errors on a physically adjacent key: {kc.fmt_pct(na['pct_errors_on_physical_neighbour'])} "
                f"(chance {kc.fmt_pct(na['chance_pct_neighbour'])}); "
                f"mean error distance {na['mean_error_distance']:.2f} vs {na['chance_error_distance']:.2f} key-units"
            )

    # top confusions overall
    Mall = confusion(y, pred, len(vocab))
    off = [(int(Mall[i, j]), vocab[i], vocab[j]) for i in range(len(vocab)) for j in range(len(vocab)) if i != j]
    off.sort(reverse=True)
    results["top_confusions"] = [{"true": t, "pred": p, "count": c} for c, t, p in off[:15]]
    print("\nmost frequent confusions (true -> predicted):")
    for c, t, pcell in off[:10]:
        d = kc.key_distance(t, pcell)
        print(f"  {t!r} -> {pcell!r}  x{c}" + (f"   [{d:.1f} key-units apart]" if d is not None else ""))

    per_key = {}
    for i, k in enumerate(vocab):
        n = int(Mall[i].sum())
        per_key[k] = {"n": n, "top1": float(Mall[i, i] / n) if n else float("nan")}
    results["per_key"] = per_key

    # ---- optional: n-gram language-model decoding -------------------------- #
    if args.lm_corpus:
        results["lm"] = run_lm_decoding(args, ckpt, model, device, vocab, norm, logp, y, sids, modes,
                                        val_ids, results)

    kc.write_json(os.path.join(args.results_dir, "eval_results.json"), results)
    write_report(os.path.join(args.results_dir, "report.md"), results)
    print(f"\nwrote {os.path.join(args.results_dir, 'eval_results.json')} and "
          f"{os.path.join(args.results_dir, 'report.md')}")
    return 0


def _lm_alphabet(vocab: list[str]) -> tuple[list[str], list[int]] | tuple[None, None]:
    """Map the model vocabulary to LM characters (space label -> ' ')."""
    chars, keep = [], []
    for i, k in enumerate(vocab):
        c = " " if k == "space" else (k if len(k) == 1 else None)
        if c is None:
            continue
        chars.append(c)
        keep.append(i)
    return (chars, keep) if len(chars) >= 2 else (None, None)


def _decode_accuracy(logp: np.ndarray, y: np.ndarray, t: np.ndarray, sids: np.ndarray,
                     lm: CharNgramLM, weight: float, beam: int, gap: float,
                     class_of_col: np.ndarray) -> float:
    """Beam-decode each contiguous typing run and return per-character accuracy."""
    correct = total = 0
    for sid in np.unique(sids):
        m = np.flatnonzero(sids == sid)
        for run in split_sequences(t[m], gap):
            rows = m[run]
            seq = logp[np.ix_(rows, class_of_col)]
            dec_cols = beam_decode(seq, lm, weight, beam)
            dec = class_of_col[dec_cols]
            correct += int((dec == y[rows]).sum())
            total += len(rows)
    return correct / max(total, 1)


def run_lm_decoding(args, ckpt, model, device, vocab, norm, logp, y, sids, modes, val_ids, results) -> dict:
    chars, keep = _lm_alphabet(vocab)
    if chars is None:
        print("\n! LM decoding skipped: the vocabulary has no single-character keys")
        return {"enabled": False, "reason": "vocabulary not character-like"}

    with open(args.lm_corpus, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    leak = prompt_leakage(args.raw_dir, results["splits"]["test"], text)
    if leak and leak["fraction"] > 0.2:
        print(
            f"\n! {kc.fmt_pct(leak['fraction'])} of the prompted lines of the test sessions appear "
            f"verbatim in the LM corpus.\n  The decoding result would be retrieval of memorised "
            "prompts, not language modelling. Use a corpus you never typed."
        )
    lm = CharNgramLM(chars, order=args.lm_order, lam=args.lm_lambda).fit(text)
    class_of_col = np.array(keep)
    print(f"\nn-gram LM: order {args.lm_order}, {len(text.split())} words of corpus, alphabet {len(chars)}")

    # times of the test windows, needed to cut the stream into typing runs
    t_all = _load_times(args.processed_dir, results["splits"]["test"], vocab)

    weight = args.lm_weight
    tuned = str(weight).lower() == "auto"
    if tuned:
        if not val_ids:
            weight = 1.0
            print("  no validation session available: using lm-weight = 1.0")
        else:
            weight = _tune_weight(args, ckpt, model, device, vocab, norm, lm, val_ids, class_of_col)
    weight = float(weight)

    out = {"enabled": True, "order": args.lm_order, "weight": weight, "tuned": tuned, "beam": args.lm_beam,
           "corpus": os.path.abspath(args.lm_corpus), "by_mode": {}}
    print("\n" + "=" * 72)
    print(f"WITH n-gram DECODING (weight {weight:g})")
    print("=" * 72)
    print(f"{'regime':<10}{'acoustic only':>16}{'+ language model':>18}{'delta':>10}")
    for mode in ("prose", "random"):
        m = modes == mode
        if not m.any():
            continue
        acc0 = results["by_mode"][mode]["top1"]
        acc1 = _decode_accuracy(logp[m], y[m], t_all[m], sids[m], lm, weight, args.lm_beam,
                                args.seq_gap_s, class_of_col)
        out["by_mode"][mode] = {"acoustic_top1": acc0, "lm_top1": acc1, "delta": acc1 - acc0}
        print(f"{mode:<10}{kc.fmt_pct(acc0):>16}{kc.fmt_pct(acc1):>18}{(acc1 - acc0) * 100:>+9.1f}")
    if "prose" in out["by_mode"] and "random" in out["by_mode"]:
        dp = out["by_mode"]["prose"]["delta"]
        dr = out["by_mode"]["random"]["delta"]
        out["prose_minus_random_lm"] = out["by_mode"]["prose"]["lm_top1"] - out["by_mode"]["random"]["lm_top1"]
        print(f"\n    The LM adds {dp * 100:+.1f} pts on prose and {dr * 100:+.1f} pts on random strings.")
        if dp > dr + 0.01:
            print(
                "    Same decoder, same acoustics, two regimes: whatever it buys on\n"
                "    prose it cannot buy on random strings. A headline figure obtained\n"
                "    this way is a statement about English, not about the keyboard."
            )
        elif weight == 0.0:
            print("    (weight 0 = decoding disabled; the tuning found no useful LM weight.)")
        else:
            print("    The LM did not help on prose here — too little acoustic signal to correct,\n"
                  "    or a corpus too small/too far from what was typed.")
    return out


def prompt_leakage(raw_dir: str, test_ids, lm_text: str) -> dict | None:
    """Fraction of the test sessions' prompted lines that appear in the LM corpus."""
    try:
        sessions = kc.list_sessions(raw_dir, list(test_ids))
    except Exception:
        return None
    hay = " ".join(lm_text.lower().split())
    lines = [" ".join(l.lower().split()) for s in sessions for l in (s.meta.get("prompt_lines") or [])]
    lines = [l for l in lines if len(l) > 20]
    if not lines:
        return None
    hits = sum(1 for l in lines if l in hay)
    return {"n_lines": len(lines), "n_in_corpus": hits, "fraction": hits / len(lines)}


def _load_times(processed_dir: str, ids, vocab) -> np.ndarray:
    idx = set(vocab)
    ts = []
    for sid in ids:
        with np.load(os.path.join(processed_dir, f"{sid}.npz"), allow_pickle=False) as z:
            t, yk = z["t"], z["y"].astype(str)
        ts.append(t[np.array([k in idx for k in yk])])
    return np.concatenate(ts)


def _tune_weight(args, ckpt, model, device, vocab, norm, lm, val_ids, class_of_col) -> float:
    """Pick the LM weight on the validation sessions — never on the test ones.

    Tuned on *prose* validation data only: the decoder exists to exploit English,
    and tuning it on random strings would trivially select weight 0.
    """
    Xv, yv, sv, mv, _ = load_eval_set(args.processed_dir, val_ids, vocab, norm)
    tv = _load_times(args.processed_dir, val_ids, vocab)
    prose = mv == "prose"
    if prose.any():
        Xv, yv, sv, tv = Xv[prose], yv[prose], sv[prose], tv[prose]
        print("  tuning lm-weight on the prose validation window(s) of:", sorted(set(sv.tolist())))
    else:
        print(
            "  ! no prose session in the validation split: the LM weight is being tuned on "
            "random strings and will collapse to 0. Hold out a prose session for validation."
        )
    lv = torch.log_softmax(torch.from_numpy(run_model(model, Xv, device)), dim=1).numpy()
    best, best_w = -1.0, 1.0
    for w in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        acc = _decode_accuracy(lv, yv, tv, sv, lm, w, args.lm_beam, args.seq_gap_s, class_of_col)
        print(f"    weight {w:>4}: val top-1 {kc.fmt_pct(acc)}")
        if acc > best:
            best, best_w = acc, w
    print(f"  chosen lm-weight = {best_w:g}")
    return best_w


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def write_report(path: str, r: dict) -> None:
    L = []
    A = L.append
    A("# Keystroke acoustic recognition — held-out session results\n")
    A(f"- Test sessions (never seen in training): `{r['splits']['test']}`")
    A(f"- Train sessions: `{r['splits']['train']}`")
    A(f"- Classes: {len(r['vocab'])} (chance top-1 {kc.fmt_pct(r['chance_top1'])})")
    A(f"- Test windows: {r['n_test_windows']}\n")

    A("## Per-character accuracy, acoustic model alone\n")
    A("| regime | n | top-1 | top-5 |")
    A("|---|---:|---:|---:|")
    for mode in ("prose", "random"):
        m = r["by_mode"].get(mode)
        if m and m["n"]:
            A(f"| {mode} | {m['n']} | {kc.fmt_pct(m['top1'])} | {kc.fmt_pct(m['top5'])} |")
    o = r["overall"]
    A(f"| **all** | {o['n']} | {kc.fmt_pct(o['top1'])} | {kc.fmt_pct(o['top5'])} |\n")

    g = r.get("prose_minus_random")
    if g:
        A("### The headline number\n")
        A(f"**prose - random = {g['top1'] * 100:+.1f} points of top-1** "
          f"({g['top5'] * 100:+.1f} points top-5).\n")
        A("That difference is the share of the score that comes from the redundancy of "
          "English, not from the acoustics. The random-string accuracy "
          f"({kc.fmt_pct(r['by_mode']['random']['top1'])}, {g['random_over_chance']:.1f}x chance) is the "
          "honest measure of what the microphone discriminates.\n")

    A("## Per session\n")
    A("| session | mode | n | top-1 | top-5 |")
    A("|---|---|---:|---:|---:|")
    for sid, m in r["by_session"].items():
        A(f"| `{sid}` | {m['mode']} | {m['n']} | {kc.fmt_pct(m['top1'])} | {kc.fmt_pct(m['top5'])} |")
    A("")

    na = (r.get("neighbour_analysis") or {}).get("all")
    if na and na.get("n_errors"):
        A("## Are the mistakes physically adjacent keys?\n")
        A(f"- Errors landing on a physical neighbour (<= {na['neighbour_radius_key_units']} key-units): "
          f"**{kc.fmt_pct(na['pct_errors_on_physical_neighbour'])}** vs {kc.fmt_pct(na['chance_pct_neighbour'])} expected by chance")
        A(f"- Mean distance of an error: **{na['mean_error_distance']:.2f}** key-units "
          f"vs {na['chance_error_distance']:.2f} by chance\n")

    if r.get("top_confusions"):
        A("Most frequent confusions:\n")
        A("| true | predicted | count | key-units apart |")
        A("|---|---|---:|---:|")
        for c in r["top_confusions"][:10]:
            d = kc.key_distance(c["true"], c["pred"])
            A(f"| `{c['true']}` | `{c['pred']}` | {c['count']} | {d:.1f} |" if d is not None
              else f"| `{c['true']}` | `{c['pred']}` | {c['count']} | - |")
        A("")

    lm = r.get("lm")
    if lm and lm.get("enabled"):
        A("## With n-gram language-model decoding\n")
        how = ("tuned on the prose validation session, never on the test sessions"
               if lm.get("tuned") else "set manually")
        A(f"Order-{lm['order']} character n-gram, beam {lm['beam']}, weight {lm['weight']:g} ({how}).\n")
        A("| regime | acoustic only | + language model | delta |")
        A("|---|---:|---:|---:|")
        for mode in ("prose", "random"):
            m = lm["by_mode"].get(mode)
            if m:
                A(f"| {mode} | {kc.fmt_pct(m['acoustic_top1'])} | {kc.fmt_pct(m['lm_top1'])} | "
                  f"{m['delta'] * 100:+.1f} pts |")
        A("")
        if "prose" in lm["by_mode"] and "random" in lm["by_mode"]:
            dp = lm["by_mode"]["prose"]["delta"] * 100
            dr = lm["by_mode"]["random"]["delta"] * 100
            A(f"The decoder is worth {dp:+.1f} points on prose and {dr:+.1f} points on random "
              "strings. Same acoustics, same decoder: the difference is English. A headline "
              "figure obtained this way describes the language model, not the keyboard.\n")

    A("## Confusion matrices\n")
    A("`confusion_all.png`, `confusion_prose.png`, `confusion_random.png` "
      "(row-normalised; CSV counts alongside).\n")
    A("## Method note\n")
    A("Windows are cut using the system keydown timestamps, so blind segmentation "
      "is deliberately out of scope. The split is per recording session, so no two "
      "presses of the same key from the same recording are ever on both sides of it.\n")

    kc.ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
