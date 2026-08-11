#!/usr/bin/env python3
"""preprocess.py — window the audio around each keydown and compute mel features.

For every logged keydown we cut a ~250 ms window around the press (it spans the
press transient *and* the release click) and turn it into a log-mel spectrogram.

Note on the threat model: we use the *system* keydown timestamps to place the
windows, so we deliberately side-step the blind-segmentation problem a real
attacker would face.  This POC measures "can the acoustics discriminate keys",
not "can keystrokes be found in an unlabelled recording".

Output (one file per session):

    data/processed/<session_id>.npz   X, y, t, session_id, mode, config
    data/processed/index.json         summary of everything processed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

import kkr_common as kc


# --------------------------------------------------------------------------- #
# Sync-beep verification
# --------------------------------------------------------------------------- #


def frame_to_mono(frames, meta) -> np.ndarray:
    """Inverse of kc.mono_to_frame (ignores clock_offset_s: this is the raw map)."""
    sr = float(meta.get("samplerate", kc.SAMPLE_RATE))
    f = np.atleast_1d(np.asarray(frames, dtype=np.float64))
    cps = meta.get("clock_checkpoints") or []
    if len(cps) >= 2:
        cp = np.asarray(cps, dtype=np.float64)
        order = np.argsort(cp[:, 0])
        fr, tm = cp[order, 0], cp[order, 1]
        out = np.interp(f, fr, tm)
        lo, hi = f < fr[0], f > fr[-1]
        out[lo] = tm[0] + (f[lo] - fr[0]) / sr
        out[hi] = tm[-1] + (f[hi] - fr[-1]) / sr
        return out
    return float(meta["audio_start_mono"]) + f / sr


def detect_beep(audio: np.ndarray, sr: int, freq: float, search_s: float = 12.0) -> float | None:
    """Return the onset (in frames) of the narrow-band sync tone, or None."""
    from scipy.signal import butter, sosfiltfilt

    seg = audio[: int(search_s * sr)]
    if seg.size < int(0.05 * sr):
        return None
    lo, hi = max(20.0, freq - 80.0), min(sr / 2 - 100.0, freq + 80.0)
    sos = butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band", output="sos")
    band = sosfiltfilt(sos, seg)

    win = max(1, int(0.005 * sr))
    env = np.sqrt(np.convolve(band**2, np.ones(win) / win, mode="same"))
    noise = np.median(env)
    peak = env.max()
    if peak < max(6.0 * noise, 1e-4):
        return None
    thr = noise + 0.3 * (peak - noise)
    above = np.flatnonzero(env >= thr)
    return float(above[0]) if above.size else None


def sync_report(audio: np.ndarray, sr: int, meta: dict) -> dict | None:
    beep = meta.get("sync_beep")
    if not beep:
        return None
    onset = detect_beep(audio, sr, float(beep.get("freq_hz", 1000.0)))
    if onset is None:
        return {"detected": False}
    t_detected = float(frame_to_mono(onset, meta)[0])
    t_logged = float(beep["t_start_mono_est"])
    lat = beep.get("output_latency_s")
    lat = float(lat) if lat is not None and np.isfinite(lat) else 0.0
    return {
        "detected": True,
        "onset_frame": onset,
        "onset_s": onset / sr,
        "t_detected_mono": t_detected,
        "t_logged_mono": t_logged,
        "delta_s": t_detected - t_logged,
        "output_latency_s": lat,
        # what is left once the (known) speaker path delay is removed
        "residual_s": t_detected - t_logged - lat,
    }


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


def melspec(win: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    import librosa

    m = librosa.feature.melspectrogram(
        y=win,
        sr=sr,
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        fmin=cfg["fmin"],
        fmax=min(cfg["fmax"], sr // 2),
        power=2.0,
        center=True,
    )
    return librosa.power_to_db(m, ref=1.0, top_db=None).astype(np.float32)


def process_session(sess: kc.Session, cfg: dict, keys: tuple[str, ...], args) -> dict:
    import soundfile as sf

    audio, sr = sf.read(sess.wav_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    meta = dict(sess.meta)
    if sr != meta.get("samplerate", sr):
        print(f"  ! WAV rate {sr} != meta rate {meta.get('samplerate')}, trusting the WAV")
        meta["samplerate"] = sr

    rep = sync_report(audio, sr, meta) if not args.no_sync_check else None
    if rep and rep.get("detected"):
        print(
            f"  sync beep at {rep['onset_s']:.3f} s | delta={rep['delta_s'] * 1e3:+.1f} ms "
            f"| residual after speaker latency={rep['residual_s'] * 1e3:+.1f} ms"
        )
        if abs(rep["residual_s"]) > args.sync_warn_ms / 1e3:
            print(
                f"  ! residual offset > {args.sync_warn_ms:.0f} ms — the monotonic anchor "
                "looks wrong for this session; inspect before trusting it."
            )
        if args.apply_sync_correction:
            meta["clock_offset_s"] = float(rep["residual_s"])
            print(f"  applying clock_offset_s = {meta['clock_offset_s'] * 1e3:+.1f} ms")
    elif rep is not None:
        print("  ! sync beep not found in the recording (falling back to the PortAudio anchor)")

    events = kc.read_key_events(sess.csv_path)
    keyset = set(keys)
    pre = int(round(args.pre_ms * 1e-3 * sr))
    post = int(round(args.post_ms * 1e-3 * sr))
    width = pre + post

    X, y, t_keep = [], [], []
    dropped_label = dropped_bounds = 0
    for ev in events:
        if ev["key"] not in keyset:
            dropped_label += 1
            continue
        center = int(round(kc.mono_to_frame(ev["timestamp"], meta)))
        start, stop = center - pre, center + post
        if start < 0 or stop > audio.size:
            dropped_bounds += 1
            continue
        win = audio[start:stop]
        if win.size != width:  # pragma: no cover - defensive
            dropped_bounds += 1
            continue
        S = melspec(win, sr, cfg)
        if args.normalize:
            S = (S - S.mean()) / (S.std() + 1e-5)
        X.append(S)
        y.append(ev["key"])
        t_keep.append(ev["timestamp"])

    if not X:
        raise SystemExit(
            f"session {sess.session_id}: no usable window. Check the key set "
            f"(--keys) and the clock mapping ({dropped_label} label drops, "
            f"{dropped_bounds} out-of-bounds)."
        )

    Xa = np.stack(X).astype(np.float16)
    ya = np.array(y, dtype="<U16")
    counts = Counter(y)
    out_cfg = dict(cfg)
    out_cfg.update(
        {
            "sr": sr,
            "pre_ms": args.pre_ms,
            "post_ms": args.post_ms,
            "window_ms": args.pre_ms + args.post_ms,
            "normalize": bool(args.normalize),
            "key_set": list(keys),
        }
    )
    out_path = os.path.join(args.out_dir, f"{sess.session_id}.npz")
    np.savez_compressed(
        out_path,
        X=Xa,
        y=ya,
        t=np.asarray(t_keep, dtype=np.float64),
        session_id=np.array(sess.session_id),
        mode=np.array(sess.mode),
        config=np.array(json.dumps(out_cfg)),
    )

    print(
        f"  {len(ya)} windows  shape={Xa.shape[1:]}  classes={len(counts)}  "
        f"dropped(label/bounds)={dropped_label}/{dropped_bounds}  -> {out_path}"
    )
    return {
        "session_id": sess.session_id,
        "mode": sess.mode,
        "n_windows": int(len(ya)),
        "n_classes": len(counts),
        "dropped_label": dropped_label,
        "dropped_bounds": dropped_bounds,
        "feature_shape": list(Xa.shape[1:]),
        "sync": rep,
        "counts": dict(sorted(counts.items())),
        "path": out_path,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--sessions", nargs="*", default=None, help="default: every session found")
    p.add_argument("--pre-ms", type=float, default=50.0, help="window start, before the keydown")
    p.add_argument("--post-ms", type=float, default=200.0, help="window end, after the keydown")
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=128)
    p.add_argument("--fmin", type=float, default=100.0)
    p.add_argument("--fmax", type=float, default=20000.0)
    p.add_argument("--keys", default="letters+space",
                   help="named set (letters, letters+space, printable, all) or a comma list")
    p.add_argument("--no-normalize", dest="normalize", action="store_false",
                   help="keep raw dB values instead of per-window z-scoring")
    p.add_argument("--no-sync-check", action="store_true")
    p.add_argument("--apply-sync-correction", action="store_true",
                   help="shift key timestamps by the beep residual (use only if it is large)")
    p.add_argument("--sync-warn-ms", type=float, default=25.0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    keys = kc.resolve_key_set(args.keys)
    cfg = {
        "n_mels": args.n_mels,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "fmin": args.fmin,
        "fmax": args.fmax,
    }
    kc.ensure_dir(args.out_dir)
    sessions = kc.list_sessions(args.raw_dir, args.sessions)
    if not sessions:
        print(f"no session found under {args.raw_dir}", file=sys.stderr)
        return 1

    index = []
    for s in sessions:
        print(f"[{s.session_id}] mode={s.mode}")
        index.append(process_session(s, cfg, keys, args))

    index_path = os.path.join(args.out_dir, "index.json")
    kc.write_json(index_path, {"config": cfg, "args": vars(args), "sessions": index})

    total = sum(e["n_windows"] for e in index)
    by_mode = Counter()
    for e in index:
        by_mode[e["mode"]] += e["n_windows"]
    print(f"\n{total} windows over {len(index)} sessions " f"({dict(by_mode)}) -> {index_path}")
    if total < 5000:
        print(f"note: the protocol targets >= 5000 labelled keystrokes (have {total}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
