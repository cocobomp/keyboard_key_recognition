#!/usr/bin/env python3
"""make_synthetic_sessions.py — fake sessions to smoke-test the pipeline.

This is NOT part of the experiment: it fabricates keystroke audio (per-key
resonances + per-session channel colouring) so that preprocess/train/eval can be
exercised end to end without a microphone, on a machine that is not the Mac.
Any accuracy obtained on synthetic data says nothing about real keyboards.

    python make_synthetic_sessions.py --out-dir data/synthetic --sessions 6
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import string

import numpy as np

import kkr_common as kc

ALPHABET = string.ascii_lowercase + " "


def key_signature(key: str, n_res: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(abs(hash(("key", key))) % (2**32))
    freqs = rng.uniform(600, 9000, n_res)
    taus = rng.uniform(0.002, 0.012, n_res)
    amps = rng.uniform(0.4, 1.0, n_res)
    return freqs, taus, amps


def click(key: str, sr: int, rng: np.random.Generator, release: bool = False) -> np.ndarray:
    freqs, taus, amps = key_signature(key)
    dur = 0.03
    t = np.arange(int(dur * sr)) / sr
    sig = np.zeros_like(t)
    for f, tau, a in zip(freqs, taus, amps):
        f = f * rng.normal(1.0, 0.02)
        if release:
            a, tau = a * 0.55, tau * 0.7
        sig += a * np.exp(-t / tau) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    sig += rng.normal(0, 0.02, sig.shape) * np.exp(-t / 0.004)
    return sig / (np.abs(sig).max() + 1e-9)


def session_channel(rng: np.random.Generator):
    """A cheap stand-in for 'the mic moved and the room changed a bit'."""
    tilt = rng.uniform(-0.6, 0.6)
    gain = rng.uniform(0.5, 1.0)
    noise = rng.uniform(0.002, 0.010)
    reverb_ms = rng.uniform(4, 18)
    return tilt, gain, noise, reverb_ms


def make_session(out_dir: str, session_id: str, mode: str, n_keys: int, sr: int,
                 seed: int, corpus: str | None) -> None:
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)

    if mode == "prose":
        if corpus and os.path.isfile(corpus):
            with open(corpus, encoding="utf-8") as fh:
                text = " ".join(fh.read().lower().split())
            text = "".join(c for c in text if c in ALPHABET)
            start = prng.randrange(max(1, len(text) - n_keys - 1))
            keys = list(text[start : start + n_keys])
        else:
            keys = [prng.choice(ALPHABET) for _ in range(n_keys)]
    else:
        keys = [prng.choice(string.ascii_lowercase) if prng.random() > 0.15 else " " for _ in range(n_keys)]

    tilt, gain, noise, reverb_ms = session_channel(rng)
    t0 = 1000.0 + seed * 37.0  # arbitrary monotonic origin
    lead = 2.5
    times, cur = [], lead
    for _ in keys:
        cur += float(np.clip(rng.normal(0.19, 0.05), 0.07, 0.9))
        times.append(cur)
    total = cur + 1.5
    n = int(total * sr)
    audio = rng.normal(0, noise, n)

    beep_t = 0.8
    beep_dur = 0.06
    bt = np.arange(int(beep_dur * sr)) / sr
    env = np.minimum(1.0, np.minimum(bt, beep_dur - bt) / 0.004)
    audio[int(beep_t * sr) : int(beep_t * sr) + len(bt)] += 0.35 * env * np.sin(2 * np.pi * 1000 * bt)

    for k, t in zip(keys, times):
        label = "space" if k == " " else k
        # the audio onset is a few ms off from the logged event, like a real OS
        onset = t + float(rng.normal(0.0, 0.003))
        press = click(label, sr, rng) * rng.uniform(0.5, 1.0)
        rel = click(label, sr, rng, release=True) * rng.uniform(0.2, 0.5)
        hold = float(np.clip(rng.normal(0.085, 0.02), 0.04, 0.16))
        for sig, at in ((press, onset), (rel, onset + hold)):
            i = int(at * sr)
            if 0 <= i and i + len(sig) < n:
                audio[i : i + len(sig)] += sig
                j = i + int(reverb_ms * 1e-3 * sr)
                if j + len(sig) < n:
                    audio[j : j + len(sig)] += 0.3 * sig

    audio = audio + tilt * np.diff(audio, prepend=audio[:1])  # first-order tilt
    audio = gain * audio / (np.abs(audio).max() * 1.05 + 1e-9)

    path = kc.ensure_dir(os.path.join(out_dir, session_id))
    import soundfile as sf

    sf.write(os.path.join(path, "audio.wav"), audio.astype(np.float32), sr, subtype="PCM_16")

    with open(os.path.join(path, "keys.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kc.CSV_FIELDS))
        w.writeheader()
        for k, t in zip(keys, times):
            w.writerow({"timestamp": f"{t0 + t:.6f}", "key": "space" if k == " " else k,
                        "session_id": session_id, "mode": mode})

    checkpoints = [[float(i), t0 + i / sr] for i in range(0, n, int(0.5 * sr))]
    kc.write_json(
        os.path.join(path, "meta.json"),
        {
            "session_id": session_id, "mode": mode, "synthetic": True,
            "capture_version": kc.CAPTURE_VERSION, "samplerate": sr, "channels": 1,
            "audio_start_mono": t0, "clock_checkpoints": checkpoints, "clock_offset_s": 0.0,
            "frames_written": n, "duration_s": n / sr, "overflow_events": 0,
            "n_key_events": len(keys), "prompt_lines": [], "prompt_seed": seed,
            "sync_beep": {"freq_hz": 1000.0, "duration_s": beep_dur,
                          "t_start_mono_est": t0 + beep_t, "output_latency_s": 0.0,
                          "t_call_mono": t0 + beep_t, "t_return_mono": t0 + beep_t + beep_dur},
        },
    )
    print(f"{session_id}: {len(keys)} keys, {n / sr:.0f}s -> {path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="data/synthetic")
    p.add_argument("--sessions", type=int, default=6)
    p.add_argument("--keys-per-session", type=int, default=400)
    p.add_argument("--samplerate", type=int, default=kc.SAMPLE_RATE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corpus", default=os.path.join("corpus", "prose_en.txt"))
    args = p.parse_args(argv)

    for i in range(args.sessions):
        mode = "prose" if i % 2 == 0 else "random"
        make_session(args.out_dir, f"syn{i:02d}_{mode}", mode, args.keys_per_session,
                     args.samplerate, args.seed + i, args.corpus)
    print(f"\n{args.sessions} synthetic sessions in {args.out_dir} (fake audio — smoke test only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
