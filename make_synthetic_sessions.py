#!/usr/bin/env python3
"""Sessions synthétiques (nuits + jours) pour valider la chaîne SANS micro.

Miroir du make_synthetic_sessions.py du projet clavier :
  - chaque session reçoit une « nuisance de canal » (tilt spectral, gain,
    niveau de bruit, réverbe simple) — c'est elle qui rend le split par
    session significatif même en synthétique ;
  - le format écrit est EXACTEMENT celui de record_audio.py (chunks FLAC +
    meta.json avec checkpoints d'horloge), pour exercer tout le pipeline ;
  - en plus : une vérité terrain data/synthetic/<sid>/ground_truth.csv
    (t_start_s, t_end_s, label) pour l'évaluation automatique.

Les « sons » sont des caricatures (ronflement = bruit basse fréquence modulé
périodique, grincement = glissando harmonique, porte = transitoire sourd,
vaisselle = tintements, parole = syllabes harmoniques). Les scores obtenus
dessus valident la plomberie, PAS les performances réelles.
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

import sed_common as sc

CHECKPOINT_PERIOD_S = 5.0  # linéaire en synthétique, inutile de densifier


# ---------------------------------------------------------------------------
# Briques sonores (16 kHz mono, float32, crête ~1 avant gain)
# ---------------------------------------------------------------------------

def _env(n: int, attack: float = 0.15, release: float = 0.3) -> np.ndarray:
    """Enveloppe attaque/plateau/relâche en fractions de la durée."""
    a, r = max(1, int(n * attack)), max(1, int(n * release))
    env = np.ones(n)
    env[:a] = np.linspace(0.0, 1.0, a)
    env[n - r :] = np.linspace(1.0, 0.0, r)
    return env


def snore_burst(rng: np.random.Generator, sr: int, voice: dict) -> np.ndarray:
    """Une inspiration ronflée : bruit passe-bas + composante gutturale."""
    dur = float(np.clip(rng.normal(1.1, 0.2), 0.7, 1.6))
    n = int(dur * sr)
    t = np.arange(n) / sr
    noise = rng.normal(0.0, 1.0, n)
    sos = butter(4, voice["cutoff_hz"], btype="low", fs=sr, output="sos")
    x = sosfilt(sos, noise)
    # « râle » périodique : train d'impulsions basse fréquence adouci
    f0 = voice["f0_hz"] * float(rng.uniform(0.92, 1.08))
    rattle = np.sign(np.sin(2 * np.pi * f0 * t)) * 0.5 + np.sin(2 * np.pi * f0 * t) * 0.5
    x = x * (0.6 + 0.4 * (rattle * 0.5 + 0.5))
    x *= _env(n, attack=0.25, release=0.35)
    return (x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def creak(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Grincement : glissando avec harmoniques et tremblement d'amplitude."""
    dur = float(rng.uniform(0.4, 1.2))
    n = int(dur * sr)
    t = np.arange(n) / sr
    f_start = float(rng.uniform(700, 1100))
    f_end = f_start * float(rng.uniform(1.2, 1.9))
    freq = np.linspace(f_start, f_end, n)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 5))
    x *= 0.7 + 0.3 * np.sin(2 * np.pi * float(rng.uniform(8, 14)) * t)
    x *= _env(n, attack=0.1, release=0.2)
    return (x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def door_thud(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Porte : choc sourd + résonance basse."""
    dur = float(rng.uniform(0.25, 0.6))
    n = int(dur * sr)
    t = np.arange(n) / sr
    noise = rng.normal(0.0, 1.0, n) * np.exp(-t / 0.03)
    sos = butter(4, 300, btype="low", fs=sr, output="sos")
    thud = sosfilt(sos, noise)
    f_res = float(rng.uniform(60, 120))
    res = np.sin(2 * np.pi * f_res * t) * np.exp(-t / 0.15)
    x = thud + 0.6 * res
    return (x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def dish_clink(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Vaisselle : sinusoïdes hautes amorties."""
    dur = float(rng.uniform(0.2, 0.5))
    n = int(dur * sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    for _ in range(int(rng.integers(3, 6))):
        f = float(rng.uniform(2000, 6500))
        tau = float(rng.uniform(0.03, 0.12))
        x += float(rng.uniform(0.4, 1.0)) * np.sin(2 * np.pi * f * t) * np.exp(-t / tau)
    return (x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def speech_like(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Pseudo-parole : syllabes harmoniques avec formants et pauses."""
    dur = float(rng.uniform(5.0, 12.0))
    n = int(dur * sr)
    x = np.zeros(n)
    f0 = float(rng.uniform(110, 220))
    pos = 0
    while pos < n - sr // 5:
        syl = int(float(rng.uniform(0.12, 0.22)) * sr)
        t = np.arange(syl) / sr
        tone = sum(
            np.sin(2 * np.pi * f0 * k * t + float(rng.uniform(0, 6.28))) / k
            for k in range(1, 9)
        )
        # deux « formants » très grossiers
        for f_formant in (float(rng.uniform(400, 900)), float(rng.uniform(1200, 2400))):
            sos = butter(2, [f_formant * 0.8, f_formant * 1.2], btype="band",
                         fs=sr, output="sos")
            tone = tone + 1.5 * sosfilt(sos, tone)
        tone *= _env(syl, attack=0.2, release=0.3)
        x[pos : pos + syl] += tone / (np.max(np.abs(tone)) + 1e-9)
        pos += syl + int(float(rng.uniform(0.05, 0.35)) * sr)
    return (x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


MAKERS = {
    "ronflement": snore_burst,  # voix passée à part
    "grincement": creak,
    "porte": door_thud,
    "vaisselle": dish_clink,
    "speech": speech_like,
}


# ---------------------------------------------------------------------------
# Planification d'une session
# ---------------------------------------------------------------------------

def plan_events(rng: np.random.Generator, kind: str, duration_s: float) -> list[dict]:
    """Liste [{t, label}] (t = début), avec épisodes de ronflement la nuit."""
    events: list[dict] = []
    margin = 5.0
    if kind == "night":
        # 2-4 épisodes de ronflement couvrant ~20-50 % de la nuit
        n_epi = int(rng.integers(2, 5))
        starts = np.sort(rng.uniform(margin, duration_s * 0.8, n_epi))
        for s in starts:
            epi_dur = float(rng.uniform(0.06, 0.16)) * duration_s
            t = float(s)
            period = float(rng.uniform(3.2, 5.0))
            while t < min(s + epi_dur, duration_s - margin):
                events.append({"t": t, "label": "ronflement"})
                t += period * float(rng.uniform(0.9, 1.1))
        for _ in range(int(rng.integers(1, 4))):
            events.append({"t": float(rng.uniform(margin, duration_s - margin)),
                           "label": "grincement"})
        if rng.random() < 0.7:
            events.append({"t": float(rng.uniform(margin, duration_s - margin)),
                           "label": "porte"})
    else:
        counts = {"porte": (2, 5), "grincement": (2, 6), "vaisselle": (3, 8),
                  "speech": (1, 3)}
        for label, (lo, hi) in counts.items():
            for _ in range(int(rng.integers(lo, hi + 1))):
                events.append({"t": float(rng.uniform(margin, duration_s - margin - 15)),
                               "label": label})
    events.sort(key=lambda e: e["t"])
    return events


def session_channel(rng: np.random.Generator) -> dict:
    """La nuisance « le micro a bougé entre les sessions »."""
    return {
        "tilt": float(rng.uniform(-0.35, 0.35)),   # filtre 1er ordre
        "gain": float(rng.uniform(0.5, 1.4)),
        "noise": float(rng.uniform(0.0008, 0.004)),  # fond ~ -60..-48 dBFS
        "reverb_ms": float(rng.uniform(8, 35)),
        "snore_cutoff_hz": float(rng.uniform(250, 450)),
        "snore_f0_hz": float(rng.uniform(70, 110)),
    }


def make_session(out_dir: str, session_id: str, kind: str, duration_s: float,
                 sr: int, seed: int, chunk_s: float, utc_start: float) -> dict:
    rng = np.random.default_rng(seed)
    chan = session_channel(rng)
    voice = {"cutoff_hz": chan["snore_cutoff_hz"], "f0_hz": chan["snore_f0_hz"]}

    n = int(duration_s * sr)
    audio = rng.normal(0.0, chan["noise"], n).astype(np.float32)
    # fond légèrement rose : lissage du bruit blanc
    audio = 0.6 * audio + 0.4 * np.convolve(audio, np.ones(8) / 8, mode="same").astype(
        np.float32
    )

    truth: list[tuple[float, float, str]] = []
    for ev in plan_events(rng, kind, duration_s):
        label = ev["label"]
        clip = (snore_burst(rng, sr, voice) if label == "ronflement"
                else MAKERS[label](rng, sr))
        amp = {"ronflement": 0.28, "grincement": 0.14, "porte": 0.30,
               "vaisselle": 0.12, "speech": 0.16}[label] * float(rng.uniform(0.7, 1.3))
        i0 = int(ev["t"] * sr)
        i1 = min(n, i0 + len(clip))
        if i1 <= i0:
            continue
        audio[i0:i1] += amp * clip[: i1 - i0]
        truth.append((i0 / sr, i1 / sr, label))

    # réverbe pas chère : copie retardée atténuée
    d = int(chan["reverb_ms"] * 1e-3 * sr)
    if d > 0:
        audio[d:] += 0.25 * audio[: n - d].copy()
    # tilt spectral 1er ordre + gain
    b = chan["tilt"]
    audio = audio + b * np.concatenate(([0.0], audio[:-1]))
    audio = (chan["gain"] * audio).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.99:
        audio *= 0.99 / peak

    # -- écriture au format record_audio.py -------------------------------
    sdir = sc.ensure_dir(os.path.join(out_dir, session_id))
    chunk_frames = int(chunk_s * sr)
    chunks = []
    for ci, start in enumerate(range(0, n, chunk_frames)):
        stop = min(n, start + chunk_frames)
        fname = f"chunk_{ci:03d}.flac"
        sf.write(os.path.join(sdir, fname), audio[start:stop], sr,
                 format="FLAC", subtype="PCM_16")
        chunks.append({"file": fname, "start_frame": start, "n_frames": stop - start})

    t0 = 1000.0 + seed * 37.0  # origine monotone arbitraire mais cohérente
    checkpoints = [[float(i), t0 + i / sr]
                   for i in range(0, n, int(CHECKPOINT_PERIOD_S * sr))]
    meta = {
        "session_id": session_id,
        "kind": kind,
        "tag": "synthetic",
        "capture_version": sc.CAPTURE_VERSION,
        "created_utc": datetime.fromtimestamp(utc_start, tz=timezone.utc).isoformat(),
        "t_utc_start_epoch": utc_start,
        "samplerate": sr,
        "channels": 1,
        "device": "synthetic",
        "audio_start_mono": t0,
        "clock_checkpoints": checkpoints,
        "clock_offset_s": 0.0,
        "chunks": chunks,
        "frames_written": n,
        "duration_s": n / sr,
        "overflow_events": 0,
        "synthetic": True,
        "channel_nuisance": chan,
    }
    sc.write_json(os.path.join(sdir, "meta.json"), meta)

    with open(os.path.join(sdir, "ground_truth.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_start_s", "t_end_s", "label"])
        for t_a, t_b, label in truth:
            w.writerow([f"{t_a:.3f}", f"{t_b:.3f}", label])
    return {"session_id": session_id, "kind": kind, "n_events": len(truth),
            "duration_s": n / sr}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default="data/synthetic")
    p.add_argument("--nights", type=int, default=4)
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--night-min", type=float, default=20.0,
                   help="durée d'une nuit synthétique (minutes)")
    p.add_argument("--day-min", type=float, default=10.0)
    p.add_argument("--chunk-min", type=float, default=10.0,
                   help="taille des chunks FLAC (minutes), comme record_audio")
    p.add_argument("--short", action="store_true",
                   help="mode smoke-test : nuits 6 min, jours 3 min, chunks 2 min")
    p.add_argument("--samplerate", type=int, default=sc.SAMPLE_RATE)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.short:
        args.night_min, args.day_min, args.chunk_min = 6.0, 3.0, 2.0
    if args.night_min > 90:
        print("attention : génération entièrement en RAM, >90 min sera lourd")

    base_utc = datetime(2026, 8, 10, 21, 0, tzinfo=timezone.utc).timestamp()
    made = []
    seed = args.seed
    for i in range(args.nights):
        made.append(make_session(
            args.out_dir, f"synnight{i:02d}", "night", args.night_min * 60.0,
            args.samplerate, seed, args.chunk_min * 60.0,
            base_utc + i * 86_400,
        ))
        seed += 1
    for i in range(args.days):
        made.append(make_session(
            args.out_dir, f"synday{i:02d}", "day", args.day_min * 60.0,
            args.samplerate, seed, args.chunk_min * 60.0,
            base_utc + i * 86_400 + 15 * 3600,  # 12h locales le jour suivant
        ))
        seed += 1

    print(f"{len(made)} sessions synthétiques dans {args.out_dir}/ :")
    for m in made:
        print(f"  {m['session_id']:<12} {m['kind']:<6} "
              f"{m['duration_s'] / 60:5.1f} min  {m['n_events']:4d} événements vrais")
    print("\nRappel : ces sons valident la PLOMBERIE, pas les performances réelles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
