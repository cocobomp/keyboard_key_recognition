"""Shared helpers for the keystroke acoustic recognition POC.

Kept deliberately small and dependency-light: capture.py must be able to run
with only sounddevice/pynput installed, while preprocess/train/eval pull in the
heavier scientific stack.
"""

from __future__ import annotations

import csv
import json
import os
import string
from dataclasses import dataclass
from typing import Iterable, Sequence

SAMPLE_RATE = 48_000
CAPTURE_VERSION = 1

# --------------------------------------------------------------------------- #
# Key labels
# --------------------------------------------------------------------------- #
# Canonical labels are lowercase single characters for printable keys, and short
# lowercase names for the rest ("space", "enter", "backspace", ...).  Note that
# the *physical* key is what the acoustics can possibly carry: shift+a and a are
# the same switch, so we always fold to the unshifted lowercase form.

_SHIFT_FOLD = {
    "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7",
    "*": "8", "(": "9", ")": "0", "_": "-", "+": "=", "{": "[", "}": "]",
    "|": "\\", ":": ";", '"': "'", "<": ",", ">": ".", "?": "/", "~": "`",
}

LETTERS = tuple(string.ascii_lowercase)
DIGITS = tuple(string.digits)
PUNCT = tuple("-=[]\\;',./`")
MODIFIERS = ("shift", "ctrl", "alt", "cmd", "caps_lock", "fn")
CONTROL = ("space", "enter", "backspace", "tab", "esc", "delete")

KEY_SETS = {
    "letters": LETTERS,
    "letters+space": LETTERS + ("space",),
    "letters+space+punct": LETTERS + ("space",) + PUNCT,
    "printable": LETTERS + DIGITS + PUNCT + ("space",),
    "all": LETTERS + DIGITS + PUNCT + CONTROL + MODIFIERS,
}


def canonical_label(raw: str) -> str:
    """Normalise a raw key string to its canonical (physical) label."""
    if raw is None:
        return ""
    s = str(raw)
    if len(s) == 1:
        s = _SHIFT_FOLD.get(s, s)
        return s.lower()
    return s.strip().lower()


def resolve_key_set(spec: str) -> tuple[str, ...]:
    """`spec` is either a named set (see KEY_SETS) or a comma separated list."""
    if spec in KEY_SETS:
        return KEY_SETS[spec]
    keys = tuple(canonical_label(k) for k in spec.split(",") if k.strip())
    if not keys:
        raise ValueError(f"empty key set: {spec!r}")
    return keys


# --------------------------------------------------------------------------- #
# Physical layout (US QWERTY, MacBook-ish staggering) for the neighbour analysis
# --------------------------------------------------------------------------- #

_ROWS = [
    ("`1234567890-=", 0.0, 0.0),
    ("qwertyuiop[]\\", 1.5, 1.0),
    ("asdfghjkl;'", 1.75, 2.0),
    ("zxcvbnm,./", 2.25, 3.0),
]

KEY_POS: dict[str, tuple[float, float]] = {}
for _chars, _x0, _y in _ROWS:
    for _i, _c in enumerate(_chars):
        KEY_POS[_c] = (_x0 + _i, _y)
KEY_POS["space"] = (5.0, 4.0)


def key_distance(a: str, b: str) -> float | None:
    """Euclidean distance in key-units between two physical keys."""
    pa, pb = KEY_POS.get(a), KEY_POS.get(b)
    if pa is None or pb is None:
        return None
    return ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5


NEIGHBOUR_RADIUS = 1.45  # includes the diagonal of a staggered row


# --------------------------------------------------------------------------- #
# Session I/O
# --------------------------------------------------------------------------- #

CSV_FIELDS = ("timestamp", "key", "session_id", "mode")


@dataclass
class Session:
    session_id: str
    path: str
    mode: str
    meta: dict

    @property
    def wav_path(self) -> str:
        return os.path.join(self.path, "audio.wav")

    @property
    def csv_path(self) -> str:
        return os.path.join(self.path, "keys.csv")


def list_sessions(raw_dir: str, only: Sequence[str] | None = None) -> list[Session]:
    """Discover capture sessions under `raw_dir` (one sub-directory each)."""
    sessions = []
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"no such raw directory: {raw_dir}")
    for name in sorted(os.listdir(raw_dir)):
        path = os.path.join(raw_dir, name)
        meta_path = os.path.join(path, "meta.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)
        sid = meta.get("session_id", name)
        if only and sid not in only:
            continue
        sessions.append(Session(sid, path, meta.get("mode", "unknown"), meta))
    if only:
        found = {s.session_id for s in sessions}
        missing = [s for s in only if s not in found]
        if missing:
            raise FileNotFoundError(f"sessions not found in {raw_dir}: {missing}")
    return sessions


def read_key_events(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        out.append(
            {
                "timestamp": float(r["timestamp"]),
                "key": canonical_label(r["key"]),
                "session_id": r["session_id"],
                "mode": r["mode"],
            }
        )
    out.sort(key=lambda r: r["timestamp"])
    return out


def mono_to_frame(t_mono, meta):
    """Map monotonic timestamps to audio frame indices.

    Uses the piecewise (frame, t_monotonic) checkpoints written by capture.py,
    which makes the mapping robust to clock drift *and* to dropped input buffers
    (an overflow shifts every later sample; a fresh checkpoint re-anchors it).
    Falls back to a single linear mapping for sessions without checkpoints.
    """
    import numpy as np

    sr = float(meta.get("samplerate", SAMPLE_RATE))
    t = np.atleast_1d(np.asarray(t_mono, dtype=np.float64))
    cps = meta.get("clock_checkpoints") or []
    offset = float(meta.get("clock_offset_s", 0.0))  # sync-beep correction

    if len(cps) >= 2:
        cp = np.asarray(cps, dtype=np.float64)
        frames, times = cp[:, 0], cp[:, 1]
        order = np.argsort(times)
        frames, times = frames[order], times[order]
        out = np.interp(t + offset, times, frames)
        # linear extrapolation at the edges, at the nominal sample rate
        lo = (t + offset) < times[0]
        hi = (t + offset) > times[-1]
        out[lo] = frames[0] + (t[lo] + offset - times[0]) * sr
        out[hi] = frames[-1] + (t[hi] + offset - times[-1]) * sr
    else:
        t0 = float(meta["audio_start_mono"])
        out = (t + offset - t0) * sr

    return out if np.ndim(t_mono) else float(out[0])


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def read_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def check_disjoint(**splits: Iterable[str]) -> None:
    """Hard guard against the one mistake that invalidates the whole experiment."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(splits[a]) & set(splits[b])
            if overlap:
                raise SystemExit(
                    f"FATAL: sessions {sorted(overlap)} appear in both '{a}' and "
                    f"'{b}'. The split must be strictly per-session — overlapping "
                    "sessions would measure memorisation, not recognition."
                )
