#!/usr/bin/env python3
"""capture.py — synchronised microphone + keystroke capture (macOS).

One run == one *session*.  A session produces:

    data/raw/<session_id>/audio.wav   48 kHz mono WAV
    data/raw/<session_id>/keys.csv    timestamp,key,session_id,mode
    data/raw/<session_id>/meta.json   clock mapping, sync beep, prompts, config

Clock discipline
----------------
Keystrokes and audio are dated with the *same* monotonic clock
(`time.monotonic()`), so nothing depends on wall-clock adjustments (NTP, sleep,
DST).  The audio side is anchored via PortAudio's ADC timestamps, converted once
into the monotonic frame of reference, and re-checkpointed roughly twice a
second.  Those checkpoints let preprocess.py map a keystroke time to a sample
index even if the input stream drops buffers mid-session.

A short 1 kHz beep is played and logged at the start of every session as an
independent safety net: preprocess.py can locate it in the recording and report
(or correct) any residual offset.

macOS: keystroke capture requires the Accessibility permission for the app that
runs this script (System Settings > Privacy & Security > Accessibility — grant
it to Terminal/iTerm, not to python itself).  Microphone access is prompted for
on first run.
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import random
import sys
import threading
import time
from datetime import datetime, timezone

import kkr_common as kc

# --------------------------------------------------------------------------- #
# Prompt text
# --------------------------------------------------------------------------- #

DEFAULT_PROSE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus", "prose_en.txt")
RANDOM_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def build_prose_lines(corpus_path: str, line_len: int, n_lines: int, rng: random.Random) -> list[str]:
    with open(corpus_path, encoding="utf-8") as fh:
        text = " ".join(fh.read().split())
    text = text.lower()
    words = [w for w in text.split(" ") if w]
    if not words:
        raise SystemExit(f"prose corpus is empty: {corpus_path}")
    start = rng.randrange(len(words))
    lines, cur = [], ""
    i = start
    while len(lines) < n_lines:
        w = words[i % len(words)]
        i += 1
        if cur and len(cur) + 1 + len(w) > line_len:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    return lines


def build_random_lines(line_len: int, n_lines: int, rng: random.Random, group: int = 5) -> list[str]:
    """Random characters, in small groups so they stay typable at a steady pace."""
    lines = []
    for _ in range(n_lines):
        chunks = []
        while sum(len(c) + 1 for c in chunks) < line_len:
            chunks.append("".join(rng.choice(RANDOM_ALPHABET) for _ in range(group)))
        lines.append(" ".join(chunks))
    return lines


# --------------------------------------------------------------------------- #
# Audio recorder
# --------------------------------------------------------------------------- #


class Recorder:
    """Continuous WAV writer keeping a monotonic-clock map of the frame index."""

    CHECKPOINT_PERIOD_S = 0.5

    def __init__(self, wav_path: str, samplerate: int, device=None, blocksize: int = 1024):
        import sounddevice as sd  # local import: capture-only dependency

        self._sd = sd
        self.wav_path = wav_path
        self.samplerate = samplerate
        self.device = device
        self.blocksize = blocksize

        self.frames_written = 0
        self._frames_seen = 0
        self.overflow_events = 0
        self.checkpoints: list[list[float]] = []
        self.audio_start_mono: float | None = None
        self._clock_offset: float | None = None  # monotonic - portaudio stream time
        self._input_latency = 0.0  # used only if the backend has no ADC timestamps
        self._last_cp_t = 0.0
        self._q: queue.Queue = queue.Queue()
        self._stream = None
        self._writer = None
        self._stop = threading.Event()

    # -- callback (audio thread: no disk I/O, no allocations beyond the copy) --
    def _callback(self, indata, frames, time_info, status):
        now = time.monotonic()
        if status:
            if getattr(status, "input_overflow", False):
                self.overflow_events += 1
            # force a re-anchor: samples may have been dropped just before this
            self._last_cp_t = 0.0

        adc = float(getattr(time_info, "inputBufferAdcTime", 0.0) or 0.0)
        cur = float(getattr(time_info, "currentTime", 0.0) or 0.0)
        if self._clock_offset is None:
            self._clock_offset = (now - cur) if (adc and cur) else None
        if self._clock_offset is not None and adc:
            t_buf = adc + self._clock_offset
        else:
            # Fallback for backends without usable ADC timestamps: this buffer was
            # captured over [now - latency - frames/sr, now - latency].  Less exact
            # than the ADC clock — check the sync beep report in preprocess.py.
            t_buf = now - self._input_latency - frames / self.samplerate

        if self.audio_start_mono is None:
            self.audio_start_mono = t_buf
        if t_buf - self._last_cp_t >= self.CHECKPOINT_PERIOD_S:
            self.checkpoints.append([float(self._frames_seen), t_buf])
            self._last_cp_t = t_buf

        self._frames_seen += frames
        self._q.put(indata.copy())

    def _write_loop(self):
        import soundfile as sf

        with sf.SoundFile(
            self.wav_path, mode="w", samplerate=self.samplerate, channels=1, subtype="PCM_16"
        ) as f:
            while not (self._stop.is_set() and self._q.empty()):
                try:
                    block = self._q.get(timeout=0.2)
                except queue.Empty:
                    continue
                f.write(block)
                self.frames_written += len(block)

    def start(self):
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._writer.start()
        self._stream = self._sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=self._callback,
        )
        try:
            self._input_latency = float(self._stream.latency)
        except Exception:
            self._input_latency = 0.0
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._stop.set()
        if self._writer is not None:
            self._writer.join(timeout=10)

    def device_name(self) -> str:
        try:
            info = self._sd.query_devices(self.device if self.device is not None else self._sd.default.device[0])
            return f"{info['name']} ({info['hostapi']})"
        except Exception:  # pragma: no cover - informational only
            return "unknown"


# --------------------------------------------------------------------------- #
# Keystroke listener
# --------------------------------------------------------------------------- #


class KeyLogger:
    """Logs keydown events (auto-repeat filtered) with monotonic timestamps."""

    def __init__(self, session_id: str, mode: str):
        self.session_id = session_id
        self.mode = mode
        self.events: list[tuple[float, str]] = []
        self._held: set[str] = set()
        self._lock = threading.Lock()
        self._listener = None

    @staticmethod
    def _label(key) -> str:
        from pynput import keyboard

        if isinstance(key, keyboard.KeyCode):
            if key.char is not None:
                return kc.canonical_label(key.char)
            return f"vk{key.vk}"
        name = str(key).replace("Key.", "")
        aliases = {
            "cmd_r": "cmd", "shift_r": "shift", "alt_r": "alt", "ctrl_r": "ctrl",
            "cmd_l": "cmd", "shift_l": "shift", "alt_l": "alt", "ctrl_l": "ctrl",
        }
        return aliases.get(name, name)

    def _on_press(self, key):
        t = time.monotonic()
        label = self._label(key)
        with self._lock:
            if label in self._held:  # OS auto-repeat: not a physical keydown
                return
            self._held.add(label)
            self.events.append((t, label))

    def _on_release(self, key):
        label = self._label(key)
        with self._lock:
            self._held.discard(label)

    def start(self):
        from pynput import keyboard

        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        self._listener.wait()

    def stop(self):
        if self._listener is not None:
            self._listener.stop()

    def count(self) -> int:
        with self._lock:
            return len(self.events)

    def snapshot(self) -> list[tuple[float, str]]:
        with self._lock:
            return list(self.events)


# --------------------------------------------------------------------------- #
# Sync beep
# --------------------------------------------------------------------------- #


def play_sync_beep(freq: float, duration: float, amplitude: float, samplerate: int) -> dict:
    """Play a short tone and return its monotonic timing (safety-net marker)."""
    import numpy as np
    import sounddevice as sd

    n = int(duration * samplerate)
    t = np.arange(n) / samplerate
    env = np.minimum(1.0, np.minimum(t, duration - t) / 0.004)  # 4 ms fades
    tone = (amplitude * env * np.sin(2 * np.pi * freq * t)).astype("float32")

    t0 = time.monotonic()
    sd.play(tone, samplerate, blocking=True)
    t1 = time.monotonic()
    try:
        out_latency = float(sd.query_devices(sd.default.device[1])["default_low_output_latency"])
    except Exception:
        out_latency = float("nan")
    return {
        "freq_hz": freq,
        "duration_s": duration,
        "t_call_mono": t0,
        "t_return_mono": t1,
        # `sd.play(blocking=True)` returns once playback is done: anchor the beep
        # at the end of the buffer, minus its duration.
        "t_start_mono_est": t1 - duration,
        "output_latency_s": out_latency,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("prose", "random"), required=True,
                   help="prose = natural English, random = random character strings")
    p.add_argument("--session-id", default=None, help="default: <mode>_<UTC timestamp>[_<tag>]")
    p.add_argument("--tag", default=None, help="free-form suffix, e.g. mic-position-b")
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--samplerate", type=int, default=kc.SAMPLE_RATE)
    p.add_argument("--device", default=None, help="input device index or name (see --list-devices)")
    p.add_argument("--blocksize", type=int, default=1024)
    p.add_argument("--target-keys", type=int, default=1300, help="stop suggesting lines past this count")
    p.add_argument("--line-length", type=int, default=58)
    p.add_argument("--prose-corpus", default=DEFAULT_PROSE)
    p.add_argument("--seed", type=int, default=None, help="prompt sampling seed (recorded in meta.json)")
    p.add_argument("--no-beep", action="store_true", help="skip the sync beep (no speakers)")
    p.add_argument("--beep-freq", type=float, default=1000.0)
    p.add_argument("--beep-duration", type=float, default=0.06)
    p.add_argument("--beep-amplitude", type=float, default=0.35)
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    try:
        import sounddevice  # noqa: F401
        import soundfile  # noqa: F401
        from pynput import keyboard  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"missing capture dependency: {exc}\ninstall with: pip install -r requirements.txt", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    session_id = args.session_id or "_".join(filter(None, [args.mode, stamp, args.tag]))
    out_dir = kc.ensure_dir(os.path.join(args.out_dir, session_id))
    if os.path.exists(os.path.join(out_dir, "keys.csv")):
        print(f"refusing to overwrite existing session at {out_dir}", file=sys.stderr)
        return 2

    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)
    n_lines = max(8, int(args.target_keys / max(1, args.line_length)) + 4)
    if args.mode == "prose":
        lines = build_prose_lines(args.prose_corpus, args.line_length, n_lines, rng)
    else:
        lines = build_random_lines(args.line_length, n_lines, rng)

    device = args.device
    if device is not None and str(device).isdigit():
        device = int(device)

    rec = Recorder(os.path.join(out_dir, "audio.wav"), args.samplerate, device, args.blocksize)
    logger = KeyLogger(session_id, args.mode)

    print(f"\n=== session {session_id} | mode={args.mode} | {args.samplerate} Hz mono ===")
    print(f"output: {out_dir}")
    rec.start()
    time.sleep(0.7)  # let the input stream settle before the sync marker
    print(f"input device: {rec.device_name()}")

    beep = None
    if not args.no_beep:
        print("sync beep...")
        beep = play_sync_beep(args.beep_freq, args.beep_duration, args.beep_amplitude, args.samplerate)

    logger.start()
    t_probe = time.monotonic()

    print(
        "\nType each line, then press Enter. Ctrl-C (or an empty line) ends the session.\n"
        "Type at your natural pace; do not fix typos — every physical keydown is\n"
        "labelled, so mistakes are usable data.\n"
    )

    aborted = False
    try:
        for i, line in enumerate(lines):
            if logger.count() >= args.target_keys:
                print(f"\ntarget of {args.target_keys} keystrokes reached.")
                break
            print(f"[{i + 1}/{len(lines)}]  {line}")
            try:
                typed = input("> ")
            except EOFError:
                break
            if typed == "" and i > 0:
                break
            if i == 0 and logger.count() == 0 and time.monotonic() - t_probe > 3:
                print(
                    "\n!! no keystrokes captured — macOS Accessibility permission is\n"
                    "   missing. System Settings > Privacy & Security > Accessibility,\n"
                    "   enable your terminal app, then restart this script.\n",
                    file=sys.stderr,
                )
            print(f"    ({logger.count()} keystrokes logged)")
    except KeyboardInterrupt:
        aborted = True
        print("\ninterrupted — finalising session.")

    logger.stop()
    time.sleep(0.4)  # trailing audio for the last keystroke's window
    rec.stop()

    events = logger.snapshot()
    csv_path = os.path.join(out_dir, "keys.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kc.CSV_FIELDS))
        w.writeheader()
        for t, key in events:
            w.writerow({"timestamp": f"{t:.6f}", "key": key, "session_id": session_id, "mode": args.mode})

    meta = {
        "session_id": session_id,
        "mode": args.mode,
        "tag": args.tag,
        "capture_version": kc.CAPTURE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "samplerate": args.samplerate,
        "channels": 1,
        "blocksize": args.blocksize,
        "device": rec.device_name(),
        "audio_start_mono": rec.audio_start_mono,
        "clock_checkpoints": rec.checkpoints,
        "clock_offset_s": 0.0,
        "frames_written": rec.frames_written,
        "duration_s": rec.frames_written / args.samplerate,
        "overflow_events": rec.overflow_events,
        "sync_beep": beep,
        "n_key_events": len(events),
        "prompt_lines": lines,
        "prompt_seed": seed,
        "prose_corpus": os.path.abspath(args.prose_corpus) if args.mode == "prose" else None,
        "aborted": aborted,
        "python": sys.version.split()[0],
    }
    kc.write_json(os.path.join(out_dir, "meta.json"), meta)

    dur = meta["duration_s"]
    print(f"\naudio     : {dur:.1f} s ({rec.frames_written} frames), overflows={rec.overflow_events}")
    print(f"keystrokes: {len(events)}")
    if events and dur > 0:
        print(f"rate      : {len(events) / dur:.1f} keys/s")
    if rec.overflow_events:
        print("note: input overflows occurred; clock checkpoints re-anchor the mapping.")
    if not events:
        print("WARNING: zero keystrokes logged — check the Accessibility permission.", file=sys.stderr)
    print(f"session written to {out_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
