#!/usr/bin/env python3
"""live.py — real-time demo: type, and watch what the model thinks you typed.

Loads a trained checkpoint, listens to the microphone and the keyboard at the
same time, and for every keydown cuts the same ~250 ms window the pipeline uses,
runs the CNN, and prints the predicted key next to the one you actually pressed.
It keeps a running accuracy and a reconstructed "what the model heard" transcript
beside the real one.

    python live.py --checkpoint models/keycnn.pt

This is the honest acoustic demo: windows are placed with the system keydown
timestamps (blind segmentation is out of scope, as everywhere in this repo), and
no language model is involved — you are watching the acoustic classifier alone
decide, character by character. Because it runs on your live keystrokes, it is
also a fresh, never-trained-on "session": if the model only memorised one
recording it will do poorly here.

macOS: needs the Accessibility permission for your terminal app, and microphone
access. Keep the same keyboard and a similar mic position to the training
sessions. Ctrl-C stops and prints the final transcript.
"""

from __future__ import annotations

import argparse
import collections
import sys
import threading
import time

import numpy as np

import kkr_common as kc
from capture import KeyLogger


# --------------------------------------------------------------------------- #
# Rolling-buffer live audio engine (in-memory sibling of capture.Recorder)
# --------------------------------------------------------------------------- #


class LiveEngine:
    """Keeps the last few seconds of audio in a ring buffer, with a monotonic
    frame<->time map identical in spirit to capture.Recorder's checkpoints."""

    CHECKPOINT_PERIOD_S = 0.5

    def __init__(self, samplerate: int, buffer_s: float, device=None, blocksize: int = 1024):
        self.samplerate = samplerate
        self.device = device
        self.blocksize = blocksize
        self.capacity = int(buffer_s * samplerate)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self.frames_seen = 0
        self.audio_start_mono = None
        self.overflow_events = 0
        self._checkpoints = collections.deque(maxlen=128)
        self._clock_offset = None
        self._input_latency = 0.0
        self._last_cp_t = 0.0
        self._lock = threading.Lock()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        now = time.monotonic()
        if status:
            if getattr(status, "input_overflow", False):
                self.overflow_events += 1
            self._last_cp_t = 0.0
        adc = float(getattr(time_info, "inputBufferAdcTime", 0.0) or 0.0)
        cur = float(getattr(time_info, "currentTime", 0.0) or 0.0)
        if self._clock_offset is None:
            self._clock_offset = (now - cur) if (adc and cur) else None
        if self._clock_offset is not None and adc:
            t_buf = adc + self._clock_offset
        else:
            t_buf = now - self._input_latency - frames / self.samplerate

        x = indata[:, 0] if indata.ndim > 1 else indata
        with self._lock:
            if self.audio_start_mono is None:
                self.audio_start_mono = t_buf
            pos = (self.frames_seen + np.arange(len(x))) % self.capacity
            self._buf[pos] = x
            if t_buf - self._last_cp_t >= self.CHECKPOINT_PERIOD_S:
                self._checkpoints.append([float(self.frames_seen), t_buf])
                self._last_cp_t = t_buf
            self.frames_seen += len(x)

    def meta(self) -> dict:
        with self._lock:
            return {
                "samplerate": self.samplerate,
                "clock_checkpoints": list(self._checkpoints),
                "audio_start_mono": self.audio_start_mono,
                "clock_offset_s": 0.0,
            }

    def read(self, start_frame: int, length: int):
        """Return `length` samples starting at absolute frame `start_frame`,
        or None if they have already scrolled out of the buffer / not arrived."""
        with self._lock:
            end = self.frames_seen
            if start_frame < 0 or start_frame < end - self.capacity or start_frame + length > end:
                return None
            idx = (start_frame + np.arange(length)) % self.capacity
            return self._buf[idx].copy()

    def has_through(self, frame: int) -> bool:
        with self._lock:
            return self.frames_seen >= frame

    def start(self):
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self.samplerate, channels=1, dtype="float32",
            blocksize=self.blocksize, device=self.device, callback=self._callback,
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


# --------------------------------------------------------------------------- #
# Feature + model wrapper (mirrors preprocess.py / train.py exactly)
# --------------------------------------------------------------------------- #


class Predictor:
    def __init__(self, checkpoint: str, device: str | None = None):
        import torch

        from train import KeyCNN

        self.torch = torch
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.vocab = list(ckpt["vocab"])
        self.cfg = ckpt["feature_config"]
        self.mu = np.asarray(ckpt["norm"]["mu"], dtype=np.float32)  # (1, n_mels, 1)
        self.sd = np.asarray(ckpt["norm"]["sd"], dtype=np.float32)
        self.device = torch.device(device) if device else torch.device("cpu")
        self.model = KeyCNN(len(self.vocab), tuple(ckpt["model"]["widths"]), ckpt["model"]["dropout"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.pre_ms = float(self.cfg.get("pre_ms", 50.0))
        self.post_ms = float(self.cfg.get("post_ms", 200.0))
        self.normalize = bool(self.cfg.get("normalize", True))
        self.sr = int(self.cfg.get("sr", kc.SAMPLE_RATE))
        self.splits = ckpt.get("splits", {})
        self.epoch = ckpt.get("epoch")

    def window_frames(self):
        pre = int(round(self.pre_ms * 1e-3 * self.sr))
        post = int(round(self.post_ms * 1e-3 * self.sr))
        return pre, post

    def features(self, win: np.ndarray) -> np.ndarray:
        from preprocess import melspec

        S = melspec(win, self.sr, self.cfg)
        if self.normalize:
            S = (S - S.mean()) / (S.std() + 1e-5)
        return ((S - self.mu[0]) / self.sd[0]).astype(np.float32)

    def predict(self, win: np.ndarray, k: int = 5):
        S = self.features(win)
        with self.torch.no_grad():
            logits = self.model(self.torch.from_numpy(S)[None, None].to(self.device))
            logits = logits[0].cpu().numpy()
        k = min(k, len(self.vocab))
        top = np.argsort(-logits)[:k]
        return [self.vocab[i] for i in top], logits


# --------------------------------------------------------------------------- #
# Pretty-printing
# --------------------------------------------------------------------------- #


def glyph(key: str) -> str:
    if key == "space":
        return "␣"
    if key == "enter":
        return "⏎"
    if key == "backspace":
        return "⌫"
    if len(key) == 1:
        return key
    return f"⟨{key}⟩"


# --------------------------------------------------------------------------- #
# Real-time loop
# --------------------------------------------------------------------------- #


def run_live(predictor: Predictor, engine: LiveEngine, args) -> int:
    logger = KeyLogger("live", "live")
    pre, post = predictor.window_frames()
    margin = int(round(args.margin_ms * 1e-3 * predictor.sr))
    vocabset = set(predictor.vocab)

    print(f"\ncheckpoint : {args.checkpoint} (epoch {predictor.epoch})")
    print(f"trained on : {predictor.splits.get('train')}")
    print(f"vocabulary : {len(predictor.vocab)} keys  (chance {kc.fmt_pct(1 / len(predictor.vocab))})")
    print(f"window     : -{predictor.pre_ms:.0f}/+{predictor.post_ms:.0f} ms around each keydown\n")

    engine.start()
    time.sleep(0.6)
    print(f"input device settled; mic latency ~{engine._input_latency * 1e3:.0f} ms")
    logger.start()
    print(
        "\nStart typing — the model guesses each key from its sound.\n"
        "  you=what you pressed   model=top-1 guess   ✓/✗=match   [..]=top-5\n"
        "Ctrl-C to stop and see the full transcript.\n"
    )

    typed, heard = [], []
    n = c1 = c5 = 0
    processed = 0
    warned = False
    t_probe = time.monotonic()

    try:
        while True:
            downs = logger.keydown_snapshot()
            now = time.monotonic()
            while processed < len(downs):
                t, key = downs[processed]
                center = int(round(kc.mono_to_frame(t, engine.meta())))
                if not engine.has_through(center + post + margin):
                    if now - t < (predictor.post_ms + args.margin_ms) * 1e-3 + 0.4:
                        break  # audio for this keydown hasn't fully arrived yet
                    processed += 1  # too old, give up on it
                    continue
                processed += 1
                win = engine.read(center - pre, pre + post)
                if win is None:
                    continue
                preds, _ = predictor.predict(win, k=5)
                pred = preds[0]
                typed.append(key)
                heard.append(pred)
                scored = key in vocabset
                if scored:
                    n += 1
                    c1 += pred == key
                    c5 += key in preds
                mark = "·" if not scored else ("✓" if pred == key else "✗")
                acc = f"  run top1 {kc.fmt_pct(c1 / n)} top5 {kc.fmt_pct(c5 / n)}" if n else ""
                print(
                    f"you {glyph(key):<4} model {glyph(pred):<4} {mark}  "
                    f"[{' '.join(glyph(p) for p in preds)}]{acc}"
                )
            if not warned and logger.count() == 0 and now - t_probe > 4:
                print(
                    "\n!! no keystrokes yet — grant Accessibility to your terminal app\n"
                    "   (System Settings > Privacy & Security > Accessibility) and restart.\n",
                    file=sys.stderr,
                )
                warned = True
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n\nstopping.")

    logger.stop()
    engine.stop()

    print("\n" + "=" * 72)
    print("TRANSCRIPT")
    print("=" * 72)
    print("you typed  : " + "".join(glyph(k) for k in typed))
    print("model heard: " + "".join(glyph(k) for k in heard))
    if n:
        print(f"\nscored on {n} keys in the vocabulary: "
              f"top-1 {kc.fmt_pct(c1 / n)}, top-5 {kc.fmt_pct(c5 / n)} "
              f"(chance {kc.fmt_pct(1 / len(predictor.vocab))})")
        print("This is the acoustic model alone, on keystrokes it was never trained on.")
    else:
        print("\nno keystroke fell inside the model vocabulary — try letters/space, "
              "or retrain with a wider --keys set.")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="models/keycnn.pt")
    p.add_argument("--device", default=None, help="cpu | mps | cuda (default: cpu)")
    p.add_argument("--audio-device", default=None, help="input device index or name")
    p.add_argument("--blocksize", type=int, default=512)
    p.add_argument("--buffer-s", type=float, default=6.0, help="seconds of audio kept in memory")
    p.add_argument("--margin-ms", type=float, default=60.0, help="extra audio waited for after a keydown")
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
        import torch  # noqa: F401
        from pynput import keyboard  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"missing dependency: {exc}\ninstall with: pip install -r requirements.txt", file=sys.stderr)
        return 2

    predictor = Predictor(args.checkpoint, args.device)
    dev = args.audio_device
    if dev is not None and str(dev).isdigit():
        dev = int(dev)
    engine = LiveEngine(predictor.sr, args.buffer_s, dev, args.blocksize)
    return run_live(predictor, engine, args)


if __name__ == "__main__":
    raise SystemExit(main())
