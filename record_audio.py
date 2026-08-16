#!/usr/bin/env python3
"""Capture audio longue durée (une nuit entière) sur macOS.

Pattern repris du Recorder du projet clavier (capture.py) :
  - le callback audio ne touche JAMAIS le disque : il horodate (horloge
    monotone, offset ADC PortAudio, re-ancrage après overflow) et pousse le
    bloc dans une queue ;
  - un thread écrivain séparé écrit sur disque ;
  - des checkpoints [frame, t_mono] ~2×/s permettent de resituer chaque
    échantillon dans le temps même si l'horloge audio dérive.

Adaptations « nuit entière » :
  - 16 kHz mono FLAC (≈ 2× plus léger que WAV, et c'est le format d'entrée
    des modèles AudioSet) ;
  - rotation de fichier toutes les ~10 min (chunk_XXX.flac) : pas de fichier
    géant, et un crash ne perd au pire que le chunk en cours (meta.json est
    réécrit à chaque rotation) ;
  - `caffeinate` empêche la mise en veille du Mac pendant la capture ;
  - vérification de l'espace disque au lancement.

Sortie : data/raw/<session_id>/chunk_XXX.flac + meta.json.

Vie privée : ce script capte TOUT, y compris les conversations. La suite du
pipeline (detect_events.py --delete-raw, embed.py --privacy/--drop-speech)
sert précisément à ne pas conserver l'audio brut. Voir README.
"""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import sed_common as sc


class ChunkedRecorder:
    """Enregistreur continu FLAC chunké avec carte horloge-monotone -> frame."""

    CHECKPOINT_PERIOD_S = 0.5

    def __init__(self, session_dir: str, samplerate: int, chunk_frames: int,
                 device=None, blocksize: int = 1024):
        import sounddevice as sd  # import local : dépendance de capture seulement

        self._sd = sd
        self.session_dir = session_dir
        self.samplerate = samplerate
        self.chunk_frames = chunk_frames
        self.device = device
        self.blocksize = blocksize

        self.frames_written = 0
        self._frames_seen = 0
        self.overflow_events = 0
        self.checkpoints: list[list[float]] = []
        self.chunks: list[dict] = []
        self.audio_start_mono: float | None = None
        self._clock_offset: float | None = None  # monotonic - horloge PortAudio
        self._input_latency = 0.0
        self._last_cp_t = 0.0

        self._q: queue.Queue = queue.Queue()
        self._stream = None
        self._writer: threading.Thread | None = None
        self._stop = threading.Event()
        self.on_chunk_closed = None  # callback(meta_partiel) après chaque rotation

    # -- thread audio : pas d'I/O disque, pas d'allocation au-delà de la copie
    def _callback(self, indata, frames, time_info, status):
        now = time.monotonic()
        if status:
            if getattr(status, "input_overflow", False):
                self.overflow_events += 1
            self._last_cp_t = 0.0  # ré-ancrage forcé : des échantillons ont pu sauter
        adc = float(getattr(time_info, "inputBufferAdcTime", 0.0) or 0.0)
        cur = float(getattr(time_info, "currentTime", 0.0) or 0.0)
        if self._clock_offset is None:
            self._clock_offset = (now - cur) if (adc and cur) else None
        if self._clock_offset is not None and adc:
            t_buf = adc + self._clock_offset
        else:  # backend sans timestamps ADC exploitables
            t_buf = now - self._input_latency - frames / self.samplerate
        if self.audio_start_mono is None:
            self.audio_start_mono = t_buf
        if t_buf - self._last_cp_t >= self.CHECKPOINT_PERIOD_S:
            self.checkpoints.append([float(self._frames_seen), t_buf])
            self._last_cp_t = t_buf
        self._frames_seen += frames
        self._q.put(indata.copy())

    # -- thread écrivain : rotation des chunks FLAC
    def _write_loop(self):
        import soundfile as sf

        ci = len(self.chunks)
        f = None
        in_chunk = 0
        try:
            while not (self._stop.is_set() and self._q.empty()):
                try:
                    block = self._q.get(timeout=0.2)
                except queue.Empty:
                    continue
                pos = 0
                while pos < len(block):
                    if f is None:
                        fname = f"chunk_{ci:03d}.flac"
                        f = sf.SoundFile(
                            os.path.join(self.session_dir, fname), mode="w",
                            samplerate=self.samplerate, channels=1,
                            format="FLAC", subtype="PCM_16",
                        )
                        self.chunks.append({
                            "file": fname,
                            "start_frame": self.frames_written,
                            "n_frames": 0,
                        })
                        in_chunk = 0
                    take = min(len(block) - pos, self.chunk_frames - in_chunk)
                    f.write(block[pos : pos + take])
                    self.frames_written += take
                    in_chunk += take
                    self.chunks[-1]["n_frames"] = in_chunk
                    pos += take
                    if in_chunk >= self.chunk_frames:
                        f.close()
                        f = None
                        ci += 1
                        if self.on_chunk_closed is not None:
                            self.on_chunk_closed()
        finally:
            if f is not None:
                f.close()

    def start(self):
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._writer.start()
        self._stream = self._sd.InputStream(
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
        self._stop.set()
        if self._writer is not None:
            self._writer.join(timeout=10)

    def device_name(self) -> str:
        try:
            info = self._sd.query_devices(self.device, "input")
            hostapi = self._sd.query_hostapis(info["hostapi"])["name"]
            return f"{info['name']} ({hostapi})"
        except Exception:
            return str(self.device)


def build_meta(session_id: str, kind: str, tag: str, rec: ChunkedRecorder,
               t_utc_start_epoch: float, requested_duration_s: float | None,
               stopped_by: str | None = None) -> dict:
    return {
        "session_id": session_id,
        "kind": kind,
        "tag": tag,
        "capture_version": sc.CAPTURE_VERSION,
        "created_utc": datetime.fromtimestamp(
            t_utc_start_epoch, tz=timezone.utc
        ).isoformat(),
        "t_utc_start_epoch": t_utc_start_epoch,
        "samplerate": rec.samplerate,
        "channels": 1,
        "blocksize": rec.blocksize,
        "device": rec.device_name(),
        "audio_start_mono": rec.audio_start_mono,
        "clock_checkpoints": rec.checkpoints,
        "clock_offset_s": 0.0,
        "chunks": rec.chunks,
        "frames_written": rec.frames_written,
        "duration_s": rec.frames_written / rec.samplerate,
        "overflow_events": rec.overflow_events,
        "requested_duration_s": requested_duration_s,
        "stopped_by": stopped_by,
        "python": sys.version.split()[0],
    }


def check_disk(session_dir: str, duration_s: float, samplerate: int) -> None:
    # FLAC PCM_16 mono ≈ 50-60 % du PCM brut ; on prévoit 2× l'estimation.
    est = duration_s * samplerate * 2 * 0.6
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(session_dir))).free
    if free < 2 * est:
        raise SystemExit(
            f"FATAL: espace disque insuffisant ({free / 1e9:.1f} Go libres, "
            f"il en faut ~{2 * est / 1e9:.1f} pour {duration_s / 3600:.1f} h). "
            "Libère de la place ou réduis --duration."
        )


def start_caffeinate() -> subprocess.Popen | None:
    """Empêche la veille du Mac pendant la capture (-d écran autorisé à dormir non,
    -i idle, -m disque, -s secteur ; -w s'arrête avec nous)."""
    path = shutil.which("caffeinate")
    if path is None:
        print("note : caffeinate introuvable (pas macOS ?) — veille non bloquée")
        return None
    return subprocess.Popen([path, "-dims", "-w", str(os.getpid())])


def parse_duration(text: str) -> float:
    """'8h', '90m', '3600' -> secondes."""
    text = text.strip().lower()
    if text.endswith("h"):
        return float(text[:-1]) * 3600
    if text.endswith("m"):
        return float(text[:-1]) * 60
    if text.endswith("s"):
        return float(text[:-1])
    return float(text)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kind", choices=("night", "day"), default="night")
    p.add_argument("--tag", default="", help="ex. position du micro : 'chevet'")
    p.add_argument("--session-id", default=None)
    p.add_argument("--out-dir", default=sc.RAW_DIR)
    p.add_argument("--duration", default=None,
                   help="'8h', '45m', '600' (s) ; défaut 8h la nuit, illimité le jour")
    p.add_argument("--chunk-min", type=float, default=10.0)
    p.add_argument("--samplerate", type=int, default=sc.SAMPLE_RATE)
    p.add_argument("--device", default=None)
    p.add_argument("--blocksize", type=int, default=1024)
    p.add_argument("--status-every", type=float, default=30.0)
    p.add_argument("--no-caffeinate", action="store_true")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    try:
        import sounddevice as sd
        import soundfile  # noqa: F401
    except ImportError as e:
        print(f"dépendance de capture manquante : {e}\n"
              "  pip install sounddevice soundfile")
        return 2

    if args.list_devices:
        print(sd.query_devices())
        return 0

    device = args.device
    if device is not None and str(device).isdigit():
        device = int(device)

    duration_s = None
    if args.duration:
        duration_s = parse_duration(args.duration)
    elif args.kind == "night":
        duration_s = 8 * 3600.0

    session_id = args.session_id or sc.make_session_id(args.kind, args.tag)
    session_dir = os.path.join(args.out_dir, session_id)
    if os.path.exists(os.path.join(session_dir, "meta.json")):
        print(f"refus d'écraser la session existante {session_dir}")
        return 2
    sc.ensure_dir(session_dir)
    check_disk(session_dir, duration_s or 8 * 3600.0, args.samplerate)

    rec = ChunkedRecorder(
        session_dir, args.samplerate,
        chunk_frames=int(args.chunk_min * 60 * args.samplerate),
        device=device, blocksize=args.blocksize,
    )
    t_utc_start = time.time()

    def flush_meta(stopped_by=None):
        sc.write_json(os.path.join(session_dir, "meta.json"),
                      build_meta(session_id, args.kind, args.tag, rec,
                                 t_utc_start, duration_s, stopped_by))

    rec.on_chunk_closed = flush_meta  # checkpoint de reprise à chaque rotation

    caff = None if args.no_caffeinate else start_caffeinate()
    stop_requested = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())

    print(f"session {session_id} -> {session_dir}")
    print(f"  device : {rec.device_name()}  |  {args.samplerate} Hz mono FLAC, "
          f"chunks de {args.chunk_min:g} min")
    if duration_s:
        print(f"  durée prévue : {duration_s / 3600:.1f} h — Ctrl-C pour arrêter avant")
    else:
        print("  durée illimitée — Ctrl-C pour arrêter")

    rec.start()
    stopped_by = "user"
    t0 = time.monotonic()
    try:
        while not stop_requested.is_set():
            time.sleep(min(args.status_every, 5.0))
            elapsed = time.monotonic() - t0
            if int(elapsed) % max(1, int(args.status_every)) < 5:
                print(f"\r  {elapsed / 60:6.1f} min  "
                      f"{rec.frames_written / args.samplerate / 60:6.1f} min écrites  "
                      f"chunks={len(rec.chunks)}  overflows={rec.overflow_events}",
                      end="", flush=True)
            if duration_s is not None and elapsed >= duration_s:
                stopped_by = "duration"
                break
    except KeyboardInterrupt:
        stopped_by = "user"
    finally:
        print("\narrêt de la capture…")
        time.sleep(0.4)  # queue de traîne
        rec.stop()
        flush_meta(stopped_by)
        if caff is not None:
            caff.terminate()

    print(f"terminé ({stopped_by}) : {rec.frames_written / args.samplerate / 60:.1f} min "
          f"dans {len(rec.chunks)} chunks, {rec.overflow_events} overflows")
    print("prochaine étape : python detect_events.py --session " + session_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
