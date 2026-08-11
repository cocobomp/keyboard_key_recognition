#!/usr/bin/env python3
"""record.py — free-form capture: the microphone + EVERY keyboard event.

Unlike capture.py there is no prompt: you just type whatever you like (write an
email, some code, whatever) and everything is recorded until you press Ctrl-C.
Both keydown and keyup are logged, on the same monotonic clock as the audio, so
the recording can be fed straight into the rest of the pipeline.

    data/raw/<session_id>/audio.wav    48 kHz mono WAV
    data/raw/<session_id>/keys.csv     keydown events  (timestamp,key,session_id,mode)
    data/raw/<session_id>/events.csv   every event     (timestamp,event,key,session_id,mode)
    data/raw/<session_id>/meta.json    clock map, sync beep, config

Privacy: this is a global keylogger. It records EVERYTHING you type while it
runs — including passwords, messages and anything in other windows — into plain
files on this machine. It is meant for capturing your own keystrokes for this
POC, on your own computer, locally. Don't run it while typing secrets, and treat
data/ as sensitive (it is git-ignored). Ctrl-C stops it.

macOS: needs the Accessibility permission for your terminal app (System Settings
> Privacy & Security > Accessibility). Microphone access is prompted on first run.

Examples
--------
    python record.py                       # free typing, mode="free", until Ctrl-C
    python record.py --duration 600         # stop automatically after 10 minutes
    python record.py --mode prose --tag micB  # label a free session as prose/random
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import kkr_common as kc
# Reuse the tested audio recorder, key logger, sync beep and file writers.
from capture import KeyLogger, Recorder, base_meta, play_sync_beep, write_session_files


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="free", choices=("free", "prose", "random"),
                   help="label stored for this session (default: free)")
    p.add_argument("--session-id", default=None, help="default: <mode>_<UTC timestamp>[_<tag>]")
    p.add_argument("--tag", default=None, help="free-form suffix, e.g. mic-position-b")
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--duration", type=float, default=None, help="auto-stop after N seconds (default: until Ctrl-C)")
    p.add_argument("--samplerate", type=int, default=kc.SAMPLE_RATE)
    p.add_argument("--device", default=None, help="input device index or name (see --list-devices)")
    p.add_argument("--blocksize", type=int, default=1024)
    p.add_argument("--status-every", type=float, default=2.0, help="seconds between the live status line")
    p.add_argument("--no-beep", action="store_true")
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

    device = args.device
    if device is not None and str(device).isdigit():
        device = int(device)

    rec = Recorder(os.path.join(out_dir, "audio.wav"), args.samplerate, device, args.blocksize)
    logger = KeyLogger(session_id, args.mode)

    print(f"\n=== free recording {session_id} | mode={args.mode} | {args.samplerate} Hz mono ===")
    print(f"output: {out_dir}")
    print("PRIVACY: every keystroke is logged to disk while this runs. Don't type secrets.")
    rec.start()
    time.sleep(0.7)  # let the input stream settle before the sync marker
    print(f"input device: {rec.device_name()}")

    beep = None
    if not args.no_beep:
        print("sync beep...")
        beep = play_sync_beep(args.beep_freq, args.beep_duration, args.beep_amplitude, args.samplerate)

    logger.start()
    t_start = time.monotonic()
    warned = False
    if args.duration:
        print(f"\nrecording — type anything; stops automatically after {args.duration:.0f}s, or Ctrl-C.\n")
    else:
        print("\nrecording — type anything; press Ctrl-C to stop.\n")

    aborted = False
    try:
        while True:
            time.sleep(args.status_every)
            el = time.monotonic() - t_start
            n = logger.count()
            print(f"\r  {el:6.1f}s  {n:6d} keydowns  ({n / el:5.1f}/s)   ", end="", flush=True)
            if not warned and el > 4 and n == 0:
                print(
                    "\n!! no keystrokes captured yet — on macOS grant Accessibility to your\n"
                    "   terminal app (System Settings > Privacy & Security > Accessibility),\n"
                    "   then restart this script.\n",
                    file=sys.stderr,
                )
                warned = True
            if args.duration and el >= args.duration:
                print("\nduration reached — finalising session.")
                break
    except KeyboardInterrupt:
        aborted = True
        print("\nstopping — finalising session.")

    logger.stop()
    time.sleep(0.4)  # trailing audio for the last keystroke's window
    rec.stop()

    downs = logger.keydown_snapshot()
    events = logger.events_snapshot()
    write_session_files(out_dir, session_id, args.mode, downs, events)

    meta = base_meta(session_id, args.mode, args.tag, rec, beep)
    meta.update(
        {
            "n_key_events": len(downs),
            "n_raw_events": len(events),
            "prompt_lines": [],          # free-form: no prompt
            "free_capture": True,
            "requested_duration_s": args.duration,
            "stopped_by": "duration" if (args.duration and not aborted) else "user",
        }
    )
    kc.write_json(os.path.join(out_dir, "meta.json"), meta)

    dur = meta["duration_s"]
    print(f"\naudio     : {dur:.1f} s ({rec.frames_written} frames), overflows={rec.overflow_events}")
    print(f"keystrokes: {len(downs)} keydowns ({len(events)} raw events)")
    if downs and dur > 0:
        print(f"rate      : {len(downs) / dur:.1f} keys/s")
    if rec.overflow_events:
        print("note: input overflows occurred; clock checkpoints re-anchor the mapping.")
    if not downs:
        print("WARNING: zero keystrokes logged — check the Accessibility permission.", file=sys.stderr)
    print(f"session written to {out_dir}")
    print("next: python preprocess.py  (windows are cut from keys.csv)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
