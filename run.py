#!/usr/bin/env python3
"""run.py — one program to record, train, evaluate and demo, in order.

This is the guided front-end for the whole POC. It drives the other scripts
(capture.py, preprocess.py, train.py, eval.py, baseline_knn.py, live.py) so you
do not have to remember the individual commands or build the per-session split by
hand.

    python run.py                 # interactive menu
    python run.py all             # record sessions -> train -> evaluate -> live demo
    python run.py collect         # record several labelled sessions
    python run.py pipeline        # preprocess -> train -> eval -> baseline on existing data
    python run.py live            # real-time demo with the trained model

The per-session split is built for you and always kept whole: entire recording
sessions go to train / validation / test, never individual keystrokes. That rule
is what makes the numbers mean recognition rather than memorisation, so it is not
optional here.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import kkr_common as kc

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_FILE = "splits.json"
CKPT = "models/keycnn.pt"


def sh(script: str, *cargs, check: bool = True) -> int:
    cmd = [PY, os.path.join(HERE, script), *[str(a) for a in cargs]]
    print(f"\n$ {' '.join(cmd)}\n")
    rc = subprocess.run(cmd).returncode
    if check and rc != 0:
        raise SystemExit(f"step failed ({script}, exit {rc}).")
    return rc


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    try:
        ans = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        ans = ""
    return ans or (default or "")


# --------------------------------------------------------------------------- #
# Collect
# --------------------------------------------------------------------------- #


def collect(args) -> None:
    n = args.sessions
    print(
        f"\nAbout to record {n} sessions (alternating prose / random), "
        f"~{args.keys} keystrokes each.\n"
        "One session = one recording. Between sessions, MOVE THE MIC a little\n"
        "(a few cm, a slightly different angle) and keep the same keyboard and\n"
        "posture. Don't fix typos — every keydown is a labelled sample.\n"
    )
    for i in range(n):
        mode = "prose" if i % 2 == 0 else "random"
        mic = chr(ord("A") + i // 2)  # new mic position every prose/random pair
        print("\n" + "-" * 60)
        print(f"session {i + 1}/{n}: mode={mode}, mic position {mic}")
        if i % 2 == 0 and i > 0:
            print(">> move the microphone slightly now (new position).")
        ask("press Enter when you are ready to record this session", "")
        sh("capture.py", "--mode", mode, "--tag", f"mic{mic}", "--target-keys", args.keys)
    print("\nall sessions recorded.")
    summarise_sessions(args.raw_dir)


def summarise_sessions(raw_dir: str) -> list:
    try:
        sessions = kc.list_sessions(raw_dir)
    except FileNotFoundError:
        print(f"no sessions found under {raw_dir}.")
        return []
    print(f"\nsessions in {raw_dir}:")
    for s in sessions:
        n = s.meta.get("n_key_events", "?")
        print(f"  {s.session_id:<40} mode={s.mode:<7} keys={n}")
    total = sum(int(s.meta.get("n_key_events", 0) or 0) for s in sessions)
    print(f"  total keystrokes: {total}"
          + ("" if total >= 5000 else f"  (protocol targets >= 5000; have {total})"))
    return sessions


# --------------------------------------------------------------------------- #
# Split building — whole sessions only
# --------------------------------------------------------------------------- #


def build_split(raw_dir: str, out_file: str = SPLIT_FILE) -> dict:
    sessions = kc.list_sessions(raw_dir)
    if len(sessions) < 2:
        raise SystemExit(
            f"need at least 2 recording sessions to make a per-session split (have {len(sessions)}). "
            "Record more with `run.py collect`."
        )
    by = {"prose": [], "random": [], "other": []}
    for s in sessions:
        by.get(s.mode, by["other"]).append(s.session_id)
    for v in by.values():
        v.sort()

    # Hold out one prose and one random session for test (the honest comparison
    # needs both regimes); fall back to the last session if a regime is missing.
    test = [lst[-1] for lst in (by["prose"], by["random"]) if lst]
    if not test:
        test = [sessions[-1].session_id]
    used = set(test)

    # Validation: a remaining session, prose first (so the LM weight can be tuned).
    val = []
    for pool in (by["prose"], by["random"], by["other"]):
        cand = [s for s in pool if s not in used]
        if cand:
            val = [cand[0]]
            used.add(cand[0])
            break

    train = [s.session_id for s in sessions if s.session_id not in used]
    if not train:
        raise SystemExit(
            "not enough sessions: every session ended up in test/val. "
            "Record more sessions (aim for >= 4, ideally 6-8)."
        )
    if not val:
        val = [train.pop()]  # last resort: borrow one train session for validation

    split = {"train": train, "val": val, "test": test}
    kc.check_disjoint(**split)
    kc.write_json(out_file, split)
    print("\nper-session split (whole sessions, never mixed):")
    for k in ("train", "val", "test"):
        print(f"  {k:<6}: {split[k]}")
    if not (by["prose"] and by["random"]):
        print("  ! only one prompt regime present — the prose-vs-random headline "
              "won't be computable. Record at least one prose and one random session.")
    print(f"written to {out_file}")
    return split


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def pipeline(args) -> None:
    summarise_sessions(args.raw_dir)
    build_split(args.raw_dir, args.split_file)
    sh("preprocess.py", "--raw-dir", args.raw_dir, "--out-dir", args.processed_dir, "--keys", args.keys_set)
    sh("train.py", "--processed-dir", args.processed_dir, "--split-file", args.split_file,
       "--out", CKPT, "--epochs", args.epochs)
    eval_args = ["eval.py", "--checkpoint", CKPT, "--processed-dir", args.processed_dir,
                 "--raw-dir", args.raw_dir, "--results-dir", args.results_dir]
    if args.lm_corpus and os.path.isfile(args.lm_corpus):
        eval_args += ["--lm-corpus", args.lm_corpus]
    sh(*eval_args)
    sh("baseline_knn.py", "--processed-dir", args.processed_dir, "--split-file", args.split_file,
       "--results-dir", args.results_dir, check=False)
    print("\n" + "=" * 60)
    print("done. Key files:")
    print(f"  {os.path.join(args.results_dir, 'report.md')}         <- read this")
    print(f"  {os.path.join(args.results_dir, 'confusion_random.png')}")
    print(f"  {CKPT}")
    print("=" * 60)


def live(args) -> None:
    if not os.path.isfile(CKPT):
        raise SystemExit(f"no trained model at {CKPT} — run `run.py pipeline` (or `all`) first.")
    live_args = ["live.py", "--checkpoint", CKPT]
    if args.device:
        live_args += ["--device", args.device]
    sh(*live_args, check=False)


# --------------------------------------------------------------------------- #
# Menu / CLI
# --------------------------------------------------------------------------- #


def menu(args) -> None:
    print(
        "\nKeystroke acoustic recognition — what would you like to do?\n"
        "  1) record sessions   (collect labelled prose/random recordings)\n"
        "  2) train + evaluate  (preprocess -> train -> eval -> baseline)\n"
        "  3) live demo         (type and see what the model thinks)\n"
        "  4) everything        (1 -> 2 -> 3)\n"
        "  q) quit"
    )
    choice = ask("choice", "4")
    if choice in ("1", "collect"):
        collect(args)
    elif choice in ("2", "pipeline"):
        pipeline(args)
    elif choice in ("3", "live"):
        live(args)
    elif choice in ("4", "all"):
        do_all(args)
    else:
        print("bye.")


def do_all(args) -> None:
    collect(args)
    pipeline(args)
    print("\nNow the live demo: type and watch the model's guesses.")
    ask("press Enter to start the live demo", "")
    live(args)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="menu",
                   choices=("menu", "collect", "pipeline", "train", "eval", "live", "all", "split"))
    p.add_argument("--sessions", type=int, default=6, help="how many sessions to record in collect")
    p.add_argument("--keys", default="800", help="target keystrokes per recorded session")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--split-file", default=SPLIT_FILE)
    p.add_argument("--keys-set", default="letters+space", help="key set for preprocess (--keys)")
    p.add_argument("--epochs", default="60")
    p.add_argument("--lm-corpus", default="corpus/lm_train.txt")
    p.add_argument("--device", default=None, help="torch device for training/live (cpu|mps|cuda)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cmd = args.command
    if cmd == "menu":
        menu(args)
    elif cmd == "collect":
        collect(args)
    elif cmd in ("pipeline", "train", "eval"):
        pipeline(args)
    elif cmd == "split":
        build_split(args.raw_dir, args.split_file)
    elif cmd == "live":
        live(args)
    elif cmd == "all":
        do_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
