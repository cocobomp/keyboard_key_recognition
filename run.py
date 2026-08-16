#!/usr/bin/env python3
"""Programme unique : enchaîne enregistrer -> détecter -> embarquer ->
clusteriser -> étiqueter -> entraîner -> évaluer -> rapport.

Même structure que le run.py du projet clavier : sous-commande positionnelle
OU menu interactif, chaque étape lancée en sous-processus (`sh()`), et le
split par session construit ici une fois pour toutes (enforcement point #1
du garde-fou check_disjoint ; #2 = train_events.py, #3 = eval_events.py).

Validation sans micro (à faire AVANT d'enregistrer quoi que ce soit) :
    python run.py all --raw-dir data/synthetic --backend mel --simulate-labels
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import sed_common as sc

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))

MENU = """
  1) record    enregistrer une session (nuit ou jour)
  2) detect    détecter les événements (non supervisé)
  3) embed     embeddings + étiquettes zéro-shot
  4) cluster   regrouper en familles de sons
  5) label     ouvrir l'UI d'étiquetage assisté
  6) split     construire le split par session (splits.json)
  7) train     entraîner le classifieur perso
  8) eval      évaluer sur les sessions de test
  9) report    rapport de nuit
 10) predict   stats horaires + chaîne de Markov (étape 2)
 11) synth     générer des sessions synthétiques (validation sans micro)
 12) pipeline  detect -> embed -> cluster -> (label) -> split -> train ->
               eval -> report
  q) quitter
"""


def sh(script: str, *cargs, check: bool = True) -> int:
    cmd = [PY, os.path.join(HERE, script), *[str(a) for a in cargs]]
    print(f"\n$ {' '.join(cmd)}\n")
    try:
        rc = subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        return 130  # l'enfant a déjà géré l'interruption
    if check and rc != 0:
        raise SystemExit(f"étape en échec ({script}, exit {rc}).")
    return rc


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    try:
        ans = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        ans = ""
    return ans or (default or "")


def summarise_sessions(raw_dir: str) -> list[sc.Session]:
    sessions = sc.list_sessions(raw_dir)
    nights = [s for s in sessions if s.kind == "night"]
    days = [s for s in sessions if s.kind == "day"]
    print(f"{len(sessions)} session(s) dans {raw_dir} "
          f"({len(nights)} nuits, {len(days)} jours) :")
    for s in sessions:
        print(f"  {s.session_id:<28} {s.kind:<6} {s.duration_s / 60:7.1f} min")
    if len(nights) < 4 or len(days) < 3:
        print("objectif du protocole : ≥ 4 nuits et ≥ 3 sessions de jour "
              "(voir README).")
    return sessions


def build_split(raw_dir: str, out_file: str) -> dict:
    """Split PAR SESSION : test = dernière nuit + dernier jour, val = une
    session, train = le reste. Toujours vérifié par check_disjoint."""
    sessions = sc.list_sessions(raw_dir)
    if len(sessions) < 3:
        raise SystemExit("il faut au moins 3 sessions pour un split "
                         "train/val/test par session.")
    nights = [s.session_id for s in sessions if s.kind == "night"]
    days = [s.session_id for s in sessions if s.kind == "day"]
    test = ([nights[-1]] if nights else []) + ([days[-1]] if days else [])
    rest = [sid for sid in nights + days if sid not in test]
    val = [nights[-2]] if len(nights) >= 2 and nights[-2] in rest else rest[:1]
    train = [sid for sid in rest if sid not in val]
    if not train:
        train, val = val, []
    split = {"train": train, "val": val, "test": test}
    sc.check_disjoint(**{k: v for k, v in split.items() if v})
    sc.write_json(out_file, split)
    print(f"split par session -> {out_file}")
    for k in ("train", "val", "test"):
        print(f"  {k:<5} {split[k]}")
    return split


def pipeline(args) -> None:
    summarise_sessions(args.raw_dir)
    sh("detect_events.py", "--raw-dir", args.raw_dir)
    sh("embed.py", "--backend", args.backend)
    sh("cluster.py")
    if args.simulate_labels:
        sh("simulate_labeling.py", "--raw-dir", args.raw_dir)
    else:
        print("\nÉtiquetage : lance `python run.py label` dans un autre "
              "terminal (ou maintenant), valide quelques clusters, puis "
              "reviens ici.")
        ask("Appuie sur Entrée quand l'étiquetage te convient", "")
    build_split(args.raw_dir, args.split_file)
    sh("train_events.py", "--split-file", args.split_file)
    sh("eval_events.py", "--raw-dir", args.raw_dir)
    split = sc.read_json(args.split_file)
    nights_test = [sid for sid in split["test"]
                   if any(s.kind == "night" for s in sc.list_sessions(args.raw_dir)
                          if s.session_id == sid)]
    if nights_test:
        sh("night_report.py", "--raw-dir", args.raw_dir,
           "--session", nights_test[0])
    print("\nFichiers clés : results/report.md, results/eval_results.json, "
          "results/night_*.md, data/labels.json")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", nargs="?", default="menu",
                   choices=("menu", "record", "detect", "embed", "cluster",
                            "label", "split", "train", "eval", "report",
                            "predict", "synth", "pipeline", "all"))
    p.add_argument("--raw-dir", default=sc.RAW_DIR)
    p.add_argument("--split-file", default=sc.SPLIT_FILE)
    p.add_argument("--backend", default="ast", choices=("ast", "mel"))
    p.add_argument("--simulate-labels", action="store_true",
                   help="(synthétique) rejoue l'étiquetage depuis la vérité "
                        "terrain au lieu d'ouvrir l'UI")
    p.add_argument("--kind", default=None, choices=(None, "night", "day"),
                   help="record : type de session")
    args, extra = p.parse_known_args()

    cmd = args.command
    if cmd == "menu":
        print(MENU)
        choice = ask("choix", "12")
        cmd = {"1": "record", "2": "detect", "3": "embed", "4": "cluster",
               "5": "label", "6": "split", "7": "train", "8": "eval",
               "9": "report", "10": "predict", "11": "synth",
               "12": "pipeline"}.get(choice, "q")
        if cmd == "q":
            return 0

    if cmd == "record":
        kind = args.kind or ask("nuit ou jour ? (night/day)", "night")
        tag = ask("tag (position du micro, ex. chevet)", "")
        sh("record_audio.py", "--kind", kind,
           *(["--tag", tag] if tag else []), *extra, check=False)
        print("\npense à varier la position du micro ENTRE les sessions "
              "(jamais pendant) — voir README.")
    elif cmd == "detect":
        sh("detect_events.py", "--raw-dir", args.raw_dir, *extra)
    elif cmd == "embed":
        sh("embed.py", "--backend", args.backend, *extra)
    elif cmd == "cluster":
        sh("cluster.py", *extra)
    elif cmd == "label":
        sh("label_ui.py", *extra, check=False)
    elif cmd == "split":
        summarise_sessions(args.raw_dir)
        build_split(args.raw_dir, args.split_file)
    elif cmd == "train":
        sh("train_events.py", "--split-file", args.split_file, *extra)
    elif cmd == "eval":
        sh("eval_events.py", "--raw-dir", args.raw_dir, *extra)
    elif cmd == "report":
        sh("night_report.py", "--raw-dir", args.raw_dir, *extra)
    elif cmd == "predict":
        sh("predict_events.py", "--raw-dir", args.raw_dir, *extra)
    elif cmd == "synth":
        sh("make_synthetic_sessions.py", *extra)
        print("\nvalider la chaîne : python run.py all --raw-dir "
              "data/synthetic --backend mel --simulate-labels")
    elif cmd in ("pipeline", "all"):
        pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
