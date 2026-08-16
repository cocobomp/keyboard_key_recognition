#!/usr/bin/env python3
"""Simule l'utilisateur dans label_ui : valide/corrige depuis la vérité terrain.

Outil de TEST (sessions synthétiques uniquement) : rejoue la boucle
d'étiquetage assisté — file d'apprentissage actif, validation par lot d'un
cluster, correction, propagation kNN — via la même classe LabelApp que l'UI,
pour vérifier la chaîne de bout en bout sans clic humain.

En N « gestes » (défaut 12), il doit couvrir la grande majorité du corpus :
c'est exactement la promesse du §3 du cahier des charges.
"""

from __future__ import annotations

import argparse
import csv
import os
from types import SimpleNamespace

import sed_common as sc
from label_ui import LabelApp


def load_truth(raw_dir: str) -> dict[str, str]:
    truth: dict[str, str] = {}
    for session in sc.list_sessions(raw_dir):
        gt_csv = os.path.join(session.path, "ground_truth.csv")
        if not os.path.isfile(gt_csv):
            continue
        with open(gt_csv, newline="") as f:
            intervals = [(float(r["t_start_s"]), float(r["t_end_s"]), r["label"])
                         for r in csv.DictReader(f)]
        for ev in sc.read_events(session.session_id):
            best, ov_best = sc.BACKGROUND_LABEL, 0.0
            for a, b, lab in intervals:
                ov = min(b, ev["t_end_s"]) - max(a, ev["t_start_s"])
                if ov > ov_best:
                    best, ov_best = lab, ov
            truth[ev["full_id"]] = best
    return truth


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw-dir", default="data/synthetic")
    p.add_argument("--gestures", type=int, default=12)
    p.add_argument("--labels-file", default=sc.LABELS_FILE)
    p.add_argument("--fresh", action="store_true",
                   help="repart d'un labels.json vide")
    args = p.parse_args()

    if args.fresh and os.path.isfile(args.labels_file):
        os.unlink(args.labels_file)

    truth = load_truth(args.raw_dir)
    if not truth:
        raise SystemExit(f"aucune vérité terrain sous {args.raw_dir} — "
                         "outil réservé aux sessions synthétiques")

    app = LabelApp(SimpleNamespace(
        events_dir=sc.EVENTS_DIR, emb_dir=sc.EMB_DIR,
        clusters_file=sc.CLUSTERS_FILE, labels_file=args.labels_file,
        knn=10, min_cos=0.85, examples=3,
    ))

    print(f"{len(truth)} événements, {args.gestures} gestes simulés :")
    for gesture in range(args.gestures):
        queue = app.build_queue(1)
        if not queue:
            print("  file vide — plus rien d'incertain")
            break
        it = queue[0]
        if it["type"] == "cluster":
            members = app.members(it["cluster_id"])
            majority = max(
                set(truth.get(m, sc.BACKGROUND_LABEL) for m in members),
                key=lambda lab: sum(
                    1 for m in members if truth.get(m) == lab),
            )
            action = "confirm" if it["proposal"] == majority else "correct"
            res = app.apply({"action": action, "cluster_id": it["cluster_id"],
                             "label": majority, "proposal": it["proposal"]})
            print(f"  [{gesture + 1:2d}] cluster #{it['cluster_id']} "
                  f"({it['size']} ev) -> « {majority} » "
                  f"[{action}] +{res.get('n_propagated', 0)} propagés")
        else:
            fid = it["full_id"]
            lab = truth.get(fid, sc.BACKGROUND_LABEL)
            action = "confirm" if it["proposal"] == lab else "correct"
            res = app.apply({"action": action, "event_ids": [fid],
                             "label": lab, "proposal": it["proposal"]})
            print(f"  [{gesture + 1:2d}] événement {fid} -> « {lab} » "
                  f"[{action}] +{res.get('n_propagated', 0)} propagés")

    # bilan : couverture et justesse des labels posés (dont propagés)
    n_ok = n_tot = n_prop_ok = n_prop = 0
    for fid, rec in app.store.labels.items():
        if fid not in truth:
            continue
        n_tot += 1
        ok = rec["label"] == truth[fid]
        n_ok += ok
        if rec["provenance"] == "propagated":
            n_prop += 1
            n_prop_ok += ok
    cover = n_tot / max(1, len(truth))
    print(f"\ncouverture : {n_tot}/{len(truth)} ({sc.fmt_pct(cover)}), "
          f"justesse {sc.fmt_pct(n_ok / max(1, n_tot))}")
    if n_prop:
        print(f"labels propagés : {n_prop}, justesse "
              f"{sc.fmt_pct(n_prop_ok / n_prop)} (révocables dans l'UI)")
    print(f"-> {args.labels_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
