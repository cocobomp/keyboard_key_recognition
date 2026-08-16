#!/usr/bin/env python3
"""Étape 2 — modélisation temporelle SIMPLE de la séquence d'événements.

Volontairement minimal (avant tout modèle séquentiel) :
  - statistiques par heure : taux d'événements par classe et par heure UTC,
    sur les sessions de référence (le split train si splits.json existe) ;
  - chaîne de Markov d'ordre 1 sur la séquence des classes : « après un
    ronflement, il y a X % de chances que le prochain son soit … » ;
  - score de nuit anormale : log-vraisemblance moyenne des transitions d'une
    session sous le modèle appris sur les sessions de référence — une nuit
    très improbable mérite un coup d'œil.

À ne prendre au sérieux qu'avec un nombre raisonnable de nuits étiquetées.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

import sed_common as sc

START = "<début>"


def session_sequence(sid: str, preds: dict[str, str], events_dir: str):
    seq = []
    for ev in sc.read_events(sid, events_dir):
        lab = preds.get(ev["full_id"])
        if lab and lab != sc.BACKGROUND_LABEL:
            seq.append((ev["t_start_s"], ev["full_id"], lab))
    return [lab for _t, _f, lab in sorted(seq)]


class Markov1:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def fit(self, sequences: list[list[str]]) -> "Markov1":
        for seq in sequences:
            prev = START
            for lab in seq:
                self.counts[prev][lab] += 1
                self.vocab.add(lab)
                prev = lab
        return self

    def prob(self, prev: str, nxt: str) -> float:
        c = self.counts.get(prev, Counter())
        v = max(1, len(self.vocab))
        return (c[nxt] + self.alpha) / (sum(c.values()) + self.alpha * v)

    def next_probs(self, prev: str) -> list[tuple[str, float]]:
        probs = [(lab, self.prob(prev, lab)) for lab in sorted(self.vocab)]
        return sorted(probs, key=lambda kv: -kv[1])

    def mean_loglik(self, seq: list[str]) -> float | None:
        if len(seq) < 2:
            return None
        prev, ll = START, 0.0
        for lab in seq:
            ll += float(np.log(self.prob(prev, lab)))
            prev = lab
        return ll / len(seq)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw-dir", default=sc.RAW_DIR)
    p.add_argument("--events-dir", default=sc.EVENTS_DIR)
    p.add_argument("--emb-dir", default=sc.EMB_DIR)
    p.add_argument("--checkpoint",
                   default=os.path.join(sc.MODELS_DIR, "sed_clf.joblib"))
    p.add_argument("--labels-file", default=sc.LABELS_FILE)
    p.add_argument("--split-file", default=sc.SPLIT_FILE)
    args = p.parse_args()

    sessions = sc.list_sessions(args.raw_dir)
    sids = [s.session_id for s in sessions
            if os.path.isfile(sc.events_csv_path(s.session_id, args.events_dir))]
    if not sids:
        raise SystemExit("aucun événement détecté — lancer la chaîne d'abord")
    preds, source = sc.predict_labels(sids, args.checkpoint, args.labels_file,
                                      args.emb_dir)
    print(f"prédictions : {source}\n")

    # 1) statistiques par heure UTC
    by_hour: dict[int, Counter] = defaultdict(Counter)
    for s in sessions:
        if s.session_id not in sids:
            continue
        for ev in sc.read_events(s.session_id, args.events_dir):
            lab = preds.get(ev["full_id"])
            if not lab or lab == sc.BACKGROUND_LABEL:
                continue
            t0 = s.meta.get("t_utc_start_epoch", 0.0)
            hour = datetime.fromtimestamp(t0 + ev["t_start_s"],
                                          tz=timezone.utc).hour
            by_hour[hour][lab] += 1
    print("événements par heure UTC (toutes sessions) :")
    for hour in sorted(by_hour):
        tops = ", ".join(f"{lab}×{n}" for lab, n in by_hour[hour].most_common(4))
        print(f"  {hour:02d}h  {tops}")

    # 2) chaîne de Markov apprise sur les sessions de référence
    ref = sids
    if os.path.isfile(args.split_file):
        split = sc.read_json(args.split_file)
        ref = [sid for sid in split.get("train", []) if sid in sids] or sids
    model = Markov1().fit([session_sequence(sid, preds, args.events_dir)
                           for sid in ref])
    print(f"\ntransitions (Markov ordre 1, appris sur {len(ref)} session(s)) :")
    for prev in [START] + sorted(model.vocab):
        tops = ", ".join(f"{lab} {100 * pr:.0f}%"
                         for lab, pr in model.next_probs(prev)[:3])
        print(f"  après {prev:<12} -> {tops}")

    # 3) nuits anormales : log-vraisemblance moyenne par session
    print("\nscore de normalité par session (log-vraisemblance moyenne ; "
          "plus bas = plus atypique) :")
    scores = []
    for sid in sids:
        ll = model.mean_loglik(session_sequence(sid, preds, args.events_dir))
        if ll is not None:
            scores.append((ll, sid))
    for ll, sid in sorted(scores):
        flag = "  <- atypique" if scores and ll == min(scores)[0] and \
            len(scores) > 2 else ""
        print(f"  {sid:<24} {ll:7.2f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
