#!/usr/bin/env python3
"""Rapport lisible d'une nuit : est-ce que je ronfle, quand, combien de temps.

Pour une session : % du temps passé à ronfler, nombre d'épisodes (bouffées
fusionnées si silence < 30 s), durée du plus long épisode, timeline heure par
heure, autres événements notables.

Prédictions : le modèle adapté si models/sed_clf.joblib existe (et même
backend d'embedding), sinon le zéro-shot, toujours écrasé par les labels
humains existants. Sortie : results/night_<session>.md + .png.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

import sed_common as sc

EPISODE_GAP_S = 30.0  # deux bouffées à moins de 30 s = même épisode

# Palette de référence (dataviz) — slots catégoriels, mode clair.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, MUTED, GRID, SURFACE = ("#0b0b0b", "#52514e", "#898781",
                                   "#e1e0d9", "#fcfcfb")


def episodes_of(intervals: list[tuple[float, float]],
                gap: float = EPISODE_GAP_S) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def fmt_dur(s: float) -> str:
    s = int(round(s))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {s % 3600 // 60:02d} min"


def timeline_png(session: sc.Session, by_label: dict[str, list],
                 top_labels: list[str], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dur = session.duration_s
    bin_s = 3600.0 if dur > 2 * 3600 else max(60.0, dur / 8)
    n_bins = int(np.ceil(dur / bin_s))
    t0 = session.meta.get("t_utc_start_epoch", 0.0)
    labels_x = [datetime.fromtimestamp(t0 + i * bin_s, tz=timezone.utc)
                .strftime("%H:%M") for i in range(n_bins)]

    minutes = {lab: np.zeros(n_bins) for lab in top_labels}
    for lab in top_labels:
        for a, b in by_label.get(lab, []):
            i0, i1 = int(a // bin_s), int(b // bin_s)
            for i in range(i0, min(i1, n_bins - 1) + 1):
                ov = min(b, (i + 1) * bin_s) - max(a, i * bin_s)
                minutes[lab][i] += max(0.0, ov) / 60.0

    fig, ax = plt.subplots(figsize=(8.2, 3.4), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    bottom = np.zeros(n_bins)
    x = np.arange(n_bins)
    for k, lab in enumerate(top_labels):
        ax.bar(x, minutes[lab], bottom=bottom, width=0.72,
               color=SERIES[k % len(SERIES)], label=lab,
               edgecolor=SURFACE, linewidth=1.2)  # 2px-équiv. : espaceur
        bottom += minutes[lab]
    ax.set_xticks(x[:: max(1, n_bins // 10)],
                  labels_x[:: max(1, n_bins // 10)], fontsize=8, color=INK2)
    ax.set_ylabel("minutes actives", fontsize=9, color=INK2)
    ax.set_title(f"{session.session_id} — activité sonore par tranche "
                 f"({'1 h' if bin_s == 3600 else fmt_dur(bin_s)}, heures UTC)",
                 fontsize=10, color=INK, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if len(top_labels) > 1:
        ax.legend(loc="upper right", fontsize=8, frameon=False,
                  labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--session", default=None,
                   help="défaut : la dernière session 'night' de --raw-dir")
    p.add_argument("--raw-dir", default=sc.RAW_DIR)
    p.add_argument("--events-dir", default=sc.EVENTS_DIR)
    p.add_argument("--emb-dir", default=sc.EMB_DIR)
    p.add_argument("--checkpoint",
                   default=os.path.join(sc.MODELS_DIR, "sed_clf.joblib"))
    p.add_argument("--labels-file", default=sc.LABELS_FILE)
    p.add_argument("--results-dir", default=sc.RESULTS_DIR)
    args = p.parse_args()

    sessions = sc.list_sessions(args.raw_dir)
    if args.session:
        session = next((s for s in sessions if s.session_id == args.session), None)
        if session is None:
            raise SystemExit(f"session {args.session} introuvable dans {args.raw_dir}")
    else:
        nights = [s for s in sessions if s.kind == "night"]
        if not nights:
            raise SystemExit(f"aucune session 'night' dans {args.raw_dir}")
        session = nights[-1]

    events = sc.read_events(session.session_id, args.events_dir)
    preds, source = sc.predict_labels([session.session_id], args.checkpoint,
                                      args.labels_file, args.emb_dir)

    by_label: dict[str, list] = defaultdict(list)
    for ev in events:
        lab = preds.get(ev["full_id"])
        if lab and lab != sc.BACKGROUND_LABEL:
            by_label[lab].append((ev["t_start_s"], ev["t_end_s"]))
    counts = Counter({lab: len(v) for lab, v in by_label.items()})
    top_labels = [lab for lab, _ in counts.most_common(len(SERIES))]

    dur = session.duration_s
    t0 = session.meta.get("t_utc_start_epoch", 0.0)
    lines = [
        f"# Nuit {session.session_id}", "",
        f"- début : {datetime.fromtimestamp(t0, tz=timezone.utc).isoformat()} "
        f"(UTC) — durée {fmt_dur(dur)}",
        f"- {len(events)} événements sonores détectés",
        f"- prédictions : {source}", "",
        "## Par type de son", "",
        "| son | événements | temps actif | % de la nuit | épisodes | "
        "plus long épisode |",
        "|---|---|---|---|---|---|",
    ]
    snore_total = 0.0
    for lab in top_labels:
        ivs = by_label[lab]
        total = sum(b - a for a, b in ivs)
        eps = episodes_of(ivs)
        longest = max((b - a for a, b in eps), default=0.0)
        if lab == "ronflement":
            snore_total = total
        lines.append(f"| {lab} | {len(ivs)} | {fmt_dur(total)} | "
                     f"{100 * total / dur:.1f}% | {len(eps)} | "
                     f"{fmt_dur(longest)} |")
    lines += ["",
              (f"**Ronflement : {fmt_dur(snore_total)} au total, soit "
               f"{100 * snore_total / dur:.1f}% de la nuit.**"
               if snore_total else
               "**Aucun ronflement détecté cette nuit"
               " (ou pas encore de labels/modèle pour le reconnaître).**"),
              ""]

    sc.ensure_dir(args.results_dir)
    png = os.path.join(args.results_dir, f"night_{session.session_id}.png")
    if top_labels:
        timeline_png(session, by_label, top_labels, png)
        lines += [f"![timeline](night_{session.session_id}.png)", ""]

    # épisodes de ronflement détaillés
    if "ronflement" in by_label:
        lines += ["## Épisodes de ronflement", ""]
        for a, b in episodes_of(by_label["ronflement"]):
            ha = datetime.fromtimestamp(t0 + a, tz=timezone.utc).strftime("%H:%M:%S")
            lines.append(f"- {ha} UTC — {fmt_dur(b - a)}")
        lines.append("")

    lines += ["_Les classes rares sont fragiles (peu d'exemples) ; "
              "corriger les erreurs dans label_ui.py améliore les nuits "
              "suivantes._"]
    md = os.path.join(args.results_dir, f"night_{session.session_id}.md")
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))
    print(f"\n-> {md}" + (f" + {png}" if top_labels else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
