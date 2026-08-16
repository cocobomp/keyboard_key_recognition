#!/usr/bin/env python3
"""Segmentation NON supervisée : trouver « il s'est passé quelque chose ».

Aucune étiquette ici. Énergie RMS par trames de 50 ms avec un plancher de
bruit GLISSANT (percentile bas sur ~30 s) : une chambre la nuit est très
silencieuse mais son niveau de fond bouge (chauffage, rue, position) — le
seuil doit suivre. Un percentile bas reste robuste même pendant un long
épisode de ronflement (le ronflement occupe < 50 % du temps d'un cycle).

Hystérésis : déclenche à plancher + --threshold-db, relâche à + --release-db.
Filtres : durée min/max, fusion des événements proches, marge avant/après.

Sorties par session :
  data/events/<session_id>/events.csv        (event_id, temps, énergie, clip)
  data/events/<session_id>/clips/evt_XXXXX.flac

Vie privée : par défaut les chunks bruts sont CONSERVÉS (on rappelle comment
les supprimer). --delete-raw les supprime après extraction réussie des clips ;
il ne reste alors que les segments d'événements (quelques secondes chacun).
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone

import numpy as np
import soundfile as sf

import sed_common as sc

FRAME_S = 0.050
HOP_S = 0.025
FLOOR_BLOCK_S = 5.0     # percentile calculé par bloc de 5 s…
FLOOR_WINDOW_BLOCKS = 6  # …lissé sur 6 blocs = 30 s glissants
FLOOR_PERCENTILE = 20.0
ABS_FLOOR_DB = -90.0


def frame_rms_db(session: sc.Session, block_s: float = 60.0):
    """dB RMS par trame (50 ms, hop 25 ms) sur toute la session, en streaming.

    Lecture par blocs de `block_s` avec retenue : jamais 8 h en RAM, et la
    continuité entre chunks est gérée par Session.read_span.
    """
    sr = session.samplerate
    frame = int(FRAME_S * sr)
    hop = int(HOP_S * sr)
    n_total = session.n_frames
    carry = np.zeros(0, dtype=np.float32)
    dbs: list[np.ndarray] = []
    pos = 0
    while pos < n_total:
        stop = min(n_total, pos + int(block_s * sr))
        buf = np.concatenate([carry, session.read_span(pos, stop)])
        if len(buf) < frame:
            carry = buf
            pos = stop
            continue
        n_fr = (len(buf) - frame) // hop + 1
        idx = np.arange(n_fr)[:, None] * hop + np.arange(frame)[None, :]
        rms = np.sqrt(np.mean(buf[idx].astype(np.float64) ** 2, axis=1))
        dbs.append(20.0 * np.log10(rms + 1e-10))
        carry = buf[n_fr * hop :]
        pos = stop
    db = (np.concatenate(dbs) if dbs else np.zeros(0)).astype(np.float32)
    return db, hop, frame


def rolling_noise_floor(db: np.ndarray) -> np.ndarray:
    """Plancher de bruit par trame : percentile bas par bloc, médiane glissante."""
    if len(db) == 0:
        return db.copy()
    block = max(1, int(FLOOR_BLOCK_S / HOP_S))
    n_blocks = int(np.ceil(len(db) / block))
    pct = np.empty(n_blocks, dtype=np.float32)
    for i in range(n_blocks):
        pct[i] = np.percentile(db[i * block : (i + 1) * block], FLOOR_PERCENTILE)
    # médiane glissante centrée sur FLOOR_WINDOW_BLOCKS blocs
    half = FLOOR_WINDOW_BLOCKS // 2
    smooth = np.empty_like(pct)
    for i in range(n_blocks):
        smooth[i] = np.median(pct[max(0, i - half) : i + half + 1])
    floor = np.repeat(smooth, block)[: len(db)]
    return np.maximum(floor, ABS_FLOOR_DB)


def hysteresis_segments(db: np.ndarray, floor: np.ndarray,
                        on_db: float, off_db: float) -> list[tuple[int, int]]:
    """Trames -> [ (frame_on, frame_off) ) avec hystérésis on/off."""
    on_mask = db > floor + on_db
    off_mask = db < floor + off_db
    segments: list[tuple[int, int]] = []
    active = False
    start = 0
    for i in range(len(db)):
        if not active and on_mask[i]:
            active, start = True, i
        elif active and off_mask[i]:
            segments.append((start, i))
            active = False
    if active:
        segments.append((start, len(db)))
    return segments


def refine_segments(segments: list[tuple[float, float]], min_dur: float,
                    max_dur: float, merge_gap: float) -> list[tuple[float, float]]:
    """Fusion des voisins, filtre min, découpe des trop longs (en secondes)."""
    merged: list[list[float]] = []
    for a, b in segments:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out: list[tuple[float, float]] = []
    for a, b in merged:
        if b - a < min_dur:
            continue
        while b - a > max_dur:
            out.append((a, a + max_dur))
            a += max_dur
        out.append((a, b))
    return out


def detect_session(session: sc.Session, args) -> int:
    out_dir = os.path.join(args.events_dir, session.session_id)
    csv_path = os.path.join(out_dir, "events.csv")
    if os.path.isfile(csv_path) and not args.force:
        print(f"  {session.session_id}: events.csv existe déjà (--force pour refaire)")
        return 0
    if not session.chunks or not os.path.isfile(session.chunk_path(session.chunks[0])):
        print(f"  {session.session_id}: audio brut absent (déjà supprimé ?) — ignorée")
        return 0

    db, hop, frame = frame_rms_db(session)
    floor = rolling_noise_floor(db)
    segs_frames = hysteresis_segments(db, floor, args.threshold_db, args.release_db)
    sr = session.samplerate
    segs_s = [(a * HOP_S, b * HOP_S + FRAME_S) for a, b in segs_frames]
    segs_s = refine_segments(segs_s, args.min_dur, args.max_dur, args.merge_gap)

    clips_dir = sc.ensure_dir(os.path.join(out_dir, "clips"))
    rows = []
    for i, (t_a, t_b) in enumerate(segs_s):
        fa = max(0, int((t_a - args.pad) * sr))
        fb = min(session.n_frames, int((t_b + args.pad) * sr))
        clip = session.read_span(fa, fb)
        clip_rel = os.path.join("clips", f"evt_{i:05d}.flac")
        sf.write(os.path.join(out_dir, clip_rel), clip, sr,
                 format="FLAC", subtype="PCM_16")
        ia, ib = int(t_a / HOP_S), max(int(t_a / HOP_S) + 1, int(t_b / HOP_S))
        ev_db = float(np.mean(db[ia:ib]))
        ev_floor = float(np.median(floor[ia:ib]))
        t_utc = datetime.fromtimestamp(
            session.frame_to_utc(fa), tz=timezone.utc
        ).isoformat()
        rows.append({
            "event_id": f"evt_{i:05d}",
            "t_start_s": f"{t_a:.3f}",
            "t_end_s": f"{t_b:.3f}",
            "t_utc_start": t_utc,
            "rms_db": f"{ev_db:.1f}",
            "snr_db": f"{ev_db - ev_floor:.1f}",
            "clip_path": clip_rel,
        })

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sc.EVENT_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    active_s = sum(b - a for a, b in segs_s)
    print(f"  {session.session_id}: {len(rows)} événements "
          f"({active_s / 60:.1f} min actives / {session.duration_s / 60:.1f} min, "
          f"plancher médian {np.median(floor):.0f} dBFS)")

    if args.delete_raw:
        for chunk in session.chunks:
            try:
                os.unlink(session.chunk_path(chunk))
            except FileNotFoundError:
                pass
        session.meta["raw_deleted"] = True
        sc.write_json(session.meta_path, session.meta)
        print(f"    audio brut supprimé ({len(session.chunks)} chunks) — "
              "ne restent que les clips d'événements")
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sessions", nargs="*", default=None,
                   help="ids de sessions (défaut : toutes celles de --raw-dir)")
    p.add_argument("--raw-dir", default=sc.RAW_DIR)
    p.add_argument("--events-dir", default=sc.EVENTS_DIR)
    p.add_argument("--threshold-db", type=float, default=10.0,
                   help="déclenchement : plancher + N dB")
    p.add_argument("--release-db", type=float, default=6.0,
                   help="relâchement : plancher + N dB (hystérésis)")
    p.add_argument("--min-dur", type=float, default=0.3)
    p.add_argument("--max-dur", type=float, default=30.0)
    p.add_argument("--merge-gap", type=float, default=1.0)
    p.add_argument("--pad", type=float, default=0.25,
                   help="marge (s) ajoutée avant/après chaque clip")
    p.add_argument("--delete-raw", action="store_true",
                   help="supprime les chunks bruts après extraction des clips")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    sessions = sc.list_sessions(args.raw_dir, only=args.sessions)
    if not sessions:
        raise SystemExit(f"aucune session dans {args.raw_dir}")
    print(f"détection sur {len(sessions)} session(s) :")
    total = sum(detect_session(s, args) for s in sessions)
    print(f"total : {total} événements -> {args.events_dir}/")
    if not args.delete_raw:
        print("rappel vie privée : l'audio brut complet est conservé dans "
              f"{args.raw_dir}/ — relancer avec --delete-raw pour ne garder "
              "que les clips d'événements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
