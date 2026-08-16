#!/usr/bin/env python3
"""Évaluation sur des sessions ENTIÈREMENT tenues à l'écart (nuits de test).

Deux niveaux de métriques :
  - par clip : précision / rappel / F1 par classe + matrice de confusion ;
  - event-based (temporel) : les événements consécutifs de même classe sont
    fusionnés en ÉPISODES, un épisode prédit compte juste s'il recouvre un
    épisode vrai de la même classe (IoU ≥ 0.3 ou début à ±2 s) — « un épisode
    de ronflement détecté au bon moment », pas des fenêtres isolées.

Baseline OBLIGATOIRE : le zéro-shot pré-entraîné seul (top-1 AudioSet mappé
vers le vocabulaire perso, sans aucun de mes labels) — le rapport montre ce
que l'adaptation personnelle apporte réellement par-dessus.

Vérité terrain, par session de test (--truth auto) :
  - ground_truth.csv présent (sessions synthétiques) : intervalle vrai ;
  - sinon : les labels manuels/batch posés via label_ui.py sur ces sessions
    (les labels « propagated » sont EXCLUS de la vérité : machine-made).

Garde-fou : check_disjoint(train du checkpoint, test) — enforcement point #3.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

import numpy as np

import sed_common as sc

IOU_THR = 0.3
ONSET_TOL_S = 2.0
EPISODE_GAP_S = 5.0


# ---------------------------------------------------------------------------
# Vérité terrain
# ---------------------------------------------------------------------------

def truth_for_session(sid: str, events: list[dict], store: sc.LabelStore,
                      raw_dir: str, mode: str):
    """-> (fid -> label vrai, intervalles vrais [(a, b, label)])."""
    gt_csv = os.path.join(raw_dir, sid, "ground_truth.csv")
    use_synth = mode == "synthetic" or (mode == "auto" and os.path.isfile(gt_csv))
    if use_synth:
        if not os.path.isfile(gt_csv):
            raise SystemExit(f"{gt_csv} introuvable (--truth synthetic)")
        with open(gt_csv, newline="") as f:
            intervals = [(float(r["t_start_s"]), float(r["t_end_s"]), r["label"])
                         for r in csv.DictReader(f)]
        by_fid = {}
        for ev in events:
            best, ov_best = sc.BACKGROUND_LABEL, 0.0
            for a, b, lab in intervals:
                ov = min(b, ev["t_end_s"]) - max(a, ev["t_start_s"])
                if ov > ov_best:
                    best, ov_best = lab, ov
            by_fid[ev["full_id"]] = best
        return by_fid, intervals, "ground_truth.csv"
    by_fid, intervals = {}, []
    for ev in events:
        rec = store.get(ev["full_id"])
        if rec is not None and rec["provenance"] in ("manual", "batch"):
            by_fid[ev["full_id"]] = rec["label"]
            intervals.append((ev["t_start_s"], ev["t_end_s"], rec["label"]))
    return by_fid, intervals, "labels manuels/batch"


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def clip_metrics(y_true, y_pred, classes: list[str]) -> dict:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0.0
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    per_class = {c: {"precision": float(p[i]), "recall": float(r[i]),
                     "f1": float(f1[i]), "support": int(sup[i])}
                 for i, c in enumerate(classes)}
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()
                              if v["support"] > 0])) if per_class else 0.0
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if len(y_true) else 0.0
    return {"per_class": per_class, "macro_f1": macro_f1, "accuracy": acc,
            "confusion": cm.tolist()}


def merge_episodes(intervals, gap: float = EPISODE_GAP_S):
    """[(a, b, label)] -> épisodes par classe (fusion si silence < gap)."""
    by_class: dict[str, list[list[float]]] = {}
    for a, b, lab in sorted(intervals):
        eps = by_class.setdefault(lab, [])
        if eps and a - eps[-1][1] <= gap:
            eps[-1][1] = max(eps[-1][1], b)
        else:
            eps.append([a, b])
    return by_class


def event_metrics(gt_intervals, pred_intervals) -> dict:
    """P/R/F1 événementiels par classe (appariement IoU ou tolérance d'onset)."""
    gt_eps = merge_episodes([iv for iv in gt_intervals
                             if iv[2] != sc.BACKGROUND_LABEL])
    pred_eps = merge_episodes([iv for iv in pred_intervals
                               if iv[2] != sc.BACKGROUND_LABEL])
    out = {}
    for lab in sorted(set(gt_eps) | set(pred_eps)):
        gts = gt_eps.get(lab, [])
        preds = pred_eps.get(lab, [])
        used = [False] * len(gts)
        tp = 0
        for pa, pb in preds:
            best_j, best_iou = -1, 0.0
            for j, (ga, gb) in enumerate(gts):
                if used[j]:
                    continue
                inter = max(0.0, min(pb, gb) - max(pa, ga))
                union = max(pb, gb) - min(pa, ga)
                iou = inter / union if union > 0 else 0.0
                ok = iou >= IOU_THR or (inter > 0 and abs(pa - ga) <= ONSET_TOL_S)
                if ok and iou >= best_iou:
                    best_j, best_iou = j, iou
            if best_j >= 0:
                used[best_j] = True
                tp += 1
        fp, fn = len(preds) - tp, len(gts) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1,
                    "n_true_episodes": len(gts), "n_pred_episodes": len(preds)}
    return out


def zeroshot_predictions(zeroshot_by_fid: dict[str, list]) -> dict[str, str]:
    preds = {}
    for fid, zs in zeroshot_by_fid.items():
        preds[fid] = sc.map_zeroshot(zs[0][0]) if zs else sc.BACKGROUND_LABEL
    return preds


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def confusion_png(cm, classes: list[str], path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    fig, ax = plt.subplots(figsize=(1.1 + 0.65 * len(classes),
                                    1.0 + 0.6 * len(classes)), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right",
                  fontsize=8, color="#52514e")
    ax.set_yticks(range(len(classes)), classes, fontsize=8, color="#52514e")
    ax.set_xlabel("prédit", fontsize=9, color="#898781")
    ax.set_ylabel("vrai", fontsize=9, color="#898781")
    ax.set_title(title, fontsize=10, color="#0b0b0b")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="#ffffff" if norm[i, j] > 0.55 else "#0b0b0b")
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------

def metrics_table(m: dict) -> list[str]:
    lines = ["| classe | précision | rappel | F1 | n |",
             "|---|---|---|---|---|"]
    for lab, v in sorted(m["per_class"].items()):
        sup = v.get("support", v.get("n_true_episodes", 0))
        if sup == 0:
            continue  # ex. « (autre) » : jamais dans la vérité terrain
        lines.append(f"| {lab} | {v['precision']:.2f} | {v['recall']:.2f} | "
                     f"{v['f1']:.2f} | {sup} |")
    return lines


def event_table(m: dict) -> list[str]:
    lines = ["| classe | précision | rappel | F1 | épisodes vrais | prédits |",
             "|---|---|---|---|---|---|"]
    for lab, v in sorted(m.items()):
        lines.append(f"| {lab} | {v['precision']:.2f} | {v['recall']:.2f} | "
                     f"{v['f1']:.2f} | {v['n_true_episodes']} | "
                     f"{v['n_pred_episodes']} |")
    return lines


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", default=os.path.join(sc.MODELS_DIR,
                                                        "sed_clf.joblib"))
    p.add_argument("--test-sessions", nargs="*", default=None,
                   help="défaut : le split test stocké dans le checkpoint")
    p.add_argument("--raw-dir", default=sc.RAW_DIR,
                   help="où chercher ground_truth.csv (sessions synthétiques)")
    p.add_argument("--events-dir", default=sc.EVENTS_DIR)
    p.add_argument("--emb-dir", default=sc.EMB_DIR)
    p.add_argument("--labels-file", default=sc.LABELS_FILE)
    p.add_argument("--truth", choices=("auto", "synthetic", "labels"),
                   default="auto")
    p.add_argument("--results-dir", default=sc.RESULTS_DIR)
    args = p.parse_args()

    import joblib

    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f"{args.checkpoint} introuvable — lancer train_events.py")
    ckpt = joblib.load(args.checkpoint)
    test_ids = list(args.test_sessions or ckpt["splits"].get("test") or [])
    if not test_ids:
        raise SystemExit("aucune session de test (checkpoint sans split test "
                         "et --test-sessions absent)")
    # enforcement point #3 : le test ne recoupe JAMAIS le train du checkpoint
    sc.check_disjoint(train=ckpt["splits"]["train"], test=test_ids)

    X, ids, zeroshot, _speech, config = sc.load_embeddings(test_ids, args.emb_dir)
    if config.get("backend") != ckpt.get("embedding_backend"):
        raise SystemExit(
            f"FATAL: embeddings de test ({config.get('backend')}) ≠ backend "
            f"d'entraînement ({ckpt.get('embedding_backend')})"
        )
    store = sc.LabelStore(args.labels_file)
    zs_by_fid = {fid: zs for fid, zs in zip(ids, zeroshot)}

    # vérité terrain + événements par session
    truth: dict[str, str] = {}
    gt_intervals_all: list[tuple] = []
    ev_span: dict[str, tuple] = {}
    truth_src = None
    for sid in test_ids:
        events = sc.read_events(sid, args.events_dir)
        for ev in events:
            # temps global unique entre sessions : décalage par session
            ev_span[ev["full_id"]] = (sid, ev["t_start_s"], ev["t_end_s"])
        t, intervals, src = truth_for_session(sid, events, store, args.raw_dir,
                                              args.truth)
        truth.update(t)
        gt_intervals_all.extend((sid, a, b, lab) for a, b, lab in intervals)
        truth_src = src
    evaluable = [fid for fid in ids if fid in truth]
    if not evaluable:
        raise SystemExit(
            "aucune vérité terrain sur les sessions de test — étiqueter des "
            "événements de ces sessions via label_ui.py (ou --truth synthetic)"
        )
    y_true = [truth[fid] for fid in evaluable]

    # prédictions du modèle adapté
    keep_idx = [i for i, fid in enumerate(ids) if fid in truth]
    y_model = ckpt["clf"].predict(X[keep_idx]).tolist()

    # baseline zéro-shot (sans aucun label perso)
    has_zeroshot = any(zs_by_fid.get(fid) for fid in evaluable)
    y_zero = ([zeroshot_predictions({f: zs_by_fid.get(f) or []
                                     for f in evaluable})[f] for f in evaluable]
              if has_zeroshot else None)

    # les prédictions hors du vocabulaire de la vérité terrain (surtout la
    # baseline : classes AudioSet brutes) sont repliées dans « (autre) » —
    # matrices compactes, macro-F1 comparables entre les deux systèmes
    truth_classes = sorted(set(y_true))
    OTHER = "(autre)"

    def fold(y):
        return [lab if lab in truth_classes else OTHER for lab in y]

    y_model = fold(y_model)
    y_zero = fold(y_zero) if y_zero else None
    classes = truth_classes + ([OTHER] if OTHER in y_model + (y_zero or [])
                               else [])
    m_model = clip_metrics(y_true, y_model, classes)
    m_zero = clip_metrics(y_true, y_zero, classes) if y_zero else None

    # event-based : timeline par session (offsets pour éviter les collisions)
    def to_timeline(pred_by_fid: dict[str, str]):
        offs, cur, out = {}, 0.0, []
        for sid in test_ids:
            offs[sid] = cur
            cur += 24 * 3600.0
        for fid, lab in pred_by_fid.items():
            sid, a, b = ev_span[fid]
            out.append((offs[sid] + a, offs[sid] + b, lab))
        return out, offs

    pred_tl, offs = to_timeline(dict(zip(evaluable, y_model)))
    gt_tl = [(offs[sid] + a, offs[sid] + b, lab)
             for sid, a, b, lab in gt_intervals_all]
    m_event = event_metrics(gt_tl, pred_tl)
    m_event_zero = (event_metrics(gt_tl, to_timeline(dict(zip(evaluable,
                                                              y_zero)))[0])
                    if y_zero else None)

    # sorties
    sc.ensure_dir(args.results_dir)
    confusion_png(m_model["confusion"], classes,
                  os.path.join(args.results_dir, "confusion.png"),
                  "Matrice de confusion (modèle adapté, sessions de test)")
    with open(os.path.join(args.results_dir, "confusion.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["vrai\\prédit"] + classes)
        for lab, row in zip(classes, m_model["confusion"]):
            w.writerow([lab] + list(row))

    results = {
        "test_sessions": test_ids,
        "truth_source": truth_src,
        "n_evaluated": len(evaluable),
        "embedding_backend": config.get("backend"),
        "model": {"clip": m_model, "event": m_event},
        "zeroshot_baseline": ({"clip": m_zero, "event": m_event_zero}
                              if m_zero else "indisponible (backend sans zéro-shot)"),
    }
    sc.write_json(os.path.join(args.results_dir, "eval_results.json"), results)

    lines = [
        "# Évaluation — sessions de test tenues à l'écart", "",
        f"- sessions de test : {', '.join(test_ids)}",
        f"- vérité terrain : {truth_src} ({len(evaluable)} événements évalués)",
        f"- backend d'embedding : {config.get('backend')}",
        f"- modèle : {ckpt.get('model_kind')} entraîné sur "
        f"{ckpt.get('n_train')} événements de {len(ckpt['splits']['train'])} "
        "session(s)", "",
        "## Par clip — modèle adapté (mes labels)", "",
        f"exactitude {sc.fmt_pct(m_model['accuracy'])}, "
        f"macro-F1 {m_model['macro_f1']:.2f}", "",
        *metrics_table(m_model), "",
    ]
    if m_zero:
        lines += [
            "## Par clip — baseline zéro-shot SEULE (aucun de mes labels)", "",
            f"exactitude {sc.fmt_pct(m_zero['accuracy'])}, "
            f"macro-F1 {m_zero['macro_f1']:.2f}", "",
            *metrics_table(m_zero), "",
            "## Ce que l'adaptation apporte", "",
            f"macro-F1 : {m_zero['macro_f1']:.2f} (zéro-shot) → "
            f"{m_model['macro_f1']:.2f} (adapté), "
            f"Δ = {m_model['macro_f1'] - m_zero['macro_f1']:+.2f}", "",
        ]
    else:
        lines += ["## Baseline zéro-shot", "",
                  "Indisponible : embeddings produits par un backend sans "
                  "sortie AudioSet (mel). Refaire embed.py --backend ast pour "
                  "la comparaison obligatoire zéro-shot vs adapté.", ""]
    lines += ["## Event-based (épisodes, IoU ≥ 0.3 ou onset ±2 s) — modèle adapté",
              "", *event_table(m_event), ""]
    if m_event_zero:
        lines += ["## Event-based — baseline zéro-shot", "",
                  *event_table(m_event_zero), ""]
    lines += ["![confusion](confusion.png)", "",
              "_Rappel : des scores sur sessions synthétiques valident la "
              "plomberie, pas les performances réelles. Les classes rares "
              "(peu d'épisodes vrais) ont des F1 très bruités._"]
    report = os.path.join(args.results_dir, "report.md")
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"test {test_ids} — vérité : {truth_src}")
    print(f"  par clip : adapté {sc.fmt_pct(m_model['accuracy'])} "
          f"(macro-F1 {m_model['macro_f1']:.2f})"
          + (f" | zéro-shot {sc.fmt_pct(m_zero['accuracy'])} "
             f"(macro-F1 {m_zero['macro_f1']:.2f})" if m_zero else
             " | zéro-shot indisponible (backend mel)"))
    for lab, v in sorted(m_event.items()):
        print(f"  épisodes {lab:<14} P {v['precision']:.2f}  R {v['recall']:.2f} "
              f" F1 {v['f1']:.2f}  ({v['n_true_episodes']} vrais)")
    print(f"-> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
