#!/usr/bin/env python3
"""Classifieur léger sur les embeddings — split PAR SESSION obligatoire.

Grâce au pré-entraînement AudioSet, une régression logistique sur les
embeddings suffit : quelques centaines d'exemples étiquetés donnent déjà un
modèle personnel utile (option --model mlp pour un petit MLP).

Discipline expérimentale (non négociable, reprise du projet clavier) :
  - le split est PAR SESSION (par nuit), jamais par événement aléatoire —
    deux ronflements de la même nuit sont quasi identiques, un split
    aléatoire mesurerait de la mémorisation ;
  - check_disjoint() fait échouer le programme si une session apparaît
    dans deux splits (enforcement point #2 ; #1 = run.py build_split,
    #3 = eval_events.py).

Déséquilibre de classes : class_weight="balanced" (le fond sonore domine) ;
les labels propagés comptent avec un poids réduit (--propagated-weight).

Checkpoint : models/sed_clf.joblib (clf + vocab + splits + backend).
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np

import sed_common as sc


def resolve_splits(args) -> dict[str, list[str]]:
    if args.train_sessions or args.test_sessions:
        split = {
            "train": list(args.train_sessions or []),
            "val": list(args.val_sessions or []),
            "test": list(args.test_sessions or []),
        }
    else:
        if not os.path.isfile(args.split_file):
            raise SystemExit(
                f"{args.split_file} introuvable — lancer `python run.py split` "
                "ou passer --train-sessions/--test-sessions explicitement."
            )
        split = sc.read_json(args.split_file)
        split = {k: list(split.get(k, [])) for k in ("train", "val", "test")}
    if not split["train"]:
        raise SystemExit("split sans sessions de train")
    sc.check_disjoint(**{k: v for k, v in split.items() if v})
    return split


def build_dataset(session_ids: list[str], store: sc.LabelStore, args):
    """(X, y, poids, full_ids) pour les événements étiquetés de ces sessions."""
    X_all, ids, _zs, _sp, config = sc.load_embeddings(session_ids, args.emb_dir)
    xs, ys, ws, kept = [], [], [], []
    for i, fid in enumerate(ids):
        rec = store.get(fid)
        if rec is None:
            continue
        w = args.propagated_weight if rec["provenance"] == "propagated" else 1.0
        if w <= 0:
            continue
        xs.append(X_all[i])
        ys.append(rec["label"])
        ws.append(w)
        kept.append(fid)
    X = np.asarray(xs, dtype=np.float32) if xs else np.zeros((0, X_all.shape[1]))
    return X, np.asarray(ys), np.asarray(ws, dtype=np.float32), kept, config


def make_model(kind: str):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0),
        )
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # NB : MLPClassifier ne supporte pas sample_weight ni class_weight —
        # le poids des labels propagés est appliqué par duplication d'exemples.
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64,), max_iter=800, random_state=0),
        )
    raise SystemExit(f"modèle inconnu : {kind}")


def fit(model, X, y, w, kind: str):
    if kind == "logreg":
        model.fit(X, y, logisticregression__sample_weight=w)
    else:
        # duplication approx. pour émuler les poids (propagated_weight < 1)
        reps = np.maximum(1, np.round(w / w.min()).astype(int)) if len(w) else []
        Xr = np.repeat(X, reps, axis=0)
        yr = np.repeat(y, reps, axis=0)
        model.fit(Xr, yr)
    return model


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split-file", default=sc.SPLIT_FILE)
    p.add_argument("--train-sessions", nargs="*", default=None)
    p.add_argument("--val-sessions", nargs="*", default=None)
    p.add_argument("--test-sessions", nargs="*", default=None)
    p.add_argument("--emb-dir", default=sc.EMB_DIR)
    p.add_argument("--labels-file", default=sc.LABELS_FILE)
    p.add_argument("--model", choices=("logreg", "mlp"), default="logreg")
    p.add_argument("--propagated-weight", type=float, default=0.5,
                   help="poids des labels propagés (0 = ignorés)")
    p.add_argument("--min-per-class", type=int, default=3,
                   help="classes plus rares : ignorées à l'entraînement (averti)")
    p.add_argument("--out", default=os.path.join(sc.MODELS_DIR, "sed_clf.joblib"))
    args = p.parse_args()

    split = resolve_splits(args)
    store = sc.LabelStore(args.labels_file)
    if not store.labels:
        raise SystemExit(f"aucun label dans {args.labels_file} — "
                         "étiqueter d'abord via label_ui.py")

    X, y, w, ids, config = build_dataset(split["train"], store, args)
    counts = Counter(y.tolist())
    rare = {c for c, n in counts.items() if n < args.min_per_class}
    if rare:
        print(f"attention : classes trop rares ignorées à l'entraînement "
              f"(< {args.min_per_class} ex.) : {sorted(rare)}")
        keep = np.asarray([lab not in rare for lab in y])
        X, y, w = X[keep], y[keep], w[keep]
    if len(set(y)) < 2:
        raise SystemExit("moins de 2 classes étiquetées sur les sessions de "
                         "train — étiqueter davantage via label_ui.py")

    print(f"train : {len(y)} événements étiquetés sur {len(split['train'])} "
          f"session(s) {split['train']}")
    for lab, n in sorted(Counter(y.tolist()).items(), key=lambda kv: -kv[1]):
        print(f"  {lab:<16} {n:4d}")

    model = fit(make_model(args.model), X, y, w, args.model)

    # validation (sessions tenues à l'écart du train, jamais celles de test)
    val_acc = None
    if split["val"]:
        Xv, yv, _wv, _idv, _ = build_dataset(split["val"], store, args)
        if len(yv):
            val_acc = float(np.mean(model.predict(Xv) == yv))
            print(f"val ({split['val']}) : {len(yv)} ex., "
                  f"exactitude {sc.fmt_pct(val_acc)}")
        else:
            print(f"val ({split['val']}) : aucun événement étiqueté")

    import joblib

    sc.ensure_dir(os.path.dirname(args.out))
    joblib.dump({
        "clf": model,
        "model_kind": args.model,
        "classes": sorted(set(y.tolist())),
        "splits": split,
        "embedding_backend": config.get("backend"),
        "embedding_config": config,
        "n_train": int(len(y)),
        "val_acc": val_acc,
        "propagated_weight": args.propagated_weight,
    }, args.out)
    print(f"-> {args.out}")
    print("évaluer sur les sessions de test : python eval_events.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
