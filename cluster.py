#!/usr/bin/env python3
"""Regroupement des embeddings en familles de sons récurrents.

But : pouvoir dire « tous ces 300 sons = ronflement » en un geste dans
label_ui.py. HDBSCAN par défaut (via scikit-learn ≥ 1.3, pas de compilation),
k-means en secours ; le « bruit » HDBSCAN (cluster -1) est gardé comme groupe
« divers » à trier via l'apprentissage actif.

Par cluster : taille, étiquette zéro-shot dominante (vote pondéré par le
score), exemples représentatifs (les plus proches du centroïde), dispersion.

Sortie : data/clusters.json  {algo, clusters: [...], assignments: {full_id: cid}}
"""

from __future__ import annotations

import argparse

import numpy as np

import sed_common as sc


def normalize_embed(X: np.ndarray, pca_dim: int | None) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    if pca_dim and 0 < pca_dim < min(Xn.shape):
        from sklearn.decomposition import PCA

        Xn = PCA(n_components=pca_dim, random_state=0).fit_transform(Xn)
    return Xn.astype(np.float32)


def run_hdbscan(Xn: np.ndarray, min_cluster_size: int) -> np.ndarray:
    from sklearn.cluster import HDBSCAN

    try:
        model = HDBSCAN(min_cluster_size=min_cluster_size, copy=True)
    except TypeError:  # anciennes versions sans paramètre copy
        model = HDBSCAN(min_cluster_size=min_cluster_size)
    return model.fit_predict(Xn)


def run_kmeans(Xn: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xn)


def dominant_zeroshot(zeroshot_rows: list[list]) -> tuple[str, float]:
    """Vote pondéré par le score sur les top-k zéro-shot du cluster."""
    votes: dict[str, float] = {}
    for row in zeroshot_rows:
        for label, score in row:
            votes[label] = votes.get(label, 0.0) + float(score)
    if not votes:
        return "?", 0.0
    label, weight = max(votes.items(), key=lambda kv: kv[1])
    return label, weight / max(1, len(zeroshot_rows))


def summarise(Xn: np.ndarray, ids: list[str], zeroshot: list[list],
              labels: np.ndarray, n_examples: int) -> list[dict]:
    clusters = []
    for cid in sorted(set(int(c) for c in labels)):
        mask = labels == cid
        idx = np.where(mask)[0]
        Xc = Xn[mask]
        centroid = Xc.mean(axis=0)
        d = np.linalg.norm(Xc - centroid, axis=1)
        order = idx[np.argsort(d)]
        zs_label, zs_conf = dominant_zeroshot([zeroshot[i] for i in idx])
        clusters.append({
            "cluster_id": cid,
            "size": int(mask.sum()),
            "zeroshot_label": zs_label,
            "zeroshot_confidence": round(zs_conf, 4),
            "examples": [ids[i] for i in order[:n_examples]],
            "spread": round(float(d.mean()), 4),
            "is_noise": cid == -1,
        })
    clusters.sort(key=lambda c: (c["is_noise"], -c["size"]))
    return clusters


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--emb-dir", default=sc.EMB_DIR)
    p.add_argument("--out", default=sc.CLUSTERS_FILE)
    p.add_argument("--algo", choices=("hdbscan", "kmeans"), default="hdbscan")
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--k", type=int, default=None,
                   help="k-means : nombre de clusters (défaut ~ n/25)")
    p.add_argument("--pca", type=int, default=50,
                   help="réduction PCA avant clustering (0 = désactivée)")
    p.add_argument("--examples", type=int, default=5)
    args = p.parse_args()

    X, ids, zeroshot, _speech, config = sc.load_embeddings(emb_dir=args.emb_dir)
    if len(ids) == 0:
        raise SystemExit("aucun embedding — lancer embed.py d'abord")
    Xn = normalize_embed(X, args.pca or None)

    if args.algo == "hdbscan":
        try:
            labels = run_hdbscan(Xn, args.min_cluster_size)
        except ImportError:
            print("HDBSCAN indisponible (scikit-learn < 1.3) — repli k-means")
            args.algo = "kmeans"
    if args.algo == "kmeans":
        k = args.k or max(2, len(ids) // 25)
        labels = run_kmeans(Xn, k)

    clusters = summarise(Xn, ids, zeroshot, labels, args.examples)
    out = {
        "algo": args.algo,
        "backend": config.get("backend"),
        "n_events": len(ids),
        "clusters": clusters,
        "assignments": {fid: int(c) for fid, c in zip(ids, labels)},
    }
    sc.write_json(args.out, out)

    n_noise = int((labels == -1).sum())
    print(f"{args.algo} : {len([c for c in clusters if not c['is_noise']])} clusters "
          f"+ {n_noise} événements « divers » (bruit) sur {len(ids)}")
    for c in clusters[:12]:
        name = "divers" if c["is_noise"] else f"#{c['cluster_id']}"
        print(f"  {name:>7}  {c['size']:4d} événements  "
              f"zéro-shot: {c['zeroshot_label']} ({c['zeroshot_confidence']:.2f})")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
