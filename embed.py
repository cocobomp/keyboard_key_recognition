#!/usr/bin/env python3
"""Embeddings + étiquettes zéro-shot par événement détecté.

C'est la brique qui rend l'étiquetage assisté possible : un modèle
pré-entraîné sur AudioSet fournit, par clip, (a) un embedding qui rapproche
les sons semblables, (b) des étiquettes proposées gratuitement — AudioSet
contient déjà « Snoring », « Breathing », « Creak », « Door », « Speech »,
« Typing »… donc le système sait proposer « ronflement » dès le premier soir.

Backends (le choix est isolé derrière embed_batch(), remplaçable) :
  ast  (défaut)  MIT/ast-finetuned-audioset-10-10-0.4593 via transformers.
                 PyTorch → tourne sur Apple Silicon (device mps), CPU sinon.
                 Embedding 768-d (moyenne des tokens du dernier bloc),
                 527 classes AudioSet en sortie sigmoïde.
  mel            Fallback léger SANS torch ni téléchargement : statistiques
                 de log-mels (moyenne/écart-type par bande + deltas, 256-d).
                 Pas d'étiquettes zéro-shot ni de détection de parole —
                 sert à valider la chaîne et de solution de secours.
  (alternatives non implémentées mais compatibles : PANNs CNN14, YAMNet,
   OpenL3 — écrire une classe avec la même interface embed_batch().)

Vie privée :
  --drop-speech          écarte les événements détectés comme parole
                         (score AudioSet) ET supprime leurs clips audio.
  --privacy embeddings-only
                         après embedding, supprime TOUS les clips audio ;
                         ne restent que embeddings + events.csv (temps).

Sortie : data/embeddings/<session_id>.npz
         (X, event_ids, zeroshot_json, speech_score, config_json)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import soundfile as sf

import sed_common as sc

AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
# Un événement = quelques secondes ; AST attend ≤ ~10.24 s (padding géré par
# le feature extractor). On tronque les clips plus longs.
MAX_CLIP_S = 10.0

SPEECH_NAME_PARTS = (
    "speech", "conversation", "narration", "babbl", "whisper", "shout",
    "yell", "chatter", "screaming",
)


# ---------------------------------------------------------------------------
# Backends — interface : embed_batch(waves, sr) -> (X, zeroshot, speech_score)
#   waves        : liste de np.ndarray float32 mono (longueurs variables)
#   X            : (n, d) float32
#   zeroshot     : liste (par clip) de [label, score] triés par score décroissant
#   speech_score : (n,) float32 dans [0, 1] (0 si le backend ne sait pas)
# ---------------------------------------------------------------------------

class MelBackend:
    """Statistiques de log-mels : déterministe, zéro téléchargement, zéro torch."""

    name = "mel"

    def __init__(self, n_mels: int = 64):
        self.n_mels = n_mels

    @property
    def config(self) -> dict:
        return {"backend": self.name, "n_mels": self.n_mels, "dim": self.n_mels * 4}

    def embed_batch(self, waves, sr):
        import librosa

        feats = []
        for w in waves:
            m = librosa.feature.melspectrogram(
                y=w, sr=sr, n_mels=self.n_mels, n_fft=1024, hop_length=256,
                fmin=50, fmax=sr // 2,
            )
            logm = librosa.power_to_db(m + 1e-10)
            d = np.diff(logm, axis=1) if logm.shape[1] > 1 else np.zeros_like(logm)
            feats.append(np.concatenate([
                logm.mean(axis=1), logm.std(axis=1), d.mean(axis=1), d.std(axis=1),
            ]))
        X = np.asarray(feats, dtype=np.float32)
        return X, [[] for _ in waves], np.zeros(len(waves), dtype=np.float32)


class ASTBackend:
    """Audio Spectrogram Transformer fine-tuné AudioSet (768-d + 527 classes)."""

    name = "ast"

    def __init__(self, device: str | None = None, topk: int = 5):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.torch = torch
        if device is None:
            device = ("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.extractor = ASTFeatureExtractor.from_pretrained(AST_MODEL)
        self.model = ASTForAudioClassification.from_pretrained(AST_MODEL)
        self.model.eval().to(device)
        self.id2label = self.model.config.id2label
        self.topk = topk
        self.speech_ids = [
            i for i, lab in self.id2label.items()
            if any(part in lab.lower() for part in SPEECH_NAME_PARTS)
        ]

    @property
    def config(self) -> dict:
        return {"backend": self.name, "model": AST_MODEL, "device": self.device,
                "dim": 768, "topk": self.topk}

    def embed_batch(self, waves, sr):
        torch = self.torch
        waves = [w[: int(MAX_CLIP_S * sr)] for w in waves]
        inputs = self.extractor(waves, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            out = self.model(
                inputs.input_values.to(self.device), output_hidden_states=True
            )
            emb = out.hidden_states[-1].mean(dim=1)          # (n, 768)
            probs = torch.sigmoid(out.logits)                # (n, 527)
        emb = emb.cpu().numpy().astype(np.float32)
        probs = probs.cpu().numpy()
        zeroshot = []
        for row in probs:
            top = np.argsort(-row)[: self.topk]
            zeroshot.append([[self.id2label[int(i)], float(row[int(i)])] for i in top])
        speech = probs[:, self.speech_ids].max(axis=1).astype(np.float32) \
            if self.speech_ids else np.zeros(len(waves), dtype=np.float32)
        return emb, zeroshot, speech


def make_backend(name: str, device: str | None, topk: int):
    if name == "mel":
        return MelBackend()
    if name == "ast":
        try:
            return ASTBackend(device=device, topk=topk)
        except ImportError as e:
            raise SystemExit(
                f"backend ast indisponible ({e}).\n"
                "  pip install torch transformers\n"
                "ou utiliser --backend mel (dégradé : pas de zéro-shot)."
            )
    raise SystemExit(f"backend inconnu : {name}")


# ---------------------------------------------------------------------------

def load_clip(path: str, sr_expected: int) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != sr_expected:
        raise SystemExit(f"{path}: samplerate {sr} != {sr_expected}")
    return data[:, 0]


def embed_session(session_id: str, backend, args) -> dict:
    out_path = os.path.join(args.emb_dir, f"{session_id}.npz")
    if os.path.isfile(out_path) and not args.force:
        print(f"  {session_id}: embeddings déjà présents (--force pour refaire)")
        return {"session_id": session_id, "skipped": True}

    events = sc.read_events(session_id, args.events_dir)
    xs, ids, zeroshot, speech = [], [], [], []
    missing = 0
    for lo in range(0, len(events), args.batch_size):
        batch = events[lo : lo + args.batch_size]
        waves, kept = [], []
        for ev in batch:
            path = sc.clip_abspath(ev, args.events_dir)
            if not os.path.isfile(path):
                missing += 1
                continue
            waves.append(load_clip(path, sc.SAMPLE_RATE))
            kept.append(ev)
        if not waves:
            continue
        X, zs, sp = backend.embed_batch(waves, sc.SAMPLE_RATE)
        xs.append(X)
        ids.extend(ev["full_id"] for ev in kept)
        zeroshot.extend(zs)
        speech.extend(sp.tolist())

    X = np.concatenate(xs, axis=0) if xs else np.zeros((0, 1), dtype=np.float32)
    speech = np.asarray(speech, dtype=np.float32)

    # --drop-speech : écarter les événements de parole et supprimer leur audio
    n_dropped = 0
    if args.drop_speech:
        if backend.name == "mel":
            print("  attention : --drop-speech inopérant avec le backend mel "
                  "(pas de détecteur de parole)")
        else:
            keep = speech < args.speech_threshold
            n_dropped = int((~keep).sum())
            for fid, is_kept in zip(ids, keep):
                if not is_kept:
                    ev_id = fid.split("/", 1)[1]
                    clip = os.path.join(args.events_dir, session_id, "clips",
                                        f"{ev_id}.flac")
                    if os.path.isfile(clip):
                        os.unlink(clip)
            X = X[keep]
            speech = speech[keep]
            ids = [fid for fid, k in zip(ids, keep) if k]
            zeroshot = [z for z, k in zip(zeroshot, keep) if k]

    config = dict(backend.config)
    config.update({"n_events": len(ids), "n_missing_clips": missing,
                   "n_dropped_speech": n_dropped,
                   "drop_speech": bool(args.drop_speech),
                   "privacy": args.privacy})
    sc.ensure_dir(args.emb_dir)
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        event_ids=np.asarray(ids),
        zeroshot_json=json.dumps(zeroshot),
        speech_score=speech,
        config_json=json.dumps(config),
    )

    # --privacy embeddings-only : plus aucun audio, il reste embeddings + temps
    n_purged = 0
    if args.privacy == "embeddings-only":
        clips_dir = os.path.join(args.events_dir, session_id, "clips")
        if os.path.isdir(clips_dir):
            for f in os.listdir(clips_dir):
                os.unlink(os.path.join(clips_dir, f))
                n_purged += 1

    top_note = ""
    if zeroshot and zeroshot[0]:
        counts: dict[str, int] = {}
        for z in zeroshot:
            if z:
                counts[z[0][0]] = counts.get(z[0][0], 0) + 1
        best = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        top_note = "  top zéro-shot : " + ", ".join(f"{k}×{v}" for k, v in best)
    print(f"  {session_id}: {len(ids)} embeddings ({config['dim']}-d)"
          + (f", {n_dropped} parole écartés" if n_dropped else "")
          + (f", {n_purged} clips supprimés (embeddings-only)" if n_purged else "")
          + (f", {missing} clips manquants" if missing else ""))
    if top_note:
        print(top_note)
    return {"session_id": session_id, "n": len(ids)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sessions", nargs="*", default=None)
    p.add_argument("--events-dir", default=sc.EVENTS_DIR)
    p.add_argument("--emb-dir", default=sc.EMB_DIR)
    p.add_argument("--backend", choices=("ast", "mel"), default="ast")
    p.add_argument("--device", default=None, help="mps / cuda / cpu (auto)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--drop-speech", action="store_true")
    p.add_argument("--speech-threshold", type=float, default=0.5)
    p.add_argument("--privacy", choices=("full", "embeddings-only"), default="full")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.sessions is None:
        args.sessions = sorted(
            d for d in os.listdir(args.events_dir)
            if os.path.isfile(sc.events_csv_path(d, args.events_dir))
        ) if os.path.isdir(args.events_dir) else []
    if not args.sessions:
        raise SystemExit(f"aucun events.csv dans {args.events_dir} — "
                         "lancer detect_events.py d'abord")

    backend = make_backend(args.backend, args.device, args.topk)
    print(f"backend {backend.name} ({backend.config.get('device', 'cpu')}) "
          f"sur {len(args.sessions)} session(s) :")
    for sid in args.sessions:
        embed_session(sid, backend, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
