"""Socle commun du projet « sound event detection » (SED) domestique.

Reprend les conventions du projet frère (reconnaissance acoustique de frappes
clavier, branche `claude/keystroke-acoustic-recognition-uzw4z5`) :
  - split strictement PAR SESSION avec garde-fou fatal (`check_disjoint`),
  - horloge monotone + checkpoints `[frame, t_mono]` pour situer chaque
    échantillon audio dans le temps,
  - `meta.json` par session, sessions listées depuis `data/raw/`.

Différences assumées : audio 16 kHz mono (attendu par les modèles AudioSet),
FLAC chunké (~10 min) au lieu d'un WAV unique, et horodatage UTC en plus de
l'horloge monotone (une nuit doit se lire en heures réelles).

Aucune dépendance lourde ici : numpy/soundfile sont importés paresseusement.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

SAMPLE_RATE = 16_000
CAPTURE_VERSION = 1

# Répertoires par défaut (tout est local et git-ignoré).
RAW_DIR = "data/raw"
EVENTS_DIR = "data/events"
EMB_DIR = "data/embeddings"
CLUSTERS_FILE = "data/clusters.json"
LABELS_FILE = "data/labels.json"
MODELS_DIR = "models"
RESULTS_DIR = "results"
SPLIT_FILE = "splits.json"

# Labels réservés (le vocabulaire est libre par ailleurs).
BACKGROUND_LABEL = "background"
SPEECH_LABEL = "speech"

# Labels usuels proposés dans l'UI (suggestions, pas une liste fermée).
COMMON_LABELS = (
    "ronflement", "respiration", "grincement", "porte", "stores", "vaisselle",
    "clavier", "pas", "chaise", "speech", "background",
)

# Classes AudioSet -> vocabulaire perso (baseline zéro-shot + propositions UI).
# Une classe absente d'ici est proposée telle quelle (en minuscules).
AUDIOSET_MAP = {
    "Snoring": "ronflement",
    "Snort": "ronflement",
    "Breathing": "respiration",
    "Gasp": "respiration",
    "Sigh": "respiration",
    "Cough": "toux",
    "Speech": "speech",
    "Conversation": "speech",
    "Narration, monologue": "speech",
    "Male speech, man speaking": "speech",
    "Female speech, woman speaking": "speech",
    "Creak": "grincement",
    "Squeak": "grincement",
    "Door": "porte",
    "Slam": "porte",
    "Knock": "porte",
    "Cupboard open or close": "porte",
    "Drawer open or close": "porte",
    "Dishes, pots, and pans": "vaisselle",
    "Cutlery, silverware": "vaisselle",
    "Glass": "vaisselle",
    "Typing": "clavier",
    "Computer keyboard": "clavier",
    "Typewriter": "clavier",
    "Walk, footsteps": "pas",
    "Silence": "background",
    "Inside, small room": "background",
    "White noise": "background",
    "Pink noise": "background",
}


def map_zeroshot(audioset_label: str) -> str:
    """Classe AudioSet -> proposition dans le vocabulaire perso."""
    return AUDIOSET_MAP.get(audioset_label, audioset_label.lower())

EVENT_CSV_FIELDS = (
    "event_id",
    "t_start_s",
    "t_end_s",
    "t_utc_start",
    "rms_db",
    "snr_db",
    "clip_path",
)


# ---------------------------------------------------------------------------
# Garde-fou méthodologique — copié du projet clavier (kkr_common.check_disjoint)
# ---------------------------------------------------------------------------

def check_disjoint(**splits: Iterable[str]) -> None:
    """Hard guard against the one mistake that invalidates the whole experiment."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(splits[a]) & set(splits[b])
            if overlap:
                raise SystemExit(
                    f"FATAL: sessions {sorted(overlap)} appear in both '{a}' and "
                    f"'{b}'. The split must be strictly per-session — overlapping "
                    "sessions would measure memorisation, not recognition."
                )


# ---------------------------------------------------------------------------
# Fichiers / JSON
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj) -> None:
    """Écriture atomique (tmp + rename) : un crash ne corrompt jamais le fichier."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@dataclass
class Session:
    session_id: str
    path: str
    kind: str  # "night" | "day"
    meta: dict = field(default_factory=dict)

    @property
    def meta_path(self) -> str:
        return os.path.join(self.path, "meta.json")

    @property
    def samplerate(self) -> int:
        return int(self.meta.get("samplerate", SAMPLE_RATE))

    @property
    def chunks(self) -> list[dict]:
        """[{file, start_frame, n_frames}, ...] triés par start_frame."""
        return sorted(self.meta.get("chunks", []), key=lambda c: c["start_frame"])

    @property
    def n_frames(self) -> int:
        cs = self.chunks
        return int(cs[-1]["start_frame"] + cs[-1]["n_frames"]) if cs else 0

    @property
    def duration_s(self) -> float:
        return self.n_frames / float(self.samplerate)

    def chunk_path(self, chunk: dict) -> str:
        return os.path.join(self.path, chunk["file"])

    # -- temps ---------------------------------------------------------------

    def frame_to_mono(self, frame: float) -> float:
        """Frame audio -> horloge monotone, via les checkpoints [frame, t_mono].

        Interpolation linéaire par morceaux ; extrapolation aux bords au débit
        nominal (même logique que kkr_common.mono_to_frame, sens inverse).
        """
        cps = self.meta.get("clock_checkpoints") or []
        sr = float(self.samplerate)
        if len(cps) >= 2:
            import numpy as np

            frames = np.asarray([c[0] for c in cps], dtype=float)
            monos = np.asarray([c[1] for c in cps], dtype=float)
            if frame <= frames[0]:
                return float(monos[0] + (frame - frames[0]) / sr)
            if frame >= frames[-1]:
                return float(monos[-1] + (frame - frames[-1]) / sr)
            return float(np.interp(frame, frames, monos))
        t0 = self.meta.get("audio_start_mono")
        if t0 is None:
            raise ValueError(f"{self.session_id}: pas d'ancre temporelle en meta.json")
        return float(t0) + frame / sr

    def frame_to_utc(self, frame: float) -> float:
        """Frame audio -> epoch UTC (secondes), via l'ancre (t_utc, t_mono)."""
        utc0 = self.meta.get("t_utc_start_epoch")
        mono0 = self.meta.get("audio_start_mono")
        if utc0 is None or mono0 is None:
            raise ValueError(f"{self.session_id}: ancre UTC absente de meta.json")
        return float(utc0) + (self.frame_to_mono(frame) - float(mono0))

    def seconds_to_frame(self, t_s: float) -> int:
        return int(round(t_s * self.samplerate))

    # -- audio ---------------------------------------------------------------

    def read_span(self, start_frame: int, end_frame: int):
        """Lit [start_frame, end_frame) en float32 mono à travers les chunks."""
        import numpy as np
        import soundfile as sf

        start_frame = max(0, int(start_frame))
        end_frame = min(self.n_frames, int(end_frame))
        if end_frame <= start_frame:
            return np.zeros(0, dtype=np.float32)
        out = np.zeros(end_frame - start_frame, dtype=np.float32)
        for chunk in self.chunks:
            c0 = int(chunk["start_frame"])
            c1 = c0 + int(chunk["n_frames"])
            lo, hi = max(start_frame, c0), min(end_frame, c1)
            if hi <= lo:
                continue
            with sf.SoundFile(self.chunk_path(chunk)) as f:
                f.seek(lo - c0)
                data = f.read(hi - lo, dtype="float32", always_2d=True)[:, 0]
            out[lo - start_frame : lo - start_frame + len(data)] = data
        return out


def list_sessions(raw_dir: str = RAW_DIR, only: Sequence[str] | None = None) -> list[Session]:
    """Une session = un sous-répertoire de raw_dir contenant meta.json."""
    sessions: list[Session] = []
    if os.path.isdir(raw_dir):
        for name in sorted(os.listdir(raw_dir)):
            path = os.path.join(raw_dir, name)
            meta_path = os.path.join(path, "meta.json")
            if not os.path.isdir(path) or not os.path.isfile(meta_path):
                continue
            meta = read_json(meta_path)
            if not meta.get("chunks"):
                # session d'un autre projet (ex. branche clavier : audio.wav
                # sans chunks) ou capture jamais démarrée — pas à nous
                continue
            sessions.append(
                Session(
                    session_id=meta.get("session_id", name),
                    path=path,
                    kind=meta.get("kind", "night"),
                    meta=meta,
                )
            )
    if only is not None:
        by_id = {s.session_id: s for s in sessions}
        missing = [sid for sid in only if sid not in by_id]
        if missing:
            raise FileNotFoundError(f"sessions introuvables dans {raw_dir}: {missing}")
        sessions = [by_id[sid] for sid in only]
    return sessions


def make_session_id(kind: str, tag: str = "") -> str:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return "_".join(filter(None, [kind, stamp, tag]))


# ---------------------------------------------------------------------------
# Événements détectés
# ---------------------------------------------------------------------------

def events_csv_path(session_id: str, events_dir: str = EVENTS_DIR) -> str:
    return os.path.join(events_dir, session_id, "events.csv")


def read_events(session_id: str, events_dir: str = EVENTS_DIR) -> list[dict]:
    """Lit events.csv d'une session ; ajoute full_id = '<session>/<event_id>'."""
    path = events_csv_path(session_id, events_dir)
    out: list[dict] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["t_start_s"] = float(row["t_start_s"])
            row["t_end_s"] = float(row["t_end_s"])
            row["rms_db"] = float(row["rms_db"])
            row["snr_db"] = float(row["snr_db"])
            row["session_id"] = session_id
            row["full_id"] = f"{session_id}/{row['event_id']}"
            out.append(row)
    out.sort(key=lambda r: r["t_start_s"])
    return out


def read_all_events(events_dir: str = EVENTS_DIR) -> list[dict]:
    out: list[dict] = []
    if os.path.isdir(events_dir):
        for sid in sorted(os.listdir(events_dir)):
            if os.path.isfile(events_csv_path(sid, events_dir)):
                out.extend(read_events(sid, events_dir))
    return out


def clip_abspath(event: dict, events_dir: str = EVENTS_DIR) -> str:
    return os.path.join(events_dir, event["session_id"], event["clip_path"])


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def load_embeddings(session_ids: Sequence[str] | None = None, emb_dir: str = EMB_DIR):
    """Concatène les .npz par session -> (X, full_ids, zeroshot, speech, config).

    zeroshot : liste (par événement) de listes [label, score] triées ; peut être
    vide (backend `mel`). config : dict du backend qui a produit les embeddings.
    """
    import numpy as np

    if session_ids is None:
        session_ids = [
            f[: -len(".npz")]
            for f in sorted(os.listdir(emb_dir))
            if f.endswith(".npz")
        ] if os.path.isdir(emb_dir) else []
    xs, ids, zeroshot, speech = [], [], [], []
    config = None
    for sid in session_ids:
        path = os.path.join(emb_dir, f"{sid}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"embeddings manquants pour la session {sid} ({path}) — lancer embed.py"
            )
        with np.load(path, allow_pickle=False) as z:
            xs.append(z["X"].astype(np.float32))
            ids.extend(str(e) for e in z["event_ids"])
            zeroshot.extend(json.loads(str(z["zeroshot_json"])))
            speech.extend(z["speech_score"].tolist())
            cfg = json.loads(str(z["config_json"]))
        if config is None:
            config = cfg
        elif config.get("backend") != cfg.get("backend"):
            raise SystemExit(
                f"FATAL: backends d'embedding mélangés ({config.get('backend')} vs "
                f"{cfg.get('backend')} pour {sid}) — ré-embarquer toutes les sessions "
                "avec le même backend."
            )
    X = np.concatenate(xs, axis=0) if xs else np.zeros((0, 0), dtype=np.float32)
    return X, ids, zeroshot, np.asarray(speech, dtype=np.float32), (config or {})


# ---------------------------------------------------------------------------
# Store de labels (utilisé par label_ui, le simulateur, train et eval)
# ---------------------------------------------------------------------------

PROVENANCES = ("manual", "batch", "propagated")


class LabelStore:
    """data/labels.json : {full_id: {label, provenance, ts, source_event}}.

    Incrémental (chaque action réécrit le fichier de façon atomique) et
    toujours corrigible : re-labelliser écrase, `remove` révoque.
    """

    def __init__(self, path: str = LABELS_FILE):
        self.path = path
        self.labels: dict[str, dict] = {}
        if os.path.isfile(path):
            self.labels = read_json(path)

    def save(self) -> None:
        write_json(self.path, self.labels)

    def set(self, full_id: str, label: str, provenance: str = "manual",
            source_event: str | None = None) -> None:
        if provenance not in PROVENANCES:
            raise ValueError(f"provenance inconnue: {provenance}")
        self.labels[full_id] = {
            "label": label,
            "provenance": provenance,
            "ts": time.time(),
            "source_event": source_event,
        }

    def remove(self, full_id: str) -> None:
        self.labels.pop(full_id, None)

    def get(self, full_id: str) -> dict | None:
        return self.labels.get(full_id)

    def by_provenance(self, *provenances: str) -> dict[str, dict]:
        return {k: v for k, v in self.labels.items() if v["provenance"] in provenances}

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self.labels.values():
            out[v["label"]] = out.get(v["label"], 0) + 1
        return out


def propagate_knn(X, full_ids: Sequence[str], store: LabelStore,
                  seed_ids: Sequence[str], label: str,
                  k: int = 10, min_cos: float = 0.85) -> list[str]:
    """Propage `label` des événements `seed_ids` vers leurs voisins non étiquetés.

    kNN en similarité cosinus dans l'espace d'embedding ; n'écrase jamais un
    label manual/batch existant. Retourne les full_ids nouvellement propagés.
    """
    import numpy as np

    idx = {fid: i for i, fid in enumerate(full_ids)}
    seeds = [s for s in seed_ids if s in idx]
    if not seeds or len(full_ids) < 2:
        return []
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    added: list[str] = []
    for seed in seeds:
        sims = Xn @ Xn[idx[seed]]
        order = np.argsort(-sims)[: k + 1]
        for j in order:
            fid = full_ids[int(j)]
            if fid == seed or sims[int(j)] < min_cos:
                continue
            existing = store.get(fid)
            if existing is not None and existing["provenance"] != "propagated":
                continue  # un label humain ne se fait jamais écraser
            if existing is not None and existing["label"] == label:
                continue
            store.set(fid, label, provenance="propagated", source_event=seed)
            added.append(fid)
    return added


# ---------------------------------------------------------------------------
# Prédictions par événement (night_report, predict_events)
# ---------------------------------------------------------------------------

def predict_labels(session_ids: Sequence[str],
                   checkpoint: str = os.path.join(MODELS_DIR, "sed_clf.joblib"),
                   labels_file: str = LABELS_FILE,
                   emb_dir: str = EMB_DIR) -> tuple[dict[str, str], str]:
    """full_id -> label prédit pour ces sessions, avec la meilleure source dispo.

    Priorité : label humain (manual/batch) > modèle entraîné > zéro-shot mappé
    > 'evenement'. Retourne aussi une description de la source dominante.
    """
    X, ids, zeroshot, _speech, config = load_embeddings(session_ids, emb_dir)
    preds: dict[str, str] = {}
    source = "zéro-shot"
    model_ok = False
    if os.path.isfile(checkpoint):
        import joblib

        ckpt = joblib.load(checkpoint)
        if ckpt.get("embedding_backend") == config.get("backend"):
            y = ckpt["clf"].predict(X)
            preds = dict(zip(ids, y.tolist()))
            source = f"modèle adapté ({ckpt.get('model_kind')})"
            model_ok = True
    if not model_ok:
        for fid, zs in zip(ids, zeroshot):
            preds[fid] = map_zeroshot(zs[0][0]) if zs else "evenement"
        if not any(zeroshot):
            source = "aucun modèle ni zéro-shot (événements bruts)"
    store = LabelStore(labels_file)
    n_human = 0
    for fid in ids:
        rec = store.get(fid)
        if rec is not None and rec["provenance"] in ("manual", "batch"):
            preds[fid] = rec["label"]
            n_human += 1
    if n_human:
        source += f" + {n_human} labels humains"
    return preds, source


# ---------------------------------------------------------------------------
# Divers
# ---------------------------------------------------------------------------

def monotonic() -> float:
    return time.monotonic()


def lan_ip() -> str:
    """IP locale pour ouvrir l'UI depuis l'iPhone (pattern app.py clavier)."""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return "127.0.0.1"
