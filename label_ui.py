#!/usr/bin/env python3
"""App web locale d'étiquetage assisté (pattern app.py du projet clavier).

Serveur HTTP stdlib (ThreadingHTTPServer) + page HTML autonome inline,
ouvrable depuis l'iPhone sur le même Wi-Fi (--host 0.0.0.0, l'IP LAN est
affichée au lancement). Aucune donnée ne quitte la machine.

Principes (le cœur du projet, §3 du cahier des charges) :
  - le système PROPOSE des étiquettes tout seul (zéro-shot AudioSet mappé
    vers le vocabulaire perso, ou label déjà propagé) ;
  - ✅ valider / ✏️ corriger / ⏭️ passer, y compris un CLUSTER ENTIER en un
    geste ;
  - apprentissage actif : les clusters non étiquetés d'abord (gros gains),
    puis les événements incertains (marge du classifieur si un modèle
    existe, sinon confiance zéro-shot faible), diversifiés par round-robin
    entre clusters ;
  - propagation kNN des labels aux voisins d'embedding (provenance
    « propagated », toujours révocable) ;
  - data/labels.json incrémental, corrigible à tout moment ;
  - utile avec zéro étiquette comme avec des centaines.

Écoute du clip (FLAC natif navigateur) + spectrogramme log-mel par clip.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import sed_common as sc


# ---------------------------------------------------------------------------
# État applicatif
# ---------------------------------------------------------------------------

class LabelApp:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.store = sc.LabelStore(args.labels_file)
        self.skipped: set[str] = set()          # full_ids ou "cluster:<id>"
        self.spec_cache = sc.ensure_dir(os.path.join("data", "cache", "specs"))

        events = sc.read_all_events(args.events_dir)
        if not events:
            raise SystemExit(f"aucun événement dans {args.events_dir} — "
                             "lancer detect_events.py d'abord")
        self.events = {e["full_id"]: e for e in events}

        try:
            X, ids, zeroshot, speech, cfg = sc.load_embeddings(emb_dir=args.emb_dir)
        except (FileNotFoundError, SystemExit):
            X, ids, zeroshot, speech, cfg = (np.zeros((0, 1), np.float32), [], [],
                                             np.zeros(0), {})
            print("note : pas d'embeddings — propositions et propagation limitées")
        self.X = X
        self.ids = list(ids)
        self.idx = {fid: i for i, fid in enumerate(self.ids)}
        self.zeroshot = {fid: zs for fid, zs in zip(self.ids, zeroshot)}
        self.emb_config = cfg

        self.clusters: list[dict] = []
        self.assignments: dict[str, int] = {}
        if os.path.isfile(args.clusters_file):
            cj = sc.read_json(args.clusters_file)
            self.clusters = cj.get("clusters", [])
            self.assignments = cj.get("assignments", {})
        else:
            print("note : pas de clusters.json — validation par lot indisponible "
                  "(lancer cluster.py)")

        self.margins = self._load_model_margins()

    # -- modèle éventuel pour l'apprentissage actif -------------------------
    def _load_model_margins(self) -> dict[str, float]:
        path = os.path.join(sc.MODELS_DIR, "sed_clf.joblib")
        if not os.path.isfile(path) or len(self.ids) == 0:
            return {}
        try:
            import joblib

            ckpt = joblib.load(path)
            clf = ckpt["clf"]
            if ckpt.get("embedding_backend") != self.emb_config.get("backend"):
                print("note : modèle entraîné sur un autre backend d'embedding — "
                      "marges ignorées")
                return {}
            proba = clf.predict_proba(self.X)
            part = np.sort(proba, axis=1)
            margin = part[:, -1] - part[:, -2] if proba.shape[1] > 1 else part[:, -1]
            print(f"apprentissage actif : marges du modèle {path} utilisées")
            return {fid: float(m) for fid, m in zip(self.ids, margin)}
        except Exception as e:  # le modèle est optionnel, ne jamais bloquer l'UI
            print(f"note : modèle ignoré ({e})")
            return {}

    # -- propositions -------------------------------------------------------
    def proposal(self, fid: str) -> tuple[str, str]:
        """-> (label proposé, origine de la proposition)."""
        rec = self.store.get(fid)
        if rec is not None:
            return rec["label"], f"label {rec['provenance']}"
        zs = self.zeroshot.get(fid) or []
        if zs:
            return sc.map_zeroshot(zs[0][0]), f"zéro-shot {zs[0][0]} ({zs[0][1]:.2f})"
        cid = self.assignments.get(fid)
        for c in self.clusters:
            if c["cluster_id"] == cid and c["zeroshot_label"] != "?":
                return sc.map_zeroshot(c["zeroshot_label"]), "zéro-shot du cluster"
        return "?", "aucune proposition (backend mel)"

    def cluster_proposal(self, c: dict) -> str:
        # majorité des labels existants du cluster, sinon zéro-shot dominant
        counts: dict[str, int] = {}
        for fid in self.members(c["cluster_id"]):
            rec = self.store.get(fid)
            if rec:
                counts[rec["label"]] = counts.get(rec["label"], 0) + 1
        if counts:
            return max(counts.items(), key=lambda kv: kv[1])[0]
        if c["zeroshot_label"] != "?":
            return sc.map_zeroshot(c["zeroshot_label"])
        return "?"

    def members(self, cluster_id: int) -> list[str]:
        return [fid for fid, cid in self.assignments.items() if cid == cluster_id]

    # -- file d'apprentissage actif -----------------------------------------
    def build_queue(self, n: int) -> list[dict]:
        items: list[dict] = []
        # 1) clusters entiers non (ou peu) étiquetés, du plus gros au plus petit
        for c in self.clusters:
            if c["is_noise"] or f"cluster:{c['cluster_id']}" in self.skipped:
                continue
            members = self.members(c["cluster_id"])
            labeled = sum(1 for m in members
                          if (self.store.get(m) or {}).get("provenance")
                          in ("manual", "batch"))
            if members and labeled / len(members) < 0.5:
                items.append(self._cluster_item(c, members))
        items.sort(key=lambda it: -it["size"])

        # 2) événements incertains, round-robin entre clusters (diversité)
        candidates: dict[int, list[tuple[float, str]]] = {}
        for fid in self.events:
            rec = self.store.get(fid)
            if rec is not None and rec["provenance"] != "propagated":
                continue
            if fid in self.skipped:
                continue
            if self.margins:
                unc = self.margins.get(fid, 0.0)          # petite marge d'abord
            else:
                zs = self.zeroshot.get(fid) or []
                unc = zs[0][1] if zs else 0.0             # faible confiance d'abord
            cid = self.assignments.get(fid, -1)
            candidates.setdefault(cid, []).append((unc, fid))
        for lst in candidates.values():
            lst.sort()
        rr = sorted(candidates)
        while len(items) < n and any(candidates.get(c) for c in rr):
            for cid in rr:
                if candidates.get(cid):
                    _, fid = candidates[cid].pop(0)
                    items.append(self._event_item(fid))
                    if len(items) >= n:
                        break
        return items[:n]

    def _cluster_item(self, c: dict, members: list[str]) -> dict:
        return {
            "type": "cluster",
            "cluster_id": c["cluster_id"],
            "size": len(members),
            "proposal": self.cluster_proposal(c),
            "origin": f"zéro-shot dominant : {c['zeroshot_label']}",
            "examples": [self._event_item(fid) for fid in c["examples"]
                         if fid in self.events][: self.args.examples],
        }

    def _event_item(self, fid: str) -> dict:
        ev = self.events[fid]
        proposal, origin = self.proposal(fid)
        rec = self.store.get(fid)
        return {
            "type": "event",
            "full_id": fid,
            "session_id": ev["session_id"],
            "t_start_s": ev["t_start_s"],
            "duration_s": round(ev["t_end_s"] - ev["t_start_s"], 2),
            "t_utc": ev["t_utc_start"],
            "snr_db": ev["snr_db"],
            "cluster_id": self.assignments.get(fid, -1),
            "proposal": proposal,
            "origin": origin,
            "current": rec,
            "has_audio": os.path.isfile(sc.clip_abspath(ev, self.args.events_dir)),
        }

    # -- actions ------------------------------------------------------------
    def apply(self, payload: dict) -> dict:
        action = payload.get("action")
        label = (payload.get("label") or "").strip()
        with self.lock:
            if action == "skip":
                for key in self._target_keys(payload):
                    self.skipped.add(key)
                return {"ok": True, "skipped": True}
            if action == "undo":
                removed = self._undo(payload)
                self.store.save()
                return {"ok": True, "n_removed": removed}
            if action in ("confirm", "correct"):
                if action == "confirm":
                    label = payload.get("proposal") or label
                if not label or label == "?":
                    return {"ok": False, "error": "label vide"}
                n_labeled, seeds = self._label_target(payload, label)
                n_prop = 0
                if not payload.get("no_propagate") and len(self.ids) > 1 and seeds:
                    n_prop = len(sc.propagate_knn(
                        self.X, self.ids, self.store, seeds[: 20], label,
                        k=self.args.knn, min_cos=self.args.min_cos,
                    ))
                self.store.save()
                return {"ok": True, "n_labeled": n_labeled, "n_propagated": n_prop,
                        "label": label}
        return {"ok": False, "error": f"action inconnue : {action}"}

    def _target_keys(self, payload: dict) -> list[str]:
        if payload.get("cluster_id") is not None:
            return [f"cluster:{payload['cluster_id']}"]
        return list(payload.get("event_ids") or [])

    def _label_target(self, payload: dict, label: str) -> tuple[int, list[str]]:
        if payload.get("cluster_id") is not None:
            members = self.members(int(payload["cluster_id"]))
            for fid in members:
                rec = self.store.get(fid)
                if rec is None or rec["provenance"] != "manual":
                    self.store.set(fid, label, provenance="batch")
            return len(members), members
        fids = [f for f in (payload.get("event_ids") or []) if f in self.events]
        for fid in fids:
            self.store.set(fid, label, provenance="manual")
        return len(fids), fids

    def _undo(self, payload: dict) -> int:
        targets = (self.members(int(payload["cluster_id"]))
                   if payload.get("cluster_id") is not None
                   else list(payload.get("event_ids") or []))
        removed = 0
        for fid in targets:
            if self.store.get(fid) is not None:
                self.store.remove(fid)
                removed += 1
        # révoque aussi les labels propagés depuis ces événements
        for fid, rec in list(self.store.labels.items()):
            if rec["provenance"] == "propagated" and rec.get("source_event") in targets:
                self.store.remove(fid)
                removed += 1
        return removed

    # -- état global --------------------------------------------------------
    def state(self) -> dict:
        prov = {p: 0 for p in sc.PROVENANCES}
        for rec in self.store.labels.values():
            prov[rec["provenance"]] += 1
        return {
            "n_events": len(self.events),
            "n_labeled": len(self.store.labels),
            "by_label": self.store.counts(),
            "by_provenance": prov,
            "n_clusters": len([c for c in self.clusters if not c["is_noise"]]),
            "backend": self.emb_config.get("backend", "aucun"),
            "active_learning": "marges du modèle" if self.margins else "zéro-shot",
            "suggestions": sorted(set(list(sc.COMMON_LABELS)
                                      + list(self.store.counts()))),
        }

    def clusters_view(self) -> list[dict]:
        out = []
        for c in self.clusters:
            members = self.members(c["cluster_id"])
            labels: dict[str, int] = {}
            for m in members:
                rec = self.store.get(m)
                if rec:
                    labels[rec["label"]] = labels.get(rec["label"], 0) + 1
            out.append({
                "cluster_id": c["cluster_id"], "size": len(members),
                "is_noise": c["is_noise"],
                "proposal": self.cluster_proposal(c),
                "labels": labels,
            })
        return out

    # -- média --------------------------------------------------------------
    def audio_bytes(self, fid: str) -> bytes | None:
        ev = self.events.get(fid)
        if ev is None:
            return None
        path = sc.clip_abspath(ev, self.args.events_dir)
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def spectrogram_png(self, fid: str) -> bytes | None:
        ev = self.events.get(fid)
        if ev is None:
            return None
        cache = os.path.join(self.spec_cache, fid.replace("/", "__") + ".png")
        if os.path.isfile(cache):
            with open(cache, "rb") as f:
                return f.read()
        path = sc.clip_abspath(ev, self.args.events_dir)
        if not os.path.isfile(path):
            return None
        import librosa
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import soundfile as sf

        y, sr = sf.read(path, dtype="float32", always_2d=True)
        y = y[:, 0]
        m = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024,
                                           hop_length=256, fmin=50)
        logm = librosa.power_to_db(m + 1e-10)
        fig, ax = plt.subplots(figsize=(6.0, 2.0), dpi=110)
        ax.imshow(logm, origin="lower", aspect="auto", cmap="magma",
                  interpolation="nearest")
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True)
        plt.close(fig)
        data = buf.getvalue()
        with open(cache, "wb") as f:
            f.write(data)
        return data


# ---------------------------------------------------------------------------
# Page (autonome : HTML + CSS + JS inline, mobile-friendly, clair/sombre)
# ---------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Étiquetage des sons</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e;
  --muted: #898781; --line: #e1e0d9; --accent: #2a78d6; --good: #0ca30c;
  --warn: #d03b3b; --chip: #edf3fc;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d0d0d; --surface: #1a1a19; --ink: #fff; --ink2: #c3c2b7;
          --muted: #898781; --line: #2c2c2a; --accent: #3987e5; --chip: #1e2a3a; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
       font: 16px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 760px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.15rem; margin: 4px 0 12px; }
.stats { display: flex; flex-wrap: wrap; gap: 8px 16px; color: var(--ink2);
         font-size: .85rem; margin-bottom: 14px; }
.stats b { color: var(--ink); }
.card { background: var(--surface); border: 1px solid var(--line);
        border-radius: 12px; padding: 16px; margin-bottom: 14px; }
.kind { font-size: .75rem; text-transform: uppercase; letter-spacing: .06em;
        color: var(--muted); }
.proposal { font-size: 1.5rem; font-weight: 700; margin: 6px 0 2px; }
.origin { color: var(--ink2); font-size: .85rem; margin-bottom: 10px; }
.meta { color: var(--muted); font-size: .8rem; margin-bottom: 10px; }
img.spec { width: 100%; height: 110px; object-fit: fill; border-radius: 8px;
           background: #1a1a19; display: block; }
audio { width: 100%; margin: 8px 0; height: 36px; }
.examples { display: grid; gap: 10px; margin: 10px 0; }
.ex { border: 1px solid var(--line); border-radius: 10px; padding: 8px; }
.ex .meta { margin: 0 0 6px; }
.actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
button { border: 1px solid var(--line); background: var(--surface);
         color: var(--ink); border-radius: 10px; padding: 10px 14px;
         font-size: .95rem; cursor: pointer; }
button.ok { background: var(--good); border-color: var(--good); color: #fff; }
button.warn { color: var(--warn); }
input[type=text] { flex: 1 1 140px; min-width: 120px; border: 1px solid var(--line);
  background: var(--bg); color: var(--ink); border-radius: 10px;
  padding: 10px 12px; font-size: .95rem; }
.note { color: var(--muted); font-size: .8rem; margin-top: 8px; min-height: 1.2em; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 2px; }
.chip { background: var(--chip); color: var(--accent); border-radius: 999px;
        padding: 2px 10px; font-size: .8rem; cursor: pointer; }
table { width: 100%; border-collapse: collapse; font-size: .85rem; }
th, td { text-align: left; padding: 6px 8px; border-top: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; border-top: none; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.done { text-align: center; color: var(--ink2); padding: 30px 0; }
summary { cursor: pointer; color: var(--ink2); }
</style>
</head>
<body>
<div class="wrap">
  <h1>🎧 Étiquetage des sons</h1>
  <div class="stats" id="stats">chargement…</div>
  <div id="item"></div>
  <div class="card">
    <details>
      <summary>Aperçu des clusters</summary>
      <table id="clusters"><thead>
        <tr><th>cluster</th><th class="num">taille</th><th>proposition</th>
            <th>labels posés</th></tr>
      </thead><tbody></tbody></table>
    </details>
  </div>
</div>
<script>
const esc = s => String(s ?? "").replace(/[&<>"]/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let queue = [], state = {}, lastAction = null;

async function getJSON(url) { const r = await fetch(url); return r.json(); }
async function refresh() {
  state = await getJSON("/api/state");
  const s = document.getElementById("stats");
  const prov = state.by_provenance || {};
  s.innerHTML =
    `<span><b>${state.n_labeled}</b>/${state.n_events} étiquetés</span>` +
    `<span>manuel <b>${prov.manual||0}</b> · lot <b>${prov.batch||0}</b> · ` +
    `propagé <b>${prov.propagated||0}</b></span>` +
    `<span>${state.n_clusters} clusters · backend ${esc(state.backend)}</span>` +
    `<span>priorité : ${esc(state.active_learning)}</span>`;
  renderClusters(await getJSON("/api/clusters"));
  if (!queue.length) queue = await getJSON("/api/queue?n=30");
  renderItem();
}
function player(ev) {
  const audio = ev.has_audio
    ? `<audio controls preload="none" src="/audio/${ev.full_id}"></audio>`
    : `<div class="note">audio supprimé (mode privacy) — spectrogramme seul</div>`;
  return `<img class="spec" loading="lazy" src="/spec/${ev.full_id}.png"
           alt="spectrogramme">${audio}`;
}
function evMeta(ev) {
  return `${esc(ev.session_id)} · t=${ev.t_start_s.toFixed(1)}s · ` +
         `${ev.duration_s}s · SNR ${ev.snr_db} dB · cluster ${ev.cluster_id}`;
}
function renderItem() {
  const el = document.getElementById("item");
  const it = queue[0];
  if (!it) {
    el.innerHTML = `<div class="card done">🎉 Rien d'urgent à étiqueter.<br>
      <span class="note">Relance cluster.py / train_events.py pour raffiner,
      ou recharge la page.</span></div>`;
    return;
  }
  const chips = (state.suggestions || []).map(l =>
    `<span class="chip" onclick="act('correct','${esc(l)}')">${esc(l)}</span>`
  ).join("");
  let body, meta = "";
  if (it.type === "cluster") {
    body = `<div class="examples">` + it.examples.map(ex =>
      `<div class="ex"><div class="meta">${evMeta(ex)}</div>${player(ex)}</div>`
    ).join("") + `</div>`;
    meta = `${it.size} événements — la validation s'applique à TOUT le cluster`;
  } else {
    body = player(it);
    meta = evMeta(it) + (it.current
      ? ` · actuellement « ${esc(it.current.label)} » (${it.current.provenance})`
      : "");
  }
  el.innerHTML = `<div class="card">
    <div class="kind">${it.type === "cluster"
      ? "Cluster #" + it.cluster_id : "Événement"}</div>
    <div class="proposal">${esc(it.proposal)}</div>
    <div class="origin">${esc(it.origin)}</div>
    <div class="meta">${meta}</div>
    ${body}
    <div class="chips">${chips}</div>
    <div class="actions">
      <button class="ok" onclick="act('confirm')">✅ Valider${
        it.type === "cluster" ? " tout" : ""}</button>
      <input type="text" id="correction" placeholder="corriger en…"
             list="labels" onkeydown="if(event.key==='Enter')actCorrect()">
      <button onclick="actCorrect()">✏️ Corriger</button>
      <button onclick="act('skip')">⏭️ Passer</button>
      <button class="warn" onclick="act('undo')">↩︎ Retirer le label</button>
    </div>
    <datalist id="labels">${(state.suggestions || []).map(l =>
      `<option value="${esc(l)}">`).join("")}</datalist>
    <div class="note" id="feedback"></div>
  </div>`;
}
function target(it) {
  return it.type === "cluster" ? { cluster_id: it.cluster_id }
                               : { event_ids: [it.full_id] };
}
async function act(action, label) {
  const it = queue[0];
  if (!it) return;
  const payload = Object.assign({ action, label, proposal: it.proposal }, target(it));
  const r = await fetch("/api/label", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload) });
  const res = await r.json();
  const fb = document.getElementById("feedback");
  if (!res.ok) { fb.textContent = "⚠️ " + (res.error || "erreur"); return; }
  if (action === "undo") { fb.textContent = `retiré (${res.n_removed})`; queue = []; }
  else if (action !== "skip") {
    fb.textContent = `« ${res.label} » posé sur ${res.n_labeled} événement(s)` +
      (res.n_propagated ? ` + ${res.n_propagated} propagés` : "");
    queue.shift();
  } else { queue.shift(); }
  setTimeout(refresh, 150);
}
function actCorrect() {
  const v = document.getElementById("correction").value.trim();
  if (v) act("correct", v);
}
function renderClusters(rows) {
  const tb = document.querySelector("#clusters tbody");
  tb.innerHTML = rows.map(c => `<tr>
    <td>${c.is_noise ? "divers" : "#" + c.cluster_id}</td>
    <td class="num">${c.size}</td>
    <td>${esc(c.proposal)}</td>
    <td>${esc(Object.entries(c.labels).map(([l, n]) => l + "×" + n)
              .join(", ") || "—")}</td></tr>`).join("");
}
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "v") act("confirm");
  if (e.key === "p") act("skip");
  if (e.key === "c") document.getElementById("correction")?.focus();
});
refresh();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Serveur HTTP (stdlib, comme app.py du projet clavier)
# ---------------------------------------------------------------------------

def make_handler(app: LabelApp):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence par requête
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):
            path = self.path.split("?")[0]
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            if path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/state":
                with app.lock:
                    self._json(app.state())
            elif path == "/api/clusters":
                with app.lock:
                    self._json(app.clusters_view())
            elif path == "/api/queue":
                n = 30
                for part in query.split("&"):
                    if part.startswith("n="):
                        try:
                            n = max(1, min(200, int(part[2:])))
                        except ValueError:
                            pass
                with app.lock:
                    self._json(app.build_queue(n))
            elif path.startswith("/audio/"):
                data = app.audio_bytes(path[len("/audio/"):])
                if data is None:
                    self.send_error(404)
                else:
                    self._send(200, data, "audio/flac")
            elif path.startswith("/spec/") and path.endswith(".png"):
                data = app.spectrogram_png(path[len("/spec/"):-len(".png")])
                if data is None:
                    self.send_error(404)
                else:
                    self._send(200, data, "image/png")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path.split("?")[0] != "/api/label":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json({"ok": False, "error": "JSON invalide"}, 400)
                return
            self._json(app.apply(payload))

    return Handler


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="0.0.0.0",
                   help="0.0.0.0 = accessible depuis l'iPhone sur le même Wi-Fi")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--events-dir", default=sc.EVENTS_DIR)
    p.add_argument("--emb-dir", default=sc.EMB_DIR)
    p.add_argument("--clusters-file", default=sc.CLUSTERS_FILE)
    p.add_argument("--labels-file", default=sc.LABELS_FILE)
    p.add_argument("--knn", type=int, default=10,
                   help="propagation : k voisins par événement étiqueté")
    p.add_argument("--min-cos", type=float, default=0.85,
                   help="propagation : similarité cosinus minimale")
    p.add_argument("--examples", type=int, default=4,
                   help="exemples audio montrés par cluster")
    args = p.parse_args()

    app = LabelApp(args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"UI d'étiquetage : http://localhost:{args.port}")
    if args.host == "0.0.0.0":
        print(f"  depuis l'iPhone : http://{sc.lan_ip()}:{args.port}")
    print("  (tout reste local ; Ctrl-C pour arrêter — labels déjà sauvés au fil de l'eau)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
