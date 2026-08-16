#!/usr/bin/env python3
"""app.py — live keystroke-sound recognition as a local web app.

Runs the trained model on the microphone + keyboard in real time (the same
engine as live.py) and serves a small web page that shows, live:

  - what you actually typed vs what the model "heard" (from sound alone),
  - a running top-1 / top-5 accuracy,
  - a rolling ticker of the last keystrokes with the model's top-5 guesses.

The page is served on your local network, so you can open it on the Mac AND on
your iPhone (same Wi-Fi) at the printed http://<mac-ip>:<port> address. The model
still runs on the Mac — this is the quick "put the demo on the phone screen"
bridge; a true on-device iPhone app is a separate Core ML build (see export
notes in the README).

    python app.py                       # real capture, needs a trained model
    python app.py --demo                # fake predictions, to preview the UI
    python app.py --checkpoint models/keycnn.pt --port 8000

Privacy: in real mode this listens to your microphone and logs your keystrokes
locally, exactly like live.py. Use it on your own machine.
"""

from __future__ import annotations

import argparse
import html
import json
import queue
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import kkr_common as kc


# --------------------------------------------------------------------------- #
# Event hub: fan out prediction events to every connected browser
# --------------------------------------------------------------------------- #


class Hub:
    def __init__(self):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.state = {
            "typed": "", "heard": "", "n": 0, "top1": 0.0, "top5": 0.0,
            "chance": 0.0, "recent": [], "info": {}, "running": True,
        }

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: dict):
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)


# --------------------------------------------------------------------------- #
# HTML page (self-contained, theme-aware, mobile-friendly)
# --------------------------------------------------------------------------- #

PAGE = """<!doctype html>
<html lang="fr"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Clavier au son</title>
<style>
  :root { color-scheme: light dark; --bg:#0e1116; --card:#171b22; --fg:#e6edf3;
          --muted:#8b949e; --ok:#3fb950; --bad:#f85149; --accent:#58a6ff; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:var(--bg); color:var(--fg); -webkit-font-smoothing:antialiased; }
  header { padding:14px 18px; display:flex; align-items:center; gap:10px;
           border-bottom:1px solid #222; position:sticky; top:0; background:var(--bg); }
  .dot { width:10px; height:10px; border-radius:50%; background:var(--bad); }
  .dot.on { background:var(--ok); box-shadow:0 0 8px var(--ok); }
  h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.2px; }
  .sub { color:var(--muted); font-size:12px; }
  main { padding:18px; max-width:900px; margin:0 auto; }
  .card { background:var(--card); border:1px solid #222; border-radius:12px; padding:16px; margin-bottom:14px; }
  .label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.8px; margin-bottom:6px; }
  .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:22px; line-height:1.5;
          word-break:break-all; white-space:pre-wrap; min-height:1.5em; }
  .heard { color:var(--accent); }
  .stats { display:flex; gap:12px; flex-wrap:wrap; }
  .stat { flex:1; min-width:110px; background:var(--card); border:1px solid #222; border-radius:12px; padding:14px; text-align:center; }
  .stat .v { font-size:30px; font-weight:700; }
  .stat .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.6px; margin-top:2px; }
  .ticker { display:flex; flex-direction:column-reverse; gap:4px; max-height:44vh; overflow:auto; }
  .row { display:flex; align-items:center; gap:10px; font-family:ui-monospace,Menlo,monospace; font-size:15px;
         padding:6px 8px; border-radius:8px; background:#10141a; }
  .row .you { font-weight:700; min-width:2.2em; }
  .row .arw { color:var(--muted); }
  .row .pred { min-width:2.2em; }
  .row .mk { margin-left:auto; }
  .ok { color:var(--ok); } .bad { color:var(--bad); } .na { color:var(--muted); }
  .top5 { color:var(--muted); letter-spacing:.15em; }
  footer { color:var(--muted); font-size:12px; text-align:center; padding:14px; }
</style></head><body>
<header>
  <span class="dot" id="dot"></span>
  <h1>Clavier au son <span class="sub" id="mode"></span></h1>
  <span class="sub" id="info" style="margin-left:auto"></span>
</header>
<main>
  <div class="stats">
    <div class="stat"><div class="v" id="top1">–</div><div class="k">top-1</div></div>
    <div class="stat"><div class="v" id="top5">–</div><div class="k">top-5</div></div>
    <div class="stat"><div class="v" id="n">0</div><div class="k">frappes</div></div>
    <div class="stat"><div class="v" id="chance">–</div><div class="k">hasard</div></div>
  </div>
  <div class="card">
    <div class="label">ce que le modèle entend</div>
    <div class="mono heard" id="heard"></div>
  </div>
  <div class="card">
    <div class="label">ce que tu tapes</div>
    <div class="mono" id="typed"></div>
  </div>
  <div class="card">
    <div class="label">frappes récentes (top-5 au son)</div>
    <div class="ticker" id="ticker"></div>
  </div>
</main>
<footer>modèle acoustique seul — la prédiction vient du son, pas de la touche. Ⓘ démo locale.</footer>
<script>
const G = s => s;
function pct(x){ return (100*x).toFixed(1)+'%'; }
function esc(s){ return (s||'').replace(/</g,'&lt;'); }
function apply(st){
  document.getElementById('top1').textContent = st.n ? pct(st.top1) : '–';
  document.getElementById('top5').textContent = st.n ? pct(st.top5) : '–';
  document.getElementById('n').textContent = st.n;
  document.getElementById('chance').textContent = st.chance ? pct(st.chance) : '–';
  document.getElementById('heard').textContent = st.heard;
  document.getElementById('typed').textContent = st.typed;
  document.getElementById('dot').className = 'dot' + (st.running ? ' on' : '');
  if (st.info && st.info.mode) document.getElementById('mode').textContent = '· ' + st.info.mode;
  if (st.info && st.info.line) document.getElementById('info').textContent = st.info.line;
  const t = document.getElementById('ticker'); t.innerHTML='';
  (st.recent||[]).forEach(r => {
    const div = document.createElement('div'); div.className='row';
    const mk = r.scored ? (r.ok?'<span class="mk ok">✓</span>':'<span class="mk bad">✗</span>')
                        : '<span class="mk na">·</span>';
    div.innerHTML = '<span class="you">'+esc(r.you)+'</span><span class="arw">→</span>'+
                    '<span class="pred">'+esc(r.pred)+'</span>'+
                    '<span class="top5">'+esc((r.top5||[]).join(' '))+'</span>'+mk;
    t.appendChild(div);
  });
}
const es = new EventSource('/stream');
es.onmessage = e => { try { apply(JSON.parse(e.data)); } catch(_){} };
es.onerror = () => { document.getElementById('dot').className='dot'; };
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #


def make_handler(hub: Hub):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence per-request logging
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q = hub.subscribe()
                try:
                    self._send_event(hub.snapshot())  # initial state
                    while True:
                        try:
                            ev = q.get(timeout=15)
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            continue
                        self._send_event(ev)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    hub.unsubscribe(q)
            else:
                self.send_error(404)

        def _send_event(self, obj: dict):
            data = "data: " + json.dumps(obj) + "\n\n"
            self.wfile.write(data.encode("utf-8"))
            self.wfile.flush()

    return Handler


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# --------------------------------------------------------------------------- #
# Producers: real capture, or a demo feed
# --------------------------------------------------------------------------- #


def glyph(key: str) -> str:
    return {"space": "␣", "enter": "⏎", "backspace": "⌫"}.get(key, key if len(key) == 1 else f"⟨{key}⟩")


def run_capture(hub: Hub, args):
    """Real microphone + keyboard producer (mirrors live.py's loop)."""
    from capture import KeyLogger
    from live import LiveEngine, Predictor

    predictor = Predictor(args.checkpoint, args.device)
    engine = LiveEngine(predictor.sr, args.buffer_s, _dev(args.audio_device), args.blocksize)
    logger = KeyLogger("app", "live")
    pre, post = predictor.window_frames()
    margin = int(round(args.margin_ms * 1e-3 * predictor.sr))
    vocabset = set(predictor.vocab)
    chance = 1.0 / len(predictor.vocab)

    hub.state.update({"chance": chance, "info": {
        "mode": f"modèle epoch {predictor.epoch}",
        "line": f"{len(predictor.vocab)} touches"}})

    engine.start()
    time.sleep(0.6)
    logger.start()

    typed, heard, recent = [], [], []
    n = c1 = c5 = 0
    processed = 0
    while True:
        downs = logger.keydown_snapshot()
        now = time.monotonic()
        while processed < len(downs):
            t, key = downs[processed]
            center = int(round(kc.mono_to_frame(t, engine.meta())))
            if not engine.has_through(center + post + margin):
                if now - t < (predictor.post_ms + args.margin_ms) * 1e-3 + 0.4:
                    break
                processed += 1
                continue
            processed += 1
            win = engine.read(center - pre, pre + post)
            if win is None:
                continue
            preds, _ = predictor.predict(win, k=5)
            pred = preds[0]
            typed.append(key)
            heard.append(pred)
            scored = key in vocabset
            if scored:
                n += 1
                c1 += pred == key
                c5 += key in preds
            row = {"you": glyph(key), "pred": glyph(pred),
                   "top5": [glyph(p) for p in preds], "ok": pred == key, "scored": scored}
            recent.append(row)
            recent[:] = recent[-40:]
            hub.state.update({
                "typed": "".join(glyph(k) for k in typed[-120:]),
                "heard": "".join(glyph(k) for k in heard[-120:]),
                "n": n, "top1": c1 / n if n else 0.0, "top5": c5 / n if n else 0.0,
                "recent": list(recent),
            })
            hub.publish(hub.snapshot())
        time.sleep(0.01)


def run_demo(hub: Hub, args):
    """Synthetic producer so the UI can be previewed with no mic/model."""
    import numpy as np

    alphabet = list("abcdefghijklmnopqrstuvwxyz") + ["space"]
    rng = np.random.default_rng(0)
    chance = 1.0 / len(alphabet)
    hub.state.update({"chance": chance, "info": {"mode": "DÉMO (fausses prédictions)", "line": "aperçu UI"}})
    typed, heard, recent = [], [], []
    n = c1 = c5 = 0
    i = 0
    while True:
        key = alphabet[int(rng.integers(len(alphabet)))]
        # fake a model that is right ~45% of the time, else picks a neighbour-ish key
        correct = rng.random() < 0.45
        if correct:
            preds = [key] + list(rng.choice(alphabet, 4, replace=False))
        else:
            preds = list(rng.choice(alphabet, 5, replace=False))
            if key in preds:
                preds.remove(key)
                preds.append(alphabet[int(rng.integers(len(alphabet)))])
        pred = preds[0]
        typed.append(key)
        heard.append(pred)
        n += 1
        c1 += pred == key
        c5 += key in preds
        recent.append({"you": glyph(key), "pred": glyph(pred), "top5": [glyph(p) for p in preds],
                       "ok": pred == key, "scored": True})
        recent[:] = recent[-40:]
        hub.state.update({
            "typed": "".join(glyph(k) for k in typed[-120:]),
            "heard": "".join(glyph(k) for k in heard[-120:]),
            "n": n, "top1": c1 / n, "top5": c5 / n, "recent": list(recent),
        })
        hub.publish(hub.snapshot())
        i += 1
        time.sleep(0.35)


def _dev(d):
    if d is not None and str(d).isdigit():
        return int(d)
    return d


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="models/keycnn.pt")
    p.add_argument("--demo", action="store_true", help="fake predictions to preview the UI (no mic/model)")
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from your phone on the same Wi-Fi")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default=None, help="torch device (cpu default)")
    p.add_argument("--audio-device", default=None)
    p.add_argument("--blocksize", type=int, default=512)
    p.add_argument("--buffer-s", type=float, default=6.0)
    p.add_argument("--margin-ms", type=float, default=60.0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.demo:
        try:
            import sounddevice  # noqa: F401
            import torch  # noqa: F401
            from pynput import keyboard  # noqa: F401
        except Exception as exc:  # pragma: no cover
            print(f"missing dependency: {exc}\ninstall with: pip install -r requirements.txt\n"
                  "(or preview the interface without a mic: python app.py --demo)", file=sys.stderr)
            return 2

    hub = Hub()
    producer = run_demo if args.demo else run_capture
    threading.Thread(target=producer, args=(hub, args), daemon=True).start()

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(hub))
    ip = lan_ip()
    print("\n" + "=" * 60)
    print("  Clavier au son — app web live" + ("  [DÉMO]" if args.demo else ""))
    print("=" * 60)
    print(f"  sur ce Mac    : http://localhost:{args.port}")
    if args.host == "0.0.0.0":
        print(f"  sur l'iPhone  : http://{ip}:{args.port}   (même Wi-Fi)")
    print("  Ctrl-C pour arrêter\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\narrêt.")
    finally:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
