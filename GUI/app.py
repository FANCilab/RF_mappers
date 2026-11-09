"""Figure browser for ./MOsPlots, published on the internet through a Cloudflare quick tunnel.

    python GUI/app.py               # load the figures, serve http://127.0.0.1:8765, open the tunnel
    python GUI/app.py --no-tunnel   # local only

Every MOsPlots figure (PNG and SVG) is read into memory once at start; requests are answered from memory, so
only files loaded at start can be served (restart to pick up new plots). The tunnel is outbound only (no open
port): cloudflared prints a random https://<name>.trycloudflare.com address, repeated here once it is up.
Anyone with that address can see the figures until the app stops (Ctrl+C stops server and tunnel).
cloudflared is taken from GUI/bin/cloudflared.exe, else from the PATH.
"""
import argparse
import json
import re
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

GUI = Path(__file__).parent
PROJECT = GUI.parent
PLOTS = PROJECT / "MOsPlots"
ONE_ROOTS = [PROJECT / "MOsRoot", PROJECT / "NM036Root"]  # settings and counts of the plotted runs
CONTENT_TYPES = {".png": "image/png", ".svg": "image/svg+xml"}
SIGNALS = {"rf_traces": "traces", "rf_deconv": "deconvolved"}

files = {}   # path relative to MOsPlots -> bytes
pages = {}   # "/" and "/api/figures" -> (bytes, content type)


def run_summary(session, fov_runs):
    """Preprocessing label and significant / good counts of the plotted run, from its meta.json files."""
    settings, significant, good = None, 0, 0
    for fov, (variant, revision) in fov_runs.items():
        for root in ONE_ROOTS:
            meta_path = root / session / "alf" / fov / variant / f"#{revision}#" / "_ss_averageRFRun.meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                settings = meta["settings"]
                significant += meta["counts"]["significant"]
                good += meta["counts"]["good"]
    if settings is None:
        return {}
    if settings["signal"] == "traces":
        label = f"traces, σ {settings['smoothing_sigma']}, HP {settings['high_pass_s']} s"
    else:
        label = f"deconvolved, σ {settings['deconv_smoothing_sigma']}"
    return {"preprocessing": label, "significant": significant, "good": good, "shifts": settings["n_shifts"]}


def load():
    """Read every figure into memory and build the figure index."""
    figures, fov_runs = [], {}
    for path in sorted(PLOTS.rglob("*")):
        if path.suffix not in CONTENT_TYPES:
            continue
        relative = path.relative_to(PLOTS)
        files[relative.as_posix()] = path.read_bytes()
        if path.suffix != ".png":
            continue
        subject, date, number, scope, variant, revision, *name = relative.with_suffix("").parts
        session = f"{subject}/{date}/{number}"
        svg = relative.with_suffix(".svg")
        figures.append({"session": session, "scope": scope, "signal": SIGNALS.get(variant, variant),
                        "revision": revision, "name": "/".join(name), "png": relative.as_posix(),
                        "svg": svg.as_posix() if (PLOTS / svg).exists() else None})
        if scope != "session":
            fov_runs.setdefault(session, {})[scope] = (variant, revision)
    sessions = [{"session": s, "fovs": sorted(runs), **run_summary(s, runs)} for s, runs in sorted(fov_runs.items())]
    pages["/api/figures"] = (json.dumps({"sessions": sessions, "figures": figures}).encode(), "application/json")
    pages["/"] = ((GUI / "index.html").read_bytes(), "text/html; charset=utf-8")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlsplit(self.path)
        path = unquote(url.path)
        if path in pages:
            body, content_type = pages[path]
            cache = "no-cache"
        elif path.startswith("/fig/") and path[5:] in files:  # only files loaded at start; no disk access
            body, content_type = files[path[5:]], CONTENT_TYPES[Path(path).suffix]
            cache = "public, max-age=86400"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        if url.query == "download":
            self.send_header("Content-Disposition", f'attachment; filename="{Path(path).name}"')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start_tunnel(port):
    """Start a Cloudflare quick tunnel to the local server; print its public address when it appears."""
    exe = GUI / "bin" / "cloudflared.exe"
    exe = str(exe) if exe.exists() else shutil.which("cloudflared")
    if exe is None:
        print("cloudflared not found (GUI/bin/cloudflared.exe or PATH); serving locally only")
        return None
    # HTTP/2 over TCP: networks that block outbound UDP (QUIC) otherwise keep the tunnel from connecting
    process = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", f"http://127.0.0.1:{port}"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    def watch():  # keeps reading cloudflared's log so it never blocks
        for line in process.stderr:
            match = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
            if match:
                print(f"Public address: {match.group(0)}", flush=True)

    threading.Thread(target=watch, daemon=True).start()
    return process


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-tunnel", action="store_true")
    args = parser.parse_args()

    load()
    print(f"Loaded {len(files)} files ({sum(map(len, files.values())) / 1e6:.0f} MB) from {PLOTS}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Local address: http://127.0.0.1:{args.port}", flush=True)
    tunnel = None if args.no_tunnel else start_tunnel(args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if tunnel:
            tunnel.terminate()
