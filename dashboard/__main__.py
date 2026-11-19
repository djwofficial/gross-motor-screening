"""Local read-only dashboard for saved tries. Owner: Revallo. Not part of the product flow.

    python -m dashboard                 # then open http://localhost:8765
    python -m dashboard --runs runs --port 8765

Reads {runs}/{session}/log.csv and the per-try JSON files written by app.session.Session.
It never writes anything, uses only the standard library, and listens on 127.0.0.1 only, so the
data never leaves this laptop. It shows scores: keep it off any screen KEE can see in testing mode.
"""

from __future__ import annotations

import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from common.body25 import BONES, LABELS, TASKS

PAGE = Path(__file__).with_name("page.html")


def list_runs(root: Path) -> list[str]:
    """Session folders that have a log.csv, newest name first. The root itself is "."."""
    if not root.is_dir():
        return []
    runs = [p.name for p in root.iterdir() if p.is_dir() and (p / "log.csv").exists()]
    runs.sort(reverse=True)
    if (root / "log.csv").exists():
        runs.insert(0, ".")
    return runs


def read_log(run_dir: Path) -> list[dict]:
    with (run_dir / "log.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def read_try(run_dir: Path, name: str) -> dict:
    """The parts of one try JSON the page needs for the replay."""
    rec = json.loads((run_dir / name).read_text())
    return {
        "t": rec["t"],
        "points": rec["points25"],
        "events": rec.get("events") or [],
        "clues": rec.get("clues") or {},
        "width": rec["meta"].get("width", 1280),
        "height": rec["meta"].get("height", 720),
    }


class Handler(BaseHTTPRequestHandler):
    root: Path

    def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(json.dumps(obj).encode(), "application/json", status)

    def _run_dir(self, q: dict) -> Path | None:
        run = q.get("run", [""])[0]
        return self.root / run if run in list_runs(self.root) else None  # only known folders

    def do_GET(self) -> None:
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path == "/":
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/info":
                self._json(
                    {
                        "runs": list_runs(self.root),
                        "bones": BONES,
                        "tasks": TASKS,
                        "labels": LABELS,
                    }
                )
            elif url.path == "/api/log":
                d = self._run_dir(q)
                self._json(read_log(d) if d else [], 200 if d else 404)
            elif url.path == "/api/try":
                d = self._run_dir(q)
                name = q.get("file", [""])[0]
                known = d is not None and name in {r["json_file"] for r in read_log(d)}
                if known and (d / name).exists():
                    self._json(read_try(d, name))
                else:
                    self._json({"error": "try not found"}, 404)
            else:
                self._json({"error": "not found"}, 404)
        except (OSError, ValueError, KeyError) as e:  # e.g. a file being written right now
            self._json({"error": str(e)}, 500)

    def log_message(self, *args) -> None:  # keep the terminal quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Local dashboard for saved tries (read-only).")
    ap.add_argument("--runs", type=Path, default=Path("runs"), help="folder with session folders")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    Handler.root = args.runs
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard: http://localhost:{args.port}   (reading {args.runs.resolve()})")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
