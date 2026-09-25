"""`loopmath builder`: one page and a JSON API on 127.0.0.1, answered from the session (spec 02, `builder`).

`GET /` is the builder page, `GET /assets/<file>` a file from `views/assets/`, `GET /api/context` the
session's context, `POST /api/predict` an edited configuration's numbers and `POST /api/predict_many` a list's
(0.2.2). Errors are `{piece, message}`, the piece null for a request as a whole. The server binds 127.0.0.1
only (port 0: a free port), prints its URL, opens the browser unless `--no-open`, and stops on Ctrl+C.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ..output import EXIT_OK, EXIT_USER, fail
from .context import Session, build_session, context_payload
from .predict import err, predict, predict_many

HOST = "127.0.0.1"
MAX_BODY = 1 << 20  # a configuration is a few kilobytes
ASSET_RE = re.compile(r"^/assets/([a-z][a-z0-9_-]*\.(?:css|js))$")
TYPES = {"css": "text/css; charset=utf-8", "js": "text/javascript; charset=utf-8"}
ROUTES = {"/api/predict": predict, "/api/predict_many": predict_many}


def finite(obj: Any) -> Any:
    """The object with NaN and infinities as null, which JSON.parse reads."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    return obj


def dumps(obj: Any) -> bytes:
    from ..recommend.storeread import _default

    try:
        text = json.dumps(obj, ensure_ascii=False, allow_nan=False, default=_default)
    except ValueError:
        text = json.dumps(finite(json.loads(json.dumps(obj, default=_default))), ensure_ascii=False)
    return text.encode("utf-8")


def page() -> str:
    from importlib import resources

    return resources.files("loopmath.views").joinpath("assets", "builder.html").read_text(encoding="utf-8")


class BuilderServer(ThreadingHTTPServer):
    """A `ThreadingHTTPServer` on 127.0.0.1 that holds the session."""

    daemon_threads = True

    def __init__(self, session: Session, port: int = 0):
        self.session = session
        super().__init__((HOST, port), Handler)

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.server_address[1]}/"


class Handler(BaseHTTPRequestHandler):
    server: BuilderServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the stdlib's name
        """Quiet: the terminal shows the URL and errors only."""

    def send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, code: int, obj: Any) -> None:
        self.send(code, dumps(obj), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - the stdlib's name
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self.send(200, page().encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/context":
            self.send_json(200, context_payload(self.server.session))
        elif (m := ASSET_RE.match(path)) is not None:
            from ..views.common import asset

            try:
                text = asset(m.group(1))
            except (FileNotFoundError, OSError):
                self.send_json(404, {"ok": False, "errors": [err(None, f"no asset {m.group(1)}")]})
                return
            self.send(200, text.encode("utf-8"), TYPES[m.group(1).rsplit(".", 1)[1]])
        else:
            self.send_json(404, {"ok": False, "errors": [err(None, f"no such path: {path}")]})

    do_HEAD = do_GET

    def do_POST(self) -> None:  # noqa: N802 - the stdlib's name
        path = self.path.split("?", 1)[0]
        route = ROUTES.get(path)
        if route is None:
            self.send_json(404, {"ok": False, "errors": [err(None, f"no such path: {path}")]})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            self.send_json(413, {"ok": False, "errors": [err(None, "the body must be one JSON object under 1 MB")]})
            self.close_connection = True
            return
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8") or "null")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.send_json(400, {"ok": False, "errors": [err(None, f"the body is not JSON: {exc}")]})
            return
        try:
            out = route(self.server.session, body)
        except Exception as exc:  # noqa: BLE001 - one bad request must not stop the page
            print(f"builder: {path} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            self.send_json(500, {"ok": False, "errors": [err(None, f"the prediction failed: {type(exc).__name__}: "
                                                                    f"{exc}")]})
            return
        self.send_json(200, out)


def start(session: Session, port: int = 0) -> tuple[BuilderServer, threading.Thread]:
    """The server running on a thread (tests, and a page that wants the builder beside it)."""
    srv = BuilderServer(session, port)
    t = threading.Thread(target=srv.serve_forever, name="loopmath-builder", daemon=True)
    t.start()
    return srv, t


def serve(args: argparse.Namespace) -> int:
    """The `loopmath builder` handler."""
    port = getattr(args, "port", 0) or 0
    if not 0 <= port <= 65535:
        return fail(f"--port takes 0 to 65535; got {port}")
    session, code = build_session(args)
    if session is None:
        return code
    context_payload(session)  # built before the first request, so the page loads at once
    try:
        srv = BuilderServer(session, port)
    except OSError as exc:
        return fail(f"cannot listen on {HOST}:{port}: {exc.strerror or exc}", EXIT_USER)
    print(f"loopmath builder: {srv.url}  (Ctrl+C stops it)", flush=True)
    if not getattr(args, "no_open", False):
        webbrowser.open(srv.url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nloopmath builder: stopped", flush=True)
    finally:
        srv.server_close()
    return EXIT_OK
