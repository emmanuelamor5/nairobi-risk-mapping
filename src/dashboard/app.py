"""
Local dashboard server.

``GET /`` is the sign-in shell and contains no parcel coordinates.
``GET /api/map`` and ``GET /api/summary`` require a Firebase ID token whose
verified claims include a second factor. The Folium document is returned
only from those routes.

Firebase web config (the public apiKey, authDomain, and projectId) is read
from the environment. The project id used to verify tokens is
``FIREBASE_PROJECT_ID``.
"""

from __future__ import annotations

import json
import os
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .auth_gate import bearer_token, mfa_satisfied, verify_firebase_id_token
from .map_view import ACTION_COLORS, action_counts, build_policy_map, parcel_rows

SHELL_PATH = Path(__file__).with_name("shell.html")
ASSETS_DIR = Path(__file__).with_name("assets")

LIFT_NOTE = (
    "River-corridor lift on the Sprint 5 Colab run was 0.929, below the "
    "bar of 1. These actions are not a validated flood-hazard map. "
    "Resettlement still required high deprivation and high hazard."
)


def firebase_web_config() -> dict:
    """Public client config. Missing keys leave the sign-in form closed."""
    raw = os.environ.get("FIREBASE_WEB_CONFIG")
    if raw:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    return {
        "apiKey": os.environ.get("FIREBASE_API_KEY", ""),
        "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
        "projectId": os.environ.get("FIREBASE_PROJECT_ID", ""),
    }


def _settlement_names(settlement_keys) -> list:
    """Plain strings. A loaded ``.npz`` stores the keys as a NumPy array, and ``array or []`` raises."""
    if settlement_keys is None:
        return []
    return [str(key) for key in settlement_keys]


def render_shell(web_config: dict, settlement_keys, lift_note: str = LIFT_NOTE) -> str:
    html = SHELL_PATH.read_text(encoding="utf-8")
    replacements = {
        "{{FIREBASE_CONFIG_JSON}}": json.dumps(web_config),
        "{{LIFT_NOTE_JSON}}": json.dumps(lift_note),
        "{{SETTLEMENTS_JSON}}": json.dumps(_settlement_names(settlement_keys)),
        "{{COLORS_JSON}}": json.dumps(ACTION_COLORS),
    }
    for needle, value in replacements.items():
        html = html.replace(needle, value)
    return html


def _content_type(path: Path) -> str:
    return {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")


def _serve_asset(handler, url_path: str) -> None:
    """Icons and the labelled attention illustration. No parcel coordinates."""
    relative = url_path[len("/assets/"):]
    parts = relative.split("/")
    if not relative or any(part in {"", ".", ".."} for part in parts):
        _unauthorised(handler, 404, "Not found.")
        return
    root = ASSETS_DIR.resolve()
    target = (root / relative).resolve()
    if root not in target.parents or not target.is_file():
        _unauthorised(handler, 404, "Not found.")
        return
    body = target.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", _content_type(target))
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "public, max-age=3600")
    handler.end_headers()
    handler.wfile.write(body)


def _unauthorised(handler, status: int, message: str) -> None:
    body = message.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(records: dict, web_config: dict, project_id: str, verifier=None, lift_note: str = LIFT_NOTE):
    """
    Build a request handler closed over the parcel table.

    ``records`` needs ``positions`` (N, 2), ``action_index``, and optionally
    flood, deprivation, settlement index/keys, and ``pos_crs``. ``verifier``
    defaults to Firebase certificate verification and must return claims or
    raise ``ValueError``.
    """
    if verifier is None:
        verifier = partial(verify_firebase_id_token, project_id=project_id)
    shell = render_shell(web_config, records.get("settlement_keys"), lift_note)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _claims(self):
            token = bearer_token(self.headers.get("Authorization", ""))
            try:
                claims = verifier(token)
            except ValueError as exc:
                _unauthorised(self, 401, str(exc))
                return None
            if not mfa_satisfied(claims):
                _unauthorised(
                    self, 403,
                    "A password session is not enough. Complete the authenticator challenge.",
                )
                return None
            return claims

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path.startswith("/assets/"):
                _serve_asset(self, parsed.path)
                return
            if parsed.path == "/":
                body = shell.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path not in {"/api/map", "/api/summary", "/api/parcels"} and not parsed.path.startswith("/api/parcel/"):
                _unauthorised(self, 404, "Not found.")
                return
            if self._claims() is None:
                return
            settlement = parse_qs(parsed.query).get("settlement", [None])[0] or None
            if parsed.path == "/api/summary":
                payload = json.dumps(action_counts(records["action_index"])).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            elif parsed.path == "/api/parcels":
                # List payload omits per-feature vectors; the XAI route has those.
                rows = parcel_rows(records, settlement=settlement)
                brief = [{k: row[k] for k in row if k != "features"} for row in rows]
                payload = json.dumps(brief).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            elif parsed.path.startswith("/api/parcel/"):
                try:
                    index = int(parsed.path.rsplit("/", 1)[-1])
                except ValueError:
                    _unauthorised(self, 404, "Unknown parcel.")
                    return
                match = [row for row in parcel_rows(records) if row["index"] == index]
                if not match:
                    _unauthorised(self, 404, "Unknown parcel.")
                    return
                payload = json.dumps(match[0]).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            else:
                fmap = build_policy_map(
                    records["positions"],
                    records["action_index"],
                    flood_proba=records.get("flood_proba"),
                    deprivation_class=records.get("deprivation_class"),
                    settlement_index=records.get("settlement_index"),
                    settlement_keys=records.get("settlement_keys"),
                    pos_crs=records.get("pos_crs"),
                    settlement=settlement,
                )
                payload = fmap.get_root().render().encode("utf-8")
                content_type = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def serve(records: dict, host: str = "127.0.0.1", port: int = 8765, verifier=None, web_config: dict = None):
    """Start the dashboard and return the server. Call ``shutdown`` to stop it."""
    config = web_config if web_config is not None else firebase_web_config()
    project_id = os.environ.get("FIREBASE_PROJECT_ID", "") or config.get("projectId", "")
    handler = make_handler(records, config, project_id, verifier=verifier)
    server = ThreadingHTTPServer((host, port), handler)
    return server


def public_url_from_cloudflared_log(line: str):
    """Return the trycloudflare URL from one cloudflared log line, if present."""
    import re

    match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line or "")
    return match.group(0) if match else None


def _start_cloudflared(port: int):
    """
    Open a public HTTPS tunnel to a port on this machine.

    Colab's notebook iframe points at localhost on the laptop, so the frame
    stays blank. cloudflared gives a real address the laptop browser can open.
    The sign-in page is what that address shows; parcel data still requires MFA.
    """
    import os
    import shutil
    import subprocess
    import time
    import urllib.request

    binary = shutil.which("cloudflared")
    if binary is None:
        binary = "/tmp/cloudflared"
        if not os.path.exists(binary):
            print("Downloading cloudflared so Colab can publish a browser URL...")
            urllib.request.urlretrieve(
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                binary,
            )
            os.chmod(binary, 0o755)
    process = subprocess.Popen(
        [binary, "tunnel", "--url", f"http://127.0.0.1:{int(port)}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 40
    while time.time() < deadline:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            continue
        found = public_url_from_cloudflared_log(line)
        if found:
            return found
    return None


def launch(records: dict, port: int = 8765, web_config: dict = None):
    """
    Serve the dashboard in a background thread and return the server.

    On Colab, also print a public HTTPS URL. The embedded iframe and the
    ``colab.dev`` link both resolve to localhost in the laptop browser, so
    they stay blank.
    """
    import threading

    server = serve(records, host="0.0.0.0", port=port, web_config=web_config)
    thread = threading.Thread(target=server.serve_forever, name="parcel-dashboard", daemon=True)
    thread.start()
    print(f"Dashboard listening on port {port}  parcels={records['positions'].shape[0]}")
    try:
        import google.colab  # noqa: F401
    except ImportError:
        print(f"Open http://127.0.0.1:{port} in a browser on this computer.")
        return server
    print("Colab cannot show this server inside the notebook. Opening a public URL...")
    public = _start_cloudflared(port)
    if public:
        print("Open this address on your laptop:", public)
    else:
        print("Could not open a public URL. Download policy_map.html instead.")
    return server


def _load_records(table_path: str = None, graphs_dir: str = None) -> dict:
    from ..models.fusion import load_policy_table, resolve_graphs_dir, resolve_policy_table
    from ..models.multi_settlement import load_settlement_graphs
    from .map_view import join_graph_positions

    table_file = Path(table_path) if table_path else None
    if table_file is None or not table_file.is_file():
        table_file = resolve_policy_table()
    graph_root = Path(graphs_dir) if graphs_dir else None
    if graph_root is None or not graph_root.is_dir():
        graph_root = resolve_graphs_dir()
    table = load_policy_table(table_file)
    keys = [str(key) for key in table.get("settlement_keys", [])]
    graphs = load_settlement_graphs(graph_root, keys)
    print(f"policy table: {table_file}")
    print(f"graphs: {graph_root}")
    return join_graph_positions(table, graphs)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parcel policy dashboard")
    parser.add_argument("--table", default=None, help="policy_table.npz or .npz.gz; searched if omitted")
    parser.add_argument("--graphs", default=None, help="directory of {settlement}_graph.pt files")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    records = _load_records(args.table, args.graphs)
    try:
        import google.colab  # noqa: F401
        launch(records, port=args.port)
        print("The server stays up while this Colab runtime is alive.")
    except ImportError:
        server = serve(records, host=args.host, port=args.port)
        print(f"Dashboard at http://{args.host}:{args.port}  parcels={records['positions'].shape[0]}")
        print("The map is served only after a Firebase authenticator sign-in.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()


if __name__ == "__main__":
    main()
