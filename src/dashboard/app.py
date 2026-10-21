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
from .map_view import ACTION_COLORS, action_counts, build_policy_map

SHELL_PATH = Path(__file__).with_name("shell.html")

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


def render_shell(web_config: dict, settlement_keys, lift_note: str = LIFT_NOTE) -> str:
    html = SHELL_PATH.read_text(encoding="utf-8")
    replacements = {
        "{{FIREBASE_CONFIG_JSON}}": json.dumps(web_config),
        "{{LIFT_NOTE_JSON}}": json.dumps(lift_note),
        "{{SETTLEMENTS_JSON}}": json.dumps(list(settlement_keys or [])),
        "{{COLORS_JSON}}": json.dumps(ACTION_COLORS),
    }
    for needle, value in replacements.items():
        html = html.replace(needle, value)
    return html


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
            if parsed.path == "/":
                body = shell.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path not in {"/api/map", "/api/summary"}:
                _unauthorised(self, 404, "Not found.")
                return
            if self._claims() is None:
                return
            settlement = parse_qs(parsed.query).get("settlement", [None])[0] or None
            if parsed.path == "/api/summary":
                payload = json.dumps(action_counts(records["action_index"])).encode("utf-8")
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


def _load_records(table_path: str, graphs_dir: str) -> dict:
    from ..models.fusion import load_policy_table
    from ..models.multi_settlement import load_settlement_graphs
    from .map_view import join_graph_positions

    table = load_policy_table(table_path)
    keys = [str(key) for key in table.get("settlement_keys", [])]
    graphs = load_settlement_graphs(graphs_dir, keys)
    return join_graph_positions(table, graphs)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parcel policy dashboard")
    parser.add_argument("--table", default="data/policy_table.npz")
    parser.add_argument("--graphs", default="data/graphs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    records = _load_records(args.table, args.graphs)
    server = serve(records, host=args.host, port=args.port)
    print(f"Dashboard at http://{args.host}:{args.port}  parcels={records['positions'].shape[0]}")
    print("The map is served only after a Firebase authenticator sign-in.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
