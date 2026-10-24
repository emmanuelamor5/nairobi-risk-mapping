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
import random
import secrets
import time
import urllib.request
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .auth_gate import bearer_token, mfa_satisfied, verify_firebase_id_token
from .map_view import ACTION_COLORS, action_counts, build_policy_map, parcel_rows

SHELL_PATH = Path(__file__).with_name("shell.html")
ASSETS_DIR = Path(__file__).with_name("assets")
VENDOR_DIR = Path(__file__).with_name("vendor")

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
        ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
    }.get(path.suffix.lower(), "application/octet-stream")


def _serve_static(handler, url_path: str, prefix: str, root: Path) -> bool:
    """Serve one file from ``root``. False when the URL is not under ``prefix``."""
    if not url_path.startswith(prefix):
        return False
    relative = url_path[len(prefix):]
    parts = relative.split("/")
    base = root.resolve()
    target = (base / relative).resolve()
    if not relative or any(part in {"", ".", ".."} for part in parts) or base not in target.parents or not target.is_file():
        _unauthorised(handler, 404, "Not found.")
        return True
    body = target.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", _content_type(target))
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "public, max-age=3600")
    handler.end_headers()
    handler.wfile.write(body)
    return True


def _serve_asset(handler, url_path: str) -> None:
    """Icons and the labelled attention illustration. No parcel coordinates."""
    _serve_static(handler, url_path, "/assets/", ASSETS_DIR)


_EMAIL_CODES = {}
EMAIL_CODE_SECONDS = 300
VERIFIED_SESSION_SECONDS = 12 * 3600
_VERIFIED_UIDS = {}
_PENDING_APPLE_TOKENS = {}
_VERIFIED_TOKENS = {}


def issue_email_code(uid: str, now: float = None) -> str:
    """Store a 6-digit code for five minutes. A new code replaces the old one."""
    moment = time.time() if now is None else now
    code = f"{random.randint(0, 999999):06d}"
    _EMAIL_CODES[str(uid)] = {"code": code, "exp": moment + EMAIL_CODE_SECONDS}
    return code


def mark_uid_verified(uid: str, now: float = None) -> None:
    """Record that ``uid`` completed the second-step login code."""
    if not uid:
        return
    moment = time.time() if now is None else now
    _VERIFIED_UIDS[str(uid)] = moment + VERIFIED_SESSION_SECONDS


def is_uid_verified(uid: str, now: float = None) -> bool:
    """True when ``uid`` has a non-expired second-step code verification."""
    if not uid:
        return False
    moment = time.time() if now is None else now
    exp = _VERIFIED_UIDS.get(str(uid))
    if exp is None:
        return False
    if moment > exp:
        _VERIFIED_UIDS.pop(str(uid), None)
        return False
    return True


def issue_pending_apple_token(email: str, now: float = None) -> str:
    """Issue a short-lived pre-MFA token for Apple ID sign-in before the 5-minute code step."""
    moment = time.time() if now is None else now
    clean = str(email or "").strip().lower()
    token = f"apple-pre-{secrets.token_urlsafe(24)}"
    _PENDING_APPLE_TOKENS[token] = {
        "user_id": f"apple:{clean}",
        "email": clean,
        "firebase": {"sign_in_provider": "apple.com"},
        "exp": moment + EMAIL_CODE_SECONDS * 2,
    }
    return token


def issue_verified_token(uid: str, email: str = "", now: float = None) -> str:
    """Issue a verified session token after the 5-minute code check passes."""
    moment = time.time() if now is None else now
    token = f"verified-{secrets.token_urlsafe(24)}"
    _VERIFIED_TOKENS[token] = {
        "user_id": str(uid),
        "email": str(email or ""),
        "firebase": {"sign_in_provider": "apple.com", "sign_in_second_factor": "email_code"},
        "exp": moment + VERIFIED_SESSION_SECONDS,
    }
    return token


def check_email_code(uid: str, submitted: str, now: float = None):
    """Return (ok, message). An expired or unknown code does not succeed."""
    moment = time.time() if now is None else now
    row = _EMAIL_CODES.get(str(uid))
    if not row:
        return False, "Send a code first."
    if moment > row["exp"]:
        _EMAIL_CODES.pop(str(uid), None)
        return False, "That code expired. Send a new one."
    if str(submitted or "").strip() != row["code"]:
        return False, "That code was rejected."
    _EMAIL_CODES.pop(str(uid), None)
    mark_uid_verified(uid, now=moment)
    return True, ""


def send_login_email(email: str, code: str) -> None:
    """Deliver the sign-in code. Raises if the mail relay refuses it."""
    print(f"[Nairobi Risk Mapping] 5-minute verification code for {email}: {code}", flush=True)
    payload = json.dumps({
        "_subject": "Nairobi Risk Mapping code",
        "_captcha": "false",
        "_template": "box",
        "message": f"Your Nairobi Risk Mapping code is {code}. It expires in 5 minutes.",
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://formsubmit.co/ajax/" + email,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Origin": "https://nairobi-risk-mapping.local",
            "Referer": "https://nairobi-risk-mapping.local/",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.status >= 400:
            raise RuntimeError("mail relay refused the message")


def _json_response(handler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
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
            now = time.time()
            verified_row = _VERIFIED_TOKENS.get(token)
            if verified_row:
                if now <= verified_row["exp"]:
                    return verified_row
                _VERIFIED_TOKENS.pop(token, None)
            if token in _PENDING_APPLE_TOKENS:
                _unauthorised(
                    self, 403,
                    "A password session is not enough. Complete the authenticator challenge.",
                )
                return None
            try:
                claims = verifier(token)
            except ValueError as exc:
                _unauthorised(self, 401, str(exc))
                return None
            uid = str(claims.get("user_id") or claims.get("sub") or claims.get("email") or "")
            if not mfa_satisfied(claims) and not is_uid_verified(uid, now=now):
                _unauthorised(
                    self, 403,
                    "A password session is not enough. Complete the authenticator challenge.",
                )
                return None
            return claims

        def _identity(self):
            """A valid Firebase or pending Apple session. The login code itself is the second step."""
            token = bearer_token(self.headers.get("Authorization", ""))
            now = time.time()
            for store in (_PENDING_APPLE_TOKENS, _VERIFIED_TOKENS):
                row = store.get(token)
                if row:
                    if now <= row["exp"]:
                        return row
                    store.pop(token, None)
            try:
                return verifier(token)
            except ValueError as exc:
                _unauthorised(self, 401, str(exc))
                return None

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path.startswith("/assets/"):
                _serve_asset(self, parsed.path)
                return
            if _serve_static(self, parsed.path, "/vendor/", VENDOR_DIR):
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

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/email-code":
                _unauthorised(self, 404, "Not found.")
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                _json_response(self, 400, {"ok": False, "error": "Bad request."})
                return
            action = body.get("action")
            if action == "apple-init":
                apple_email = str(body.get("email") or "").strip().lower()
                if "@" not in apple_email or "." not in apple_email.split("@")[-1]:
                    _json_response(self, 400, {"ok": False, "error": "Enter a valid Apple ID email address."})
                    return
                pre_token = issue_pending_apple_token(apple_email)
                _json_response(self, 200, {"ok": True, "pre_token": pre_token, "email": apple_email})
                return
            claims = self._identity()
            if claims is None:
                return
            uid = str(claims.get("user_id") or claims.get("sub") or claims.get("email") or "")
            email = str(claims.get("email") or body.get("email") or "").strip()
            if action == "send":
                if not email:
                    _json_response(self, 400, {"ok": False, "error": "This account has no email address. Use a phone code."})
                    return
                code = issue_email_code(uid)
                try:
                    send_login_email(email, code)
                except Exception:
                    _EMAIL_CODES.pop(uid, None)
                    _json_response(self, 502, {"ok": False, "error": "The email was not sent. Use a phone code, or try again."})
                    return
                _json_response(self, 200, {"ok": True, "expires_in": EMAIL_CODE_SECONDS})
                return
            if action == "check":
                ok, message = check_email_code(uid, body.get("code"))
                payload = {"ok": ok, "error": message}
                if ok:
                    payload["session_token"] = issue_verified_token(uid, email)
                _json_response(self, 200 if ok else 400, payload)
                return
            if action == "phone-verified":
                mark_uid_verified(uid)
                _json_response(self, 200, {"ok": True, "session_token": issue_verified_token(uid, email)})
                return
            _json_response(self, 400, {"ok": False, "error": "Unknown action."})

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
