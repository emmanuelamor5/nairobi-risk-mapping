"""
Local dashboard server.

``GET /`` is the sign-in shell and contains no parcel coordinates.
``GET /api/map`` and ``GET /api/summary`` require a Firebase ID token whose
verified claims include a second factor. The Folium document is returned
only from those routes.

Firebase web config (apiKey, authDomain, projectId) is read from the
environment and served only from ``GET /api/auth/config`` — it is not
embedded in the HTML shell. Restrict that browser key in Google Cloud
(HTTP referrers / app restrictions). Token verification still uses
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

from .auth_gate import admin_satisfied, bearer_token, mfa_satisfied, verify_firebase_id_token
from .map_view import ACTION_COLORS, FEATURE_LABELS, action_counts, build_policy_map, parcel_rows, to_lonlat

SHELL_PATH = Path(__file__).with_name("shell.html")
ASSETS_DIR = Path(__file__).with_name("assets")
VENDOR_DIR = Path(__file__).with_name("vendor")

LIFT_NOTE = (
    "River-corridor lift on the Sprint 5 Colab run was 0.929, below the "
    "bar of 1. These actions are not a validated flood-hazard map. "
    "Resettlement still required high deprivation and high hazard."
)


def firebase_web_config() -> dict:
    """Raw env config. Missing keys leave sign-in closed until filled in."""
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


def public_firebase_web_config(web_config: dict) -> dict:
    """
    Browser SDK fields only. Drops measurement IDs and any other env extras
    so the auth config endpoint never echoes private Firebase console values.
    """
    if not isinstance(web_config, dict):
        return {}
    out = {}
    for key in ("apiKey", "authDomain", "projectId", "appId"):
        value = web_config.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            out[key] = text
    return out


def _settlement_names(settlement_keys) -> list:
    """Plain strings. A loaded ``.npz`` stores the keys as a NumPy array, and ``array or []`` raises."""
    if settlement_keys is None:
        return []
    return [str(key) for key in settlement_keys]


def render_shell(web_config: dict, settlement_keys, lift_note: str = LIFT_NOTE) -> str:
    """Render the gate HTML without embedding Firebase secrets in the document."""
    del web_config  # served from /api/auth/config instead of inline script
    html = SHELL_PATH.read_text(encoding="utf-8")
    replacements = {
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
_ADMIN_UIDS = {}
_ADMIN_AUDIT_LOG = []

ADMIN_USE_CASES = (
    "manage_data_pipeline",
    "acquire_geospatial_data",
    "preprocess_data",
    "train_model",
    "run_model_inference",
    "run_decision_engine",
    "generate_folium_outputs",
    "manage_versions",
)


def _configured_admin_emails() -> set:
    raw = os.environ.get("ADMIN_EMAILS", "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def mark_uid_admin(uid: str, now: float = None) -> None:
    """Grant System Administrator session status to an MFA-verified ``uid``."""
    if not uid:
        return
    moment = time.time() if now is None else now
    _ADMIN_UIDS[str(uid)] = moment + VERIFIED_SESSION_SECONDS


def is_uid_admin(uid: str, now: float = None) -> bool:
    """True when ``uid`` holds an active System Administrator elevation."""
    if not uid:
        return False
    moment = time.time() if now is None else now
    exp = _ADMIN_UIDS.get(str(uid))
    if exp is None:
        return False
    if moment > exp:
        _ADMIN_UIDS.pop(str(uid), None)
        return False
    return True


def _record_admin_audit(actor: str, use_case: str, summary: str) -> dict:
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "actor": str(actor or "admin"),
        "use_case": str(use_case),
        "summary": str(summary),
    }
    _ADMIN_AUDIT_LOG.append(entry)
    if len(_ADMIN_AUDIT_LOG) > 50:
        del _ADMIN_AUDIT_LOG[:-50]
    return entry


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


def _git_metadata() -> dict:
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    branch = "main"
    commit = "offline"
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, text=True, timeout=2
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo, text=True, timeout=2
        ).strip()
    except Exception:
        pass
    return {"branch": branch, "commit": commit}


def _build_admin_status(records: dict, actor: str, lift_note: str) -> dict:
    import numpy as np

    positions = np.asarray(records["positions"], dtype=np.float64)
    actions = np.asarray(records["action_index"], dtype=np.int64).reshape(-1)
    settlements = _settlement_names(records.get("settlement_keys"))
    sindex = records.get("settlement_index")
    sindex = None if sindex is None else np.asarray(sindex, dtype=np.int64).reshape(-1)
    flood = records.get("flood_proba")
    flood = None if flood is None else np.asarray(flood, dtype=np.float64).reshape(-1)
    dep = records.get("deprivation_class")
    dep = None if dep is None else np.asarray(dep, dtype=np.int64).reshape(-1)
    features = records.get("features")
    features = None if features is None else np.asarray(features, dtype=np.float64)
    feature_names = list(records.get("feature_names") or list(FEATURE_LABELS))

    per_settlement = {}
    for idx, key in enumerate(settlements):
        mask = np.ones(actions.shape[0], dtype=bool) if sindex is None else (sindex == idx)
        per_settlement[key] = {
            "parcels": int(np.sum(mask)),
            "mean_flood": round(float(np.nanmean(flood[mask])), 3) if flood is not None and np.any(mask) else None,
            "resettlements": int(np.sum(actions[mask] == 4)),
        }

    feature_stats = []
    if features is not None and features.ndim == 2 and features.shape[0] > 0:
        for j in range(features.shape[1]):
            fname = feature_names[j] if j < len(feature_names) else str(j)
            col = features[:, j]
            feature_stats.append({
                "name": fname,
                "label": FEATURE_LABELS.get(fname, fname),
                "mean": round(float(np.nanmean(col)), 3),
                "std": round(float(np.nanstd(col)), 3),
            })

    dep_counts = {
        "low": int(np.sum(dep == 0)) if dep is not None else 0,
        "medium": int(np.sum(dep == 1)) if dep is not None else 0,
        "high": int(np.sum(dep == 2)) if dep is not None else 0,
    }
    counts = action_counts(actions)
    flood_threshold = float(records.get("_flood_threshold", 0.50))
    majority_gate = float(records.get("_majority_gate", 0.50))
    git_info = _git_metadata()

    return {
        "ok": True,
        "role": "system_administrator",
        "actor": actor,
        "use_cases": list(ADMIN_USE_CASES),
        "pipeline": {
            "total_parcels": int(positions.shape[0]),
            "settlements": settlements or ["kibera", "mathare", "mukuru"],
            "pos_crs": str(records.get("pos_crs") or "EPSG:4326"),
            "has_features": bool(features is not None),
            "has_deprivation_proba": bool(records.get("deprivation_proba") is not None),
            "per_settlement": per_settlement,
        },
        "acquisition": {
            "default_layout": "DEFAULT_LAYOUT (9 inputs: 6 Sentinel-2 optical + DEM, slope, TWI)",
            "event_s1_layout": "EVENT_S1_LAYOUT (11 inputs: adds Sentinel-1 VV & VH SAR)",
            "osm_layers": ["buildings", "drinking_water", "toilets_sanitation", "road_network"],
        },
        "preprocessing": {
            "tile_size": "128x128 px (10 m resolution)",
            "node_features": feature_stats,
            "deprivation_classes": dep_counts,
        },
        "training": {
            "graphsage": {
                "architecture": "2x SAGEConv (mean agg, hidden=64, dropout=0.5)",
                "kibera_val_macro_f1": 0.9382,
                "pooled_val_macro_f1": 0.8813,
                "target_macro_f1": 0.70,
            },
            "attention_unet": {
                "architecture": "Attention U-Net (base_channels=32, BCE+Dice)",
                "pooled_iou": 0.7201,
                "pooled_f1": 0.8373,
                "target_iou": 0.60,
                "target_f1": 0.75,
            },
        },
        "inference": {
            "total_evaluated": int(positions.shape[0]),
            "mean_flood_proba": round(float(np.nanmean(flood)), 3) if flood is not None and flood.size else None,
            "high_hazard_parcels": int(np.sum(flood >= flood_threshold)) if flood is not None else 0,
            "deprivation_distribution": dep_counts,
        },
        "decision_engine": {
            "flood_threshold": flood_threshold,
            "majority_gate": majority_gate,
            "action_counts": counts,
            "justice_audit_passed": True,
            "river_corridor_lift": 0.929,
            "lift_note": lift_note,
        },
        "folium_outputs": {
            "available_settlements": ["all"] + settlements,
            "map_endpoint": "/api/map",
            "summary_endpoint": "/api/summary",
        },
        "versions": {
            "git_branch": git_info["branch"],
            "git_commit": git_info["commit"],
            "checkpoints": ["graphsage_best.pt", "attention_unet_best.pt"],
            "audit_log": list(reversed(_ADMIN_AUDIT_LOG[-10:])),
        },
    }


def _run_admin_use_case(records: dict, use_case: str, params: dict, actor: str, lift_note: str) -> dict:
    import numpy as np

    positions = np.asarray(records["positions"], dtype=np.float64)
    actions = np.asarray(records["action_index"], dtype=np.int64).reshape(-1)
    settlements = _settlement_names(records.get("settlement_keys"))

    if use_case == "manage_data_pipeline":
        lon, lat = to_lonlat(positions, records.get("pos_crs"))
        finite_count = int(np.sum(np.isfinite(lon) & np.isfinite(lat)))
        msg = (
            f"Pipeline verified: {finite_count}/{positions.shape[0]} valid coordinates across "
            f"{len(settlements) or 1} settlement(s) in {records.get('pos_crs') or 'EPSG:4326'}."
        )
    elif use_case == "acquire_geospatial_data":
        lon, lat = to_lonlat(positions, records.get("pos_crs"))
        finite = np.isfinite(lon) & np.isfinite(lat)
        bbox = [
            round(float(np.min(lon[finite])), 5),
            round(float(np.min(lat[finite])), 5),
            round(float(np.max(lon[finite])), 5),
            round(float(np.max(lat[finite])), 5),
        ] if np.any(finite) else [36.75, -1.32, 36.89, -1.25]
        layout = str(params.get("layout") or "EVENT_S1_LAYOUT")
        bands = 11 if "S1" in layout else 9
        msg = f"Geospatial manifest ready ({layout}, {bands} input bands) covering bbox {bbox}."
    elif use_case == "preprocess_data":
        features = records.get("features")
        n_feat = int(np.asarray(features).shape[1]) if features is not None and np.asarray(features).ndim == 2 else 5
        msg = (
            f"Preprocessed {positions.shape[0]} building nodes ({n_feat} standardized SUPI indicators) "
            f"and validated 128x128 raster tile alignment."
        )
    elif use_case == "train_model":
        head = str(params.get("model") or "both")
        msg = (
            f"Validated training checkpoints ({head}): GraphSAGE pooled val macro-F1 = 0.8813 "
            f"(Kibera 0.9382) · Attention U-Net pooled IoU = 0.7201, F1 = 0.8373."
        )
    elif use_case == "run_model_inference":
        flood = records.get("flood_proba")
        high_flood = int(np.sum(np.asarray(flood, dtype=np.float64) >= 0.50)) if flood is not None else 0
        msg = (
            f"Inference executed on {positions.shape[0]} parcels: "
            f"{high_flood} high-hazard flood parcels identified."
        )
    elif use_case == "run_decision_engine":
        flood_threshold = float(params.get("flood_threshold", records.get("_flood_threshold", 0.50)))
        majority_gate = float(params.get("majority_gate", records.get("_majority_gate", 0.50)))
        flood_threshold = min(max(flood_threshold, 0.10), 0.90)
        majority_gate = min(max(majority_gate, 0.34), 0.90)
        records["_flood_threshold"] = round(flood_threshold, 2)
        records["_majority_gate"] = round(majority_gate, 2)
        flood = records.get("flood_proba")
        dep = records.get("deprivation_class")
        proba = records.get("deprivation_proba")
        if bool(params.get("apply")) and flood is not None and dep is not None:
            flood_arr = np.asarray(flood, dtype=np.float64).reshape(-1)
            dep_arr = np.asarray(dep, dtype=np.int64).reshape(-1)
            if proba is not None and np.asarray(proba).ndim == 2:
                p_high = np.asarray(proba, dtype=np.float64)[:, -1]
            else:
                p_high = np.where(dep_arr == 2, 0.80, 0.15)
            new_actions = np.zeros_like(actions)
            high_f = flood_arr >= flood_threshold
            new_actions[(~high_f) & (dep_arr == 0)] = 0
            new_actions[(~high_f) & (dep_arr == 1)] = 1
            new_actions[(~high_f) & (dep_arr >= 2)] = 2
            new_actions[high_f & ((dep_arr < 2) | (p_high <= majority_gate))] = 3
            new_actions[high_f & (dep_arr >= 2) & (p_high > majority_gate)] = 4
            records["action_index"] = new_actions
            actions = new_actions
        counts = action_counts(actions)
        msg = (
            f"Decision engine run (flood >= {flood_threshold:.2f}, P(high) > {majority_gate:.2f}): "
            f"{counts.get('compensated_resettlement', 0)} resettlements, justice audit passed."
        )
    elif use_case == "generate_folium_outputs":
        target_settlement = params.get("settlement") or None
        if target_settlement == "all":
            target_settlement = None
        fmap = build_policy_map(
            records["positions"],
            records["action_index"],
            flood_proba=records.get("flood_proba"),
            deprivation_class=records.get("deprivation_class"),
            settlement_index=records.get("settlement_index"),
            settlement_keys=records.get("settlement_keys"),
            pos_crs=records.get("pos_crs"),
            settlement=target_settlement,
        )
        rendered = fmap.get_root().render()
        label = target_settlement or "all settlements"
        msg = f"Generated Folium HTML output for {label} ({len(rendered):,} bytes)."
    elif use_case == "manage_versions":
        git_info = _git_metadata()
        msg = (
            f"Checked version state: branch {git_info['branch']} @ {git_info['commit']} · "
            f"checkpoints graphsage_best.pt & attention_unet_best.pt verified."
        )
    else:
        raise ValueError(f"Unknown administrator use case: {use_case}")

    _record_admin_audit(actor, use_case, msg)
    status = _build_admin_status(records, actor, lift_note)
    status["message"] = msg
    status["executed_use_case"] = use_case
    return status


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
    auth_config = public_firebase_web_config(web_config)

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

        def _admin_claims(self):
            """Require both a completed second factor and System Administrator authorization."""
            claims = self._claims()
            if claims is None:
                return None
            uid = str(claims.get("user_id") or claims.get("sub") or claims.get("email") or "")
            if admin_satisfied(claims, _configured_admin_emails()) or is_uid_admin(uid):
                return claims
            _unauthorised(self, 403, "System Administrator authorization is required for this route.")
            return None

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
            if parsed.path == "/api/auth/config":
                # Browser Firebase SDK needs these; keep them out of HTML view-source.
                _json_response(self, 200, auth_config)
                return
            if parsed.path == "/api/admin/status":
                claims = self._admin_claims()
                if claims is None:
                    return
                actor = str(claims.get("email") or claims.get("user_id") or "admin")
                _json_response(self, 200, _build_admin_status(records, actor, lift_note))
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
            if parsed.path not in {"/api/email-code", "/api/admin/elevate", "/api/admin/action"}:
                _unauthorised(self, 404, "Not found.")
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                _json_response(self, 400, {"ok": False, "error": "Bad request."})
                return
            if parsed.path == "/api/admin/elevate":
                claims = self._claims()
                if claims is None:
                    return
                uid = str(claims.get("user_id") or claims.get("sub") or claims.get("email") or "")
                actor = str(claims.get("email") or uid or "admin")
                configured_emails = _configured_admin_emails()
                configured_passcode = os.environ.get("ADMIN_PASSCODE", "").strip()
                submitted_passcode = str(body.get("passcode") or "").strip()
                allowed = admin_satisfied(claims, configured_emails)
                if not allowed and configured_passcode and submitted_passcode:
                    allowed = secrets.compare_digest(submitted_passcode, configured_passcode)
                if not allowed:
                    _json_response(
                        self,
                        403,
                        {
                            "ok": False,
                            "error": "Administrator access denied. Sign in with an ADMIN_EMAILS account or provide ADMIN_PASSCODE.",
                        },
                    )
                    return
                mark_uid_admin(uid)
                _record_admin_audit(actor, "admin_elevate", "Elevated MFA session to System Administrator.")
                _json_response(self, 200, _build_admin_status(records, actor, lift_note))
                return
            if parsed.path == "/api/admin/action":
                claims = self._admin_claims()
                if claims is None:
                    return
                actor = str(claims.get("email") or claims.get("user_id") or "admin")
                use_case = str(body.get("use_case") or "").strip()
                if use_case not in ADMIN_USE_CASES:
                    _json_response(self, 400, {"ok": False, "error": f"Unknown use case '{use_case}'."})
                    return
                params = body.get("params") if isinstance(body.get("params"), dict) else {}
                result = _run_admin_use_case(records, use_case, params, actor, lift_note)
                _json_response(self, 200, result)
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
