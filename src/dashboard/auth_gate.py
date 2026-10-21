"""
Firebase Authentication gate for the parcel dashboard.

Parcel actions include compensated resettlement. The map is served only
when the caller's Firebase ID token was minted after a second factor.
Password-only sessions are rejected. The check looks at the verified
claim ``firebase.sign_in_second_factor``, which Firebase sets to ``totp``
or ``phone`` once the multi-factor challenge has succeeded.

Token signature verification uses Google's Firebase certs. Tests pass a
decoded claim dict straight into ``mfa_satisfied`` and never call Google.
"""

from __future__ import annotations


SECOND_FACTORS = frozenset({"totp", "phone"})


def mfa_satisfied(claims: dict) -> bool:
    """True when a verified ID-token payload records a completed second factor."""
    if not isinstance(claims, dict):
        return False
    firebase = claims.get("firebase")
    if not isinstance(firebase, dict):
        return False
    return firebase.get("sign_in_second_factor") in SECOND_FACTORS


def verify_firebase_id_token(token: str, project_id: str) -> dict:
    """
    Verify a Firebase ID token for ``project_id`` and return its claims.

    Raises ``ValueError`` when the token is missing, the project id is
    missing, or Google rejects the signature, audience, or expiry.
    """
    if not token or not str(token).strip():
        raise ValueError("Missing Firebase ID token.")
    if not project_id or not str(project_id).strip():
        raise ValueError("FIREBASE_PROJECT_ID is not set.")
    from google.auth.transport import requests
    from google.oauth2 import id_token

    try:
        claims = id_token.verify_firebase_token(
            str(token).strip(),
            requests.Request(),
            audience=str(project_id).strip(),
        )
    except Exception as exc:
        raise ValueError(f"Firebase ID token was rejected: {exc}") from exc
    if not isinstance(claims, dict):
        raise ValueError("Firebase token verification did not return claims.")
    return claims


def bearer_token(header_value: str) -> str:
    """Return the token from an ``Authorization`` header, or an empty string."""
    if not header_value:
        return ""
    parts = str(header_value).split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return ""
