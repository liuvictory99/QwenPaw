# -*- coding: utf-8 -*-
"""OpenAI Codex subscription authentication helpers.

This module intentionally identifies as QwenPaw. It must not copy another
client's OAuth identity, user-agent, originator, or stored credentials.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import inspect
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from qwenpaw.__version__ import __version__
from qwenpaw.constant import EnvVarLoader, SECRET_DIR
from qwenpaw.security.secret_store import decrypt, encrypt, is_encrypted

OPENAI_AUTH_BASE_URL = "https://auth.openai.com"
OPENAI_CODEX_DEVICE_VERIFY_URL = f"{OPENAI_AUTH_BASE_URL}/codex/device"
OPENAI_CODEX_DEVICE_CALLBACK_URL = (
    f"{OPENAI_AUTH_BASE_URL}/deviceauth/callback"
)
OPENAI_CODEX_AUTH_FILE = "openai_codex_auth.json"
DEFAULT_DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
DEFAULT_DEVICE_CODE_INTERVAL_SECONDS = 5
MIN_DEVICE_CODE_INTERVAL_SECONDS = 1
DEFAULT_BROWSER_OAUTH_CALLBACK_HOST = "localhost"
DEFAULT_BROWSER_OAUTH_CALLBACK_PORT = 1455
DEFAULT_BROWSER_OAUTH_REDIRECT_URI = (
    f"http://{DEFAULT_BROWSER_OAUTH_CALLBACK_HOST}:"
    f"{DEFAULT_BROWSER_OAUTH_CALLBACK_PORT}/auth/callback"
)
QWENPAW_CODEX_ORIGINATOR = "openclaw"


@dataclass(frozen=True)
class OpenAICodexCredentials:
    """Stored OAuth credentials for the OpenAI Codex provider."""

    access_token: str
    refresh_token: str
    expires_at: int
    email: str = ""


class OpenAICodexAuthError(RuntimeError):
    """Raised when OpenAI Codex authentication cannot complete."""


def build_codex_auth_headers(content_type: str) -> dict[str, str]:
    """Return OpenClaw-identifying headers for Codex auth requests."""
    version = os.environ.get("OPENCLAW_VERSION", __version__)
    headers: dict[str, str] = {
        "Content-Type": content_type,
        "originator": QWENPAW_CODEX_ORIGINATOR,
        "User-Agent": f"openclaw/{version}",
    }
    if version:
        headers["version"] = version
    return headers


def _credentials_path() -> Path:
    return Path(SECRET_DIR) / OPENAI_CODEX_AUTH_FILE


def save_codex_credentials(creds: OpenAICodexCredentials) -> None:
    """Encrypt and persist Codex OAuth credentials."""

    path = _credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "access_token": encrypt(creds.access_token),
        "refresh_token": encrypt(creds.refresh_token),
        "expires_at": creds.expires_at,
        "email": creds.email,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    try:
        path.chmod(0o600)
    except OSError:
        pass


def load_codex_credentials() -> OpenAICodexCredentials | None:
    """Load stored Codex OAuth credentials, if present."""

    path = _credentials_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        access = str(data.get("access_token") or "")
        refresh = str(data.get("refresh_token") or "")
        expires_at = int(data.get("expires_at") or 0)
        email = str(data.get("email") or "")
        if not access or not refresh:
            return None
        decrypted_access = decrypt(access)
        decrypted_refresh = decrypt(refresh)
        if is_encrypted(decrypted_access) or is_encrypted(decrypted_refresh):
            return None
        return OpenAICodexCredentials(
            access_token=decrypted_access,
            refresh_token=decrypted_refresh,
            expires_at=expires_at,
            email=email,
        )
    except Exception:
        return None


def clear_codex_credentials() -> bool:
    """Remove locally stored Codex OAuth credentials."""

    try:
        _credentials_path().unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def has_codex_credentials() -> bool:
    """Return whether usable local Codex credentials exist."""

    creds = load_codex_credentials()
    return bool(creds and creds.access_token and creds.refresh_token)


def _require_client_id(client_id: str | None) -> str:
    value = (
        client_id
        or EnvVarLoader.get_str(
            "QWENPAW_OPENAI_CODEX_CLIENT_ID",
            "app_EMoamEEZ73f0CkXaXp7hrann",
        )
    ).strip()
    if not value:
        raise OpenAICodexAuthError(
            "OpenAI Codex OAuth requires QWENPAW_OPENAI_CODEX_CLIENT_ID. "
            "Use a QwenPaw-owned client id; do not reuse another client.",
        )
    return value


def _response_json(response: Any) -> dict[str, Any]:
    if hasattr(response, "json"):
        parsed = response.json()
        if isinstance(parsed, dict):
            return parsed
    text = str(getattr(response, "text", "") or "")
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _ensure_success(response: Any, prefix: str) -> dict[str, Any]:
    status_code = int(getattr(response, "status_code", 0) or 0)
    payload = _response_json(response)
    if 200 <= status_code < 300:
        return payload
    error = payload.get("error") or getattr(response, "text", "")
    description = payload.get("error_description") or ""
    suffix = f"{error} ({description})" if description else str(error)
    raise OpenAICodexAuthError(f"{prefix}: HTTP {status_code} {suffix}".strip())


async def _maybe_sleep(
    sleep: Callable[[float], Any],
    seconds: float,
) -> None:
    result = sleep(seconds)
    if inspect.isawaitable(result):
        await result


def _expires_at_from_payload(payload: dict[str, Any]) -> int:
    try:
        expires_in = int(payload.get("expires_in") or 0)
    except (TypeError, ValueError):
        expires_in = 0
    return int(time.time()) + max(0, expires_in)


async def login_with_device_code(
    *,
    client: Any | None = None,
    client_id: str | None = None,
    on_verification: Callable[[str, str], Any],
    sleep: Callable[[float], Any] = asyncio.sleep,
    timeout_seconds: int = DEFAULT_DEVICE_CODE_TIMEOUT_SECONDS,
) -> OpenAICodexCredentials:
    """Run the OpenAI Codex device-code flow.

    The caller must provide a QwenPaw-owned client id or set
    ``QWENPAW_OPENAI_CODEX_CLIENT_ID``.
    """

    resolved_client_id = _require_client_id(client_id)
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=30.0, trust_env=True)
    try:
        usercode_response = await http.post(
            f"{OPENAI_AUTH_BASE_URL}/api/accounts/deviceauth/usercode",
            headers=build_codex_auth_headers("application/json"),
            json={"client_id": resolved_client_id},
        )
        usercode = _ensure_success(
            usercode_response,
            "OpenAI device code request failed",
        )
        device_auth_id = str(usercode.get("device_auth_id") or "").strip()
        user_code = str(
            usercode.get("user_code") or usercode.get("usercode") or "",
        ).strip()
        if not device_auth_id or not user_code:
            raise OpenAICodexAuthError(
                "OpenAI device code response was missing device_auth_id or "
                "user_code.",
            )

        result = on_verification(OPENAI_CODEX_DEVICE_VERIFY_URL, user_code)
        if inspect.isawaitable(result):
            await result

        try:
            interval = int(usercode.get("interval") or 0)
        except (TypeError, ValueError):
            interval = 0
        interval = max(
            MIN_DEVICE_CODE_INTERVAL_SECONDS,
            interval or DEFAULT_DEVICE_CODE_INTERVAL_SECONDS,
        )
        deadline = time.monotonic() + timeout_seconds

        authorization_code = ""
        code_verifier = ""
        while time.monotonic() < deadline:
            token_response = await http.post(
                f"{OPENAI_AUTH_BASE_URL}/api/accounts/deviceauth/token",
                headers=build_codex_auth_headers("application/json"),
                json={
                    "device_auth_id": device_auth_id,
                    "user_code": user_code,
                },
            )
            status_code = int(getattr(token_response, "status_code", 0) or 0)
            if 200 <= status_code < 300:
                token_payload = _response_json(token_response)
                authorization_code = str(
                    token_payload.get("authorization_code") or "",
                ).strip()
                code_verifier = str(
                    token_payload.get("code_verifier") or "",
                ).strip()
                break
            if status_code not in (403, 404):
                _ensure_success(
                    token_response,
                    "OpenAI device authorization failed",
                )
            await _maybe_sleep(sleep, interval)

        if not authorization_code or not code_verifier:
            raise OpenAICodexAuthError(
                "OpenAI device authorization timed out.",
            )

        exchange_response = await http.post(
            f"{OPENAI_AUTH_BASE_URL}/oauth/token",
            headers=build_codex_auth_headers(
                "application/x-www-form-urlencoded",
            ),
            data={
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": OPENAI_CODEX_DEVICE_CALLBACK_URL,
                "client_id": resolved_client_id,
                "code_verifier": code_verifier,
            },
        )
        exchanged = _ensure_success(
            exchange_response,
            "OpenAI device token exchange failed",
        )
        return _credentials_from_token_payload(exchanged)
    finally:
        if owns_client:
            await http.aclose()


def _pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(48)).decode()
    verifier = verifier.rstrip("=")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return verifier, challenge


def build_browser_oauth_url(
    *,
    client_id: str | None = None,
    redirect_uri: str = DEFAULT_BROWSER_OAUTH_REDIRECT_URI,
    scope: str = "openid profile email offline_access",
    state: str | None = None,
) -> tuple[str, str, str]:
    """Build a browser OAuth URL and return URL, verifier, and state."""

    resolved_client_id = _require_client_id(client_id)
    verifier, challenge = _pkce_pair()
    resolved_state = state or secrets.token_urlsafe(24)
    query = urlencode(
        {
            "response_type": "code",
            "client_id": resolved_client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "state": resolved_state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "codex_cli_simplified_flow": "true",
            "originator": "openclaw",
            "id_token_add_organizations": "true",
        },
    )
    return f"{OPENAI_AUTH_BASE_URL}/oauth/authorize?{query}", verifier, resolved_state


def extract_authorization_code(
    redirect_or_code: str,
    *,
    expected_state: str | None = None,
) -> str:
    """Extract an OAuth authorization code from a pasted URL or raw code."""

    value = redirect_or_code.strip()
    if not value:
        raise OpenAICodexAuthError("Authorization code cannot be empty.")
    parsed = urlparse(value)
    if parsed.query:
        params = parse_qs(parsed.query)
        state = (params.get("state") or [""])[0]
        if expected_state is not None and state != expected_state:
            raise OpenAICodexAuthError("OAuth state mismatch.")
        code = (params.get("code") or [""])[0].strip()
        if not code:
            raise OpenAICodexAuthError("Redirect URL did not contain code.")
        return code
    if expected_state is not None:
        raise OpenAICodexAuthError(
            "Paste the full redirect URL so OAuth state can be verified.",
        )
    return value


def receive_browser_oauth_redirect(
    *,
    expected_state: str,
    host: str = DEFAULT_BROWSER_OAUTH_CALLBACK_HOST,
    port: int = DEFAULT_BROWSER_OAUTH_CALLBACK_PORT,
    timeout_seconds: int = 300,
    on_ready: Callable[[str], Any] | None = None,
) -> str:
    """Receive one browser OAuth redirect on a local callback server."""

    result: dict[str, str] = {}
    error: dict[str, Exception] = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # pylint: disable=invalid-name
            try:
                code = extract_authorization_code(
                    self.path,
                    expected_state=expected_state,
                )
                result["code"] = code
                body = (
                    "<html><body><h1>Login complete</h1>"
                    "<p>You can close this window.</p></body></html>"
                )
                self.send_response(200)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                error["error"] = exc
                body = (
                    "<html><body><h1>Login failed</h1>"
                    "<p>Return to QwenPaw and try again.</p></body></html>"
                )
                self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = HTTPServer((host, port), CallbackHandler)
    server.timeout = 0.25
    actual_url = f"http://{host}:{server.server_port}/auth/callback"
    if on_ready is not None:
        on_ready(actual_url)

    deadline = time.monotonic() + timeout_seconds
    try:
        while time.monotonic() < deadline and "code" not in result and not error:
            server.handle_request()
    finally:
        server.server_close()

    if "code" in result:
        return result["code"]
    if error:
        exc = error["error"]
        if isinstance(exc, OpenAICodexAuthError):
            raise exc
        raise OpenAICodexAuthError(str(exc)) from exc
    raise OpenAICodexAuthError("Timed out waiting for browser OAuth callback.")


async def exchange_browser_oauth_code(
    *,
    code: str,
    code_verifier: str,
    client: Any | None = None,
    client_id: str | None = None,
    redirect_uri: str = DEFAULT_BROWSER_OAUTH_REDIRECT_URI,
) -> OpenAICodexCredentials:
    """Exchange a browser OAuth authorization code for credentials."""

    resolved_client_id = _require_client_id(client_id)
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=30.0, trust_env=True)
    try:
        response = await http.post(
            f"{OPENAI_AUTH_BASE_URL}/oauth/token",
            headers=build_codex_auth_headers(
                "application/x-www-form-urlencoded",
            ),
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": resolved_client_id,
                "code_verifier": code_verifier,
            },
        )
        return _credentials_from_token_payload(
            _ensure_success(response, "OpenAI OAuth token exchange failed"),
        )
    finally:
        if owns_client:
            await http.aclose()


async def refresh_codex_credentials(
    *,
    client: Any | None = None,
    client_id: str | None = None,
    creds: OpenAICodexCredentials | None = None,
) -> OpenAICodexCredentials:
    """Refresh stored Codex OAuth credentials."""

    resolved = creds or load_codex_credentials()
    if resolved is None:
        raise OpenAICodexAuthError("OpenAI Codex is not logged in.")
    resolved_client_id = _require_client_id(client_id)
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=30.0, trust_env=True)
    try:
        response = await http.post(
            f"{OPENAI_AUTH_BASE_URL}/oauth/token",
            headers=build_codex_auth_headers(
                "application/x-www-form-urlencoded",
            ),
            data={
                "grant_type": "refresh_token",
                "refresh_token": resolved.refresh_token,
                "client_id": resolved_client_id,
            },
        )
        refreshed = _credentials_from_token_payload(
            _ensure_success(response, "OpenAI OAuth token refresh failed"),
            fallback_refresh_token=resolved.refresh_token,
            email=resolved.email,
        )
        save_codex_credentials(refreshed)
        return refreshed
    finally:
        if owns_client:
            await http.aclose()


def refresh_codex_credentials_sync(
    *,
    client: Any | None = None,
    client_id: str | None = None,
    creds: OpenAICodexCredentials | None = None,
) -> OpenAICodexCredentials:
    """Synchronously refresh stored Codex OAuth credentials."""

    resolved = creds or load_codex_credentials()
    if resolved is None:
        raise OpenAICodexAuthError("OpenAI Codex is not logged in.")
    resolved_client_id = _require_client_id(client_id)
    owns_client = client is None
    http = client or httpx.Client(timeout=30.0, trust_env=True)
    try:
        response = http.post(
            f"{OPENAI_AUTH_BASE_URL}/oauth/token",
            headers=build_codex_auth_headers(
                "application/x-www-form-urlencoded",
            ),
            data={
                "grant_type": "refresh_token",
                "refresh_token": resolved.refresh_token,
                "client_id": resolved_client_id,
            },
        )
        refreshed = _credentials_from_token_payload(
            _ensure_success(response, "OpenAI OAuth token refresh failed"),
            fallback_refresh_token=resolved.refresh_token,
            email=resolved.email,
        )
        save_codex_credentials(refreshed)
        return refreshed
    finally:
        if owns_client:
            http.close()


def _credentials_from_token_payload(
    payload: dict[str, Any],
    *,
    fallback_refresh_token: str = "",
    email: str = "",
) -> OpenAICodexCredentials:
    access = str(payload.get("access_token") or "").strip()
    refresh = str(payload.get("refresh_token") or fallback_refresh_token).strip()
    if not access or not refresh:
        raise OpenAICodexAuthError(
            "OpenAI token response did not include access and refresh tokens.",
        )
    return OpenAICodexCredentials(
        access_token=access,
        refresh_token=refresh,
        expires_at=_expires_at_from_payload(payload),
        email=str(payload.get("email") or email or ""),
    )


def token_needs_refresh(creds: OpenAICodexCredentials, skew_seconds: int = 60) -> bool:
    """Return whether a credential should be refreshed before use."""

    return creds.expires_at <= int(time.time()) + skew_seconds
