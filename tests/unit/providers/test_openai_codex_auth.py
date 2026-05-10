# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import threading
import time
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen
from pathlib import Path

import pytest

from qwenpaw.security.secret_store import is_encrypted


@pytest.fixture(autouse=True)
def _isolated_codex_auth(monkeypatch, tmp_path: Path):
    import qwenpaw.providers.openai_codex_auth as auth
    import qwenpaw.security.secret_store as secret_store

    test_key = bytes.fromhex("ab" * 32)
    monkeypatch.setattr(secret_store, "_cached_master_key", test_key)
    monkeypatch.setattr(secret_store, "_cached_fernet", None)
    monkeypatch.setattr(secret_store, "_get_secret_dir", lambda: tmp_path)
    monkeypatch.setattr(auth, "SECRET_DIR", tmp_path)
    yield


def test_codex_headers_identify_as_openclaw() -> None:
    from qwenpaw.providers.openai_codex_auth import build_codex_auth_headers

    headers = build_codex_auth_headers("application/json")

    assert headers["Content-Type"] == "application/json"
    assert headers["originator"] == "openclaw"
    assert headers["User-Agent"].startswith("openclaw/")
    assert "version" in headers
    assert headers["version"]  # non-empty


def test_codex_credentials_are_encrypted_at_rest() -> None:
    import qwenpaw.providers.openai_codex_auth as auth
    from qwenpaw.providers.openai_codex_auth import (
        OpenAICodexCredentials,
        load_codex_credentials,
        save_codex_credentials,
    )

    creds = OpenAICodexCredentials(
        access_token="access-secret",
        refresh_token="refresh-secret",
        expires_at=1234567890,
        email="user@example.com",
    )

    save_codex_credentials(creds)

    raw = json.loads(
        (Path(auth.SECRET_DIR) / "openai_codex_auth.json").read_text(
            encoding="utf-8",
        ),
    )
    assert is_encrypted(raw["access_token"])
    assert is_encrypted(raw["refresh_token"])
    assert raw["email"] == "user@example.com"
    assert load_codex_credentials() == creds


def test_codex_credentials_reject_unusable_encrypted_tokens(monkeypatch) -> None:
    import qwenpaw.providers.openai_codex_auth as auth
    from qwenpaw.providers.openai_codex_auth import load_codex_credentials

    path = Path(auth.SECRET_DIR) / "openai_codex_auth.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "access_token": "ENC:unreadable-access",
                "refresh_token": "ENC:unreadable-refresh",
                "expires_at": 1234567890,
            },
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(auth, "decrypt", lambda value: value)

    assert load_codex_credentials() is None


def test_extract_authorization_code_requires_redirect_url_when_state_expected() -> None:
    from qwenpaw.providers.openai_codex_auth import (
        OpenAICodexAuthError,
        extract_authorization_code,
    )

    with pytest.raises(OpenAICodexAuthError, match="full redirect URL"):
        extract_authorization_code("raw-auth-code", expected_state="state-1")


def test_browser_oauth_redirect_uses_callback_host() -> None:
    from qwenpaw.providers.openai_codex_auth import (
        DEFAULT_BROWSER_OAUTH_CALLBACK_HOST,
        build_browser_oauth_url,
    )

    url, _, _ = build_browser_oauth_url(client_id="qwenpaw-mytest")
    query = parse_qs(urlparse(url).query)
    redirect_uri = query["redirect_uri"][0]

    assert urlparse(redirect_uri).hostname == DEFAULT_BROWSER_OAUTH_CALLBACK_HOST


def test_receive_browser_oauth_redirect_validates_state() -> None:
    from qwenpaw.providers.openai_codex_auth import receive_browser_oauth_redirect

    result: dict[str, str] = {}
    def run_callback_server():
        try:
            result["code"] = receive_browser_oauth_redirect(
                expected_state="state-1",
                host="127.0.0.1",
                port=0,
                timeout_seconds=5,
                on_ready=lambda url: result.setdefault("url", url),
            )
        except PermissionError as exc:
            result["permission_error"] = str(exc)

    thread = threading.Thread(target=run_callback_server)
    thread.start()
    deadline = time.monotonic() + 5
    while "url" not in result and time.monotonic() < deadline:
        time.sleep(0.01)

    if "permission_error" in result:
        pytest.skip("Local socket bind is not permitted in this sandbox")
    assert "url" in result
    urlopen(f"{result['url']}?code=code-1&state=state-1", timeout=2).read()
    thread.join(timeout=2)

    assert result["code"] == "code-1"


async def test_device_code_login_uses_openclaw_headers_and_polling() -> None:
    from qwenpaw.providers.openai_codex_auth import login_with_device_code

    requests: list[dict] = []

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    class FakeClient:
        async def post(self, url, **kwargs):
            requests.append({"url": url, **kwargs})
            if url.endswith("/api/accounts/deviceauth/usercode"):
                return FakeResponse(
                    200,
                    {
                        "device_auth_id": "device-1",
                        "user_code": "ABCD-EFGH",
                        "interval": 1,
                    },
                )
            if url.endswith("/api/accounts/deviceauth/token"):
                return FakeResponse(
                    200,
                    {
                        "authorization_code": "auth-code",
                        "code_verifier": "verifier",
                    },
                )
            if url.endswith("/oauth/token"):
                return FakeResponse(
                    200,
                    {
                        "access_token": "access",
                        "refresh_token": "refresh",
                        "expires_in": 3600,
                    },
                )
            raise AssertionError(url)

    prompts: list[tuple[str, str]] = []
    creds = await login_with_device_code(
        client=FakeClient(),
        client_id="qwenpaw-mytest",
        on_verification=lambda url, code: prompts.append((url, code)),
        sleep=lambda _: None,
    )

    assert creds.access_token == "access"
    assert creds.refresh_token == "refresh"
    assert prompts == [("https://auth.openai.com/codex/device", "ABCD-EFGH")]
    assert all(
        request["headers"]["originator"] == "openclaw"
        for request in requests
    )
    assert all(
        "version" in request["headers"]
        for request in requests
    )
