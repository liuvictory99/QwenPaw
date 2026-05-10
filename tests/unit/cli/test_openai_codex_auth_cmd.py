# -*- coding: utf-8 -*-
from __future__ import annotations

from click.testing import CliRunner


def test_browser_login_falls_back_to_manual_redirect_quickly(monkeypatch):
    import qwenpaw.cli.providers_cmd as providers_cmd
    import qwenpaw.providers.openai_codex_auth as auth
    from qwenpaw.providers.openai_codex_auth import (
        OpenAICodexAuthError,
        OpenAICodexCredentials,
    )

    captured: dict[str, object] = {}

    def fake_callback(**kwargs):
        captured["timeout_seconds"] = kwargs.get("timeout_seconds")
        raise OpenAICodexAuthError("callback unavailable")

    async def fake_exchange_browser_oauth_code(**kwargs):
        captured["code"] = kwargs["code"]
        return OpenAICodexCredentials(
            access_token="access",
            refresh_token="refresh",
            expires_at=9999999999,
        )

    monkeypatch.setattr(
        auth,
        "build_browser_oauth_url",
        lambda **_: (
            "http://auth.example/authorize?state=state-1",
            "verifier-1",
            "state-1",
        ),
    )
    monkeypatch.setattr(auth, "receive_browser_oauth_redirect", fake_callback)
    monkeypatch.setattr(
        auth,
        "exchange_browser_oauth_code",
        fake_exchange_browser_oauth_code,
    )
    monkeypatch.setattr(auth, "save_codex_credentials", lambda creds: None)
    monkeypatch.setattr(providers_cmd.click, "launch", lambda url: None)
    monkeypatch.setattr(
        providers_cmd.click,
        "prompt",
        lambda *_, **__: (
            "http://127.0.0.1:1455/auth/callback?code=manual-code&state=state-1"
        ),
    )

    result = CliRunner().invoke(
        providers_cmd.auth_login_cmd,
        ["openai-codex", "--method", "browser"],
    )

    assert result.exit_code == 0, result.output
    assert captured["code"] == "manual-code"
    assert captured["timeout_seconds"] <= 30
