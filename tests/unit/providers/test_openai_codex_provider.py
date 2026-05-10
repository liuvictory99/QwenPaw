# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

from qwenpaw.cli.providers_cmd import _is_configured
import qwenpaw.providers.provider_manager as provider_manager_module
from qwenpaw.providers.openai_codex_provider import OpenAICodexProvider
from qwenpaw.providers.provider_manager import ProviderManager


def test_openai_codex_builtin_provider_registered(monkeypatch, tmp_path):
    monkeypatch.setattr(
        provider_manager_module,
        "SECRET_DIR",
        tmp_path / ".qwenpaw.secret",
    )

    manager = ProviderManager()
    provider = manager.get_provider("openai-codex")

    assert isinstance(provider, OpenAICodexProvider)
    assert provider.name == "OpenAI Codex"
    assert provider.require_api_key is False
    assert provider.freeze_url is True
    assert provider.support_connection_check is False
    assert [model.id for model in provider.models] == [
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
    ]


def test_openai_codex_provider_uses_bearer_token(monkeypatch):
    import qwenpaw.providers.openai_codex_provider as module

    for key in (
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(key, raising=False)

    provider = OpenAICodexProvider()
    monkeypatch.setattr(
        module,
        "load_codex_credentials",
        lambda: SimpleNamespace(access_token="access-token", expires_at=9999999999),
    )

    client = provider._client(timeout=7)  # pylint: disable=protected-access

    assert client.api_key == "access-token"
    assert str(client.base_url).rstrip("/") == provider.base_url


def test_openai_codex_provider_refreshes_expired_token(monkeypatch):
    import qwenpaw.providers.openai_codex_provider as module

    for key in (
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(key, raising=False)

    provider = OpenAICodexProvider()
    monkeypatch.setattr(
        module,
        "load_codex_credentials",
        lambda: SimpleNamespace(access_token="expired-token", expires_at=1),
    )
    monkeypatch.setattr(
        module,
        "refresh_codex_credentials_sync",
        lambda **_: SimpleNamespace(
            access_token="fresh-token",
            expires_at=9999999999,
        ),
    )

    client = provider._client(timeout=7)  # pylint: disable=protected-access

    assert client.api_key == "fresh-token"


async def test_openai_codex_provider_refreshes_expired_token_without_blocking(
    monkeypatch,
):
    import qwenpaw.providers.openai_codex_provider as module

    provider = OpenAICodexProvider()
    monkeypatch.setattr(
        module,
        "load_codex_credentials",
        lambda: SimpleNamespace(access_token="expired-token", expires_at=1),
    )

    async def fake_refresh(**_):
        return SimpleNamespace(access_token="fresh-token", expires_at=9999999999)

    monkeypatch.setattr(module, "refresh_codex_credentials", fake_refresh)

    assert await provider._access_token_async() == "fresh-token"  # pylint: disable=protected-access


async def test_openai_codex_chat_model_refreshes_token_before_each_call(
    monkeypatch,
):
    from qwenpaw.providers.openai_chat_model_compat import OpenAIChatModelCompat
    from qwenpaw.providers.openai_codex_provider import OpenAICodexChatModelCompat

    seen_tokens: list[str] = []
    tokens = iter(["fresh-1", "fresh-2"])
    for key in (
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(key, raising=False)

    async def fake_parent_call(self, messages, **kwargs):
        seen_tokens.append(self.client.api_key)
        return "ok"

    monkeypatch.setattr(OpenAIChatModelCompat, "__call__", fake_parent_call)
    model = OpenAICodexChatModelCompat(
        model_name="gpt-5.5",
        api_key="initial",
        access_token_resolver=lambda: next(tokens),
        client_kwargs={"base_url": "https://chatgpt.com/backend-api/codex"},
    )

    messages = [{"role": "user", "content": "ping"}]
    assert await model(messages) == "ok"
    assert await model(messages) == "ok"

    assert seen_tokens == ["fresh-1", "fresh-2"]


def test_openai_codex_chat_model_defers_expired_token_refresh(monkeypatch):
    import qwenpaw.providers.openai_codex_provider as module

    for key in (
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(key, raising=False)

    provider = OpenAICodexProvider()
    monkeypatch.setattr(
        module,
        "load_codex_credentials",
        lambda: SimpleNamespace(access_token="expired-token", expires_at=1),
    )
    monkeypatch.setattr(
        module,
        "refresh_codex_credentials_sync",
        lambda **_: (_ for _ in ()).throw(AssertionError("sync refresh")),
    )

    chat_model = provider.get_chat_model_instance("gpt-5.5")

    assert chat_model.client.api_key == "expired-token"


def test_openai_codex_is_not_configured_without_credentials(monkeypatch):
    import qwenpaw.providers.openai_codex_provider as module

    provider = OpenAICodexProvider()
    monkeypatch.setattr(module, "has_codex_credentials", lambda: False)

    assert _is_configured(provider) is False


def test_openai_codex_is_configured_with_credentials(monkeypatch):
    import qwenpaw.providers.openai_codex_provider as module

    provider = OpenAICodexProvider()
    monkeypatch.setattr(module, "has_codex_credentials", lambda: True)

    assert _is_configured(provider) is True


async def test_openai_codex_check_model_connection_uses_streaming_payload(
    monkeypatch,
):
    provider = OpenAICodexProvider()
    captured: list[dict] = []

    class FakeStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.append(kwargs)
            return FakeStream()

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
    )

    async def fake_client_async(timeout=5):
        return fake_client

    monkeypatch.setattr(provider, "_client_async", fake_client_async)

    ok, msg = await provider.check_model_connection("gpt-5.5", timeout=4)

    assert ok is True
    assert msg == ""
    assert captured == [
        {
            "model": "gpt-5.5",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "ping"}],
                },
            ],
            "timeout": 4,
            "max_tokens": 1,
            "stream": True,
        },
    ]
