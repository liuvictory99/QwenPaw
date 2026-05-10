# -*- coding: utf-8 -*-
"""OpenAI Codex subscription provider."""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, List

from agentscope.model import ChatModelBase
from openai import AsyncOpenAI

from qwenpaw.constant import EnvVarLoader
from qwenpaw.exceptions import ProviderError
from qwenpaw.providers.openai_codex_auth import (
    has_codex_credentials,
    load_codex_credentials,
    refresh_codex_credentials,
    refresh_codex_credentials_sync,
    token_needs_refresh,
)
from qwenpaw.providers.openai_chat_model_compat import OpenAIChatModelCompat
from qwenpaw.providers.openai_provider import OpenAIProvider
from qwenpaw.providers.provider import ModelInfo

OPENAI_CODEX_BASE_URL = EnvVarLoader.get_str(
    "QWENPAW_OPENAI_CODEX_BASE_URL",
    "https://chatgpt.com/backend-api/codex",
)

OPENAI_CODEX_MODELS: List[ModelInfo] = [
    ModelInfo(
        id="gpt-5.5",
        name="GPT-5.5",
        supports_image=True,
        supports_video=False,
        probe_source="documentation",
    ),
    ModelInfo(
        id="gpt-5.5-pro",
        name="GPT-5.5 Pro",
        supports_image=True,
        supports_video=False,
        probe_source="documentation",
    ),
    ModelInfo(
        id="gpt-5.4",
        name="GPT-5.4",
        supports_image=True,
        supports_video=False,
        probe_source="documentation",
    ),
    ModelInfo(
        id="gpt-5.4-mini",
        name="GPT-5.4 Mini",
        supports_image=True,
        supports_video=False,
        probe_source="documentation",
    ),
]


class OpenAICodexChatModelCompat(OpenAIChatModelCompat):
    """Chat model that refreshes the Codex bearer token before each call."""

    def __init__(
        self,
        *args: Any,
        access_token_resolver: Callable[[], str | Awaitable[str]],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._access_token_resolver = access_token_resolver

    async def __call__(
        self,
        messages: list[dict],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        token = self._access_token_resolver()
        if inspect.isawaitable(token):
            token = await token
        self.client.api_key = token
        return await super().__call__(messages, *args, **kwargs)


class OpenAICodexProvider(OpenAIProvider):
    """Provider that uses locally stored Codex OAuth credentials."""

    def __init__(self, **data):
        data.setdefault("id", "openai-codex")
        data.setdefault("name", "OpenAI Codex")
        data.setdefault("base_url", OPENAI_CODEX_BASE_URL)
        data.setdefault("api_key", "")
        data.setdefault("api_key_prefix", "")
        data.setdefault("models", OPENAI_CODEX_MODELS)
        data.setdefault("freeze_url", True)
        data.setdefault("require_api_key", False)
        data.setdefault("support_model_discovery", False)
        data.setdefault("support_connection_check", False)
        super().__init__(**data)

    def _access_token(self) -> str:
        creds = load_codex_credentials()
        if creds is None or not creds.access_token:
            raise ProviderError(
                "OpenAI Codex is not logged in. Run "
                "`qwenpaw models auth login openai-codex` first.",
            )
        if token_needs_refresh(creds):
            creds = refresh_codex_credentials_sync(creds=creds)
        return creds.access_token

    def _stored_access_token(self) -> str:
        creds = load_codex_credentials()
        if creds is None or not creds.access_token:
            raise ProviderError(
                "OpenAI Codex is not logged in. Run "
                "`qwenpaw models auth login openai-codex` first.",
            )
        return creds.access_token

    async def _access_token_async(self) -> str:
        creds = load_codex_credentials()
        if creds is None or not creds.access_token:
            raise ProviderError(
                "OpenAI Codex is not logged in. Run "
                "`qwenpaw models auth login openai-codex` first.",
            )
        if token_needs_refresh(creds):
            creds = await refresh_codex_credentials(creds=creds)
        return creds.access_token

    def has_credentials(self) -> bool:
        return has_codex_credentials()

    def _client(self, timeout: float = 5) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self._access_token(),
            timeout=timeout,
        )

    async def _client_async(self, timeout: float = 5) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=await self._access_token_async(),
            timeout=timeout,
        )

    async def check_connection(self, timeout: float = 5) -> tuple[bool, str]:
        return (
            False,
            "OpenAI Codex uses model-specific checks; run auth status first.",
        )

    async def fetch_models(self, timeout: float = 5) -> List[ModelInfo]:
        return list(self.models) + list(self.extra_models)

    async def check_model_connection(
        self,
        model_id: str,
        timeout: float = 5,
    ) -> tuple[bool, str]:
        model_id = (model_id or "").strip()
        if not model_id:
            return False, "Empty model ID"

        try:
            client = await self._client_async(timeout=timeout)
            res = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "ping"}],
                    },
                ],
                timeout=timeout,
                max_tokens=1,
                stream=True,
            )
            async for _ in res:
                break
            return True, ""
        except Exception:
            return (
                False,
                f"Unknown exception when connecting to model '{model_id}'",
            )

    def get_chat_model_instance(self, model_id: str) -> ChatModelBase:
        return OpenAICodexChatModelCompat(
            model_name=model_id,
            stream=True,
            api_key=self._stored_access_token(),
            access_token_resolver=self._access_token_async,
            stream_tool_parsing=False,
            client_kwargs={"base_url": self.base_url},
            generate_kwargs=self.get_effective_generate_kwargs(model_id),
        )
