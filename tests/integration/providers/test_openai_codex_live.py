# -*- coding: utf-8 -*-
from __future__ import annotations

import os

import pytest

from qwenpaw.providers.openai_codex_auth import has_codex_credentials
from qwenpaw.providers.openai_codex_provider import OpenAICodexProvider


@pytest.mark.asyncio
async def test_openai_codex_live_model_contract() -> None:
    if os.environ.get("QWENPAW_OPENAI_CODEX_LIVE_TEST") == "0":
        pytest.skip("QWENPAW_OPENAI_CODEX_LIVE_TEST=0 disables live Codex check")
    if not has_codex_credentials():
        pytest.skip("OpenAI Codex credentials are not configured")

    provider = OpenAICodexProvider()
    ok, msg = await provider.check_model_connection("gpt-5.5", timeout=20)

    assert ok, msg
