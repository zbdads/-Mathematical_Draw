"""Shared OpenAI-compatible client helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")


def build_openai_client(
    *,
    base_url: str,
    api_key_env: str,
    fallback_api_key_env: str | None = None,
    timeout: float = 180.0,
    max_retries: int = 2,
) -> OpenAI:
    """Build an OpenAI-compatible client from environment variables."""
    api_key = os.environ.get(api_key_env)
    if not api_key and fallback_api_key_env:
        api_key = os.environ.get(fallback_api_key_env)
    if not api_key:
        env_hint = api_key_env if fallback_api_key_env is None else f"{api_key_env} / {fallback_api_key_env}"
        raise EnvironmentError(f"请在项目根目录 .env 文件中设置 {env_hint}")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
