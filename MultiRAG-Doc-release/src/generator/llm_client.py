"""LLM 调用客户端。

封装 OpenAI 兼容的聊天补全调用。
API Key 从项目根目录 .env 文件读取（LLM_API_KEY）。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from src.config import CFG
from src.openai_compat import build_openai_client


def get_client(*, timeout: float | None = None, max_retries: int | None = None) -> OpenAI:
    if not os.environ.get(CFG.generator.api_key_env):
        raise EnvironmentError(f"请在项目根目录 .env 文件中设置 {CFG.generator.api_key_env}=sk-...")
    return build_openai_client(
        base_url=CFG.generator.base_url,
        api_key_env=CFG.generator.api_key_env,
        fallback_api_key_env="LLM_API_KEY" if CFG.generator.api_key_env != "LLM_API_KEY" else None,
        timeout=timeout if timeout is not None else CFG.generator.timeout_seconds,
        max_retries=max_retries if max_retries is not None else 2,
    )


def _wire_api() -> str:
    return CFG.generator.wire_api.lower().strip()


def _split_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    inputs: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if role == "system":
            instructions.append(str(content))
        else:
            inputs.append({"role": role, "content": str(content)})
    return "\n\n".join(part for part in instructions if part), inputs


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks)


def _responses_kwargs(
    messages: list[dict[str, Any]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str | None = None,
    include_stream_flag: bool = False,
) -> dict[str, Any]:
    instructions, inputs = _split_messages(messages)
    kwargs: dict[str, Any] = {
        "model": model,
        "input": inputs,
        "max_output_tokens": max_tokens,
    }
    if include_stream_flag:
        kwargs["stream"] = True
    if instructions:
        kwargs["instructions"] = instructions
    effective_reasoning_effort = (
        CFG.generator.reasoning_effort if reasoning_effort is None else reasoning_effort
    )
    if effective_reasoning_effort:
        kwargs["reasoning"] = {"effort": effective_reasoning_effort}
    if CFG.generator.disable_response_storage:
        kwargs["store"] = False
    if temperature is not None:
        kwargs["temperature"] = temperature
    return kwargs


def generate(
    messages: list[dict[str, str]],
    model: str = "deepseek-chat",
    temperature: float = 0.0,
    max_tokens: int = 512,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> str:
    """调用 LLM 并返回回答文本。

    Args:
        messages: OpenAI-compatible messages 列表，由 prompt_builder.build_messages() 构造。
        model: 模型名称。
        temperature: 采样温度，0.0 为确定性输出。
        max_tokens: 最大输出 token 数。

    Returns:
        LLM 返回的文本内容。
    """
    client = get_client(timeout=timeout, max_retries=max_retries)
    if _wire_api() == "responses":
        response = client.responses.create(
            **_responses_kwargs(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
        )
        return _response_text(response)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def generate_stream(
    messages: list[dict[str, str]],
    model: str = "deepseek-chat",
    temperature: float = 0.0,
    max_tokens: int = 512,
    on_token: Callable[[str], None] | None = None,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> str:
    """以 stream 方式调用 LLM，逐 chunk 回调 on_token，返回完整文本。

    Args:
        messages: OpenAI-compatible messages 列表。
        model: 模型名称。
        temperature: 采样温度。
        max_tokens: 最大输出 token 数。
        on_token: 每个文本 chunk 到达时的回调，用于实时输出。

    Returns:
        LLM 返回的完整文本内容。
    """
    client = get_client(timeout=timeout, max_retries=max_retries)
    if _wire_api() == "responses":
        chunks: list[str] = []
        with client.responses.stream(
            **_responses_kwargs(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
        ) as stream:
            for event in stream:
                if getattr(event, "type", "") != "response.output_text.delta":
                    continue
                delta = getattr(event, "delta", "")
                if delta:
                    chunks.append(delta)
                    if on_token is not None:
                        on_token(delta)
        return "".join(chunks)

    chunks: list[str] = []
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            chunks.append(delta)
            if on_token is not None:
                on_token(delta)
    return "".join(chunks)
