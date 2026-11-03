"""Thin async client for OpenAI-compatible endpoints (vLLM).

Gemma-2's chat template has no `system` role, so the system prompt is folded into
the first user turn. Training data (training/prepare_data.py) is built the same way,
so the fine-tuned model sees an identical prompt layout at train and inference time.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

Message = dict[str, str]


def fold_system_prompt(messages: list[Message]) -> list[Message]:
    """Merge a leading system message into the first user message (Gemma-2 format)."""
    if not messages or messages[0]["role"] != "system":
        return list(messages)
    system, rest = messages[0]["content"].strip(), list(messages[1:])
    if rest and rest[0]["role"] == "user":
        rest[0] = {"role": "user", "content": f"{system}\n\n{rest[0]['content']}"}
    else:
        rest.insert(0, {"role": "user", "content": system})
    return rest


def merge_consecutive(messages: list[Message]) -> list[Message]:
    """Gemma-2 requires strict user/assistant alternation; merge same-role runs."""
    out: list[Message] = []
    for m in messages:
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append(dict(m))
    return out


class ChatLLM(Protocol):
    async def chat(self, model: str, messages: list[Message], **kwargs: Any) -> str: ...

    async def structured(self, model: str, messages: list[Message], schema: type[BaseModel], **kwargs: Any) -> BaseModel | None: ...


class OpenAICompatLLM:
    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout: float = 60.0):
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)

    async def chat(self, model: str, messages: list[Message], **kwargs: Any) -> str:
        msgs = merge_consecutive(fold_system_prompt(messages))
        resp = await self._client.chat.completions.create(model=model, messages=msgs, **kwargs)
        return (resp.choices[0].message.content or "").strip()

    async def structured(self, model: str, messages: list[Message], schema: type[BaseModel], **kwargs: Any) -> BaseModel | None:
        """JSON output constrained by vLLM structured outputs (guided decoding).

        Returns None instead of raising when the output still fails validation, so
        callers can fall back to a deterministic default.
        """
        kwargs.setdefault("temperature", 0.0)
        content = await self.chat(
            model,
            messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            },
            **kwargs,
        )
        return parse_json_model(content, schema)

    async def complete(self, model: str, prompt: str, **kwargs: Any) -> str:
        resp = await self._client.completions.create(model=model, prompt=prompt, **kwargs)
        return resp.choices[0].text


def parse_json_model(content: str, schema: type[BaseModel]) -> BaseModel | None:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return schema.model_validate(json.loads(text[start : end + 1]))
    except (json.JSONDecodeError, ValidationError):
        return None
