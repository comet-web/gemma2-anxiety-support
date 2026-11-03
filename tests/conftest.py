import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))


class FakeLLM:
    """Scripted stand-in for OpenAICompatLLM. `structured_queue` items are pydantic-compatible
    dicts (or None to simulate unparseable output); `chat_reply` is returned for free text."""

    def __init__(self, structured_queue=None, chat_reply="reply", completion="safe"):
        self.structured_queue = list(structured_queue or [])
        self.chat_reply = chat_reply
        self.completion = completion
        self.calls = []

    async def chat(self, model, messages, **kwargs):
        self.calls.append(("chat", model, messages))
        return self.chat_reply

    async def structured(self, model, messages, schema, **kwargs):
        self.calls.append(("structured", model, schema.__name__))
        if not self.structured_queue:
            return None
        item = self.structured_queue.pop(0)
        return None if item is None else schema.model_validate(item)

    async def complete(self, model, prompt, **kwargs):
        self.calls.append(("complete", model, prompt))
        if isinstance(self.completion, Exception):
            raise self.completion
        return self.completion


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture
def cactus_row():
    return json.loads((ROOT / "tests" / "fixtures" / "cactus_row.json").read_text(encoding="utf-8"))
