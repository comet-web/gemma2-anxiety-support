"""Llama Guard 3 moderation for user queries and bot replies.

The prompt is built explicitly (Meta's documented Llama Guard 3 format) and sent to
vLLM's /v1/completions endpoint. Building it ourselves, instead of relying on the
tokenizer's chat template, keeps the taxonomy under our control and behaves the same
for the 1B and 8B checkpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

# Llama Guard 3 hazard taxonomy (MLCommons). Keys are the codes the model emits.
CATEGORIES: dict[str, str] = {
    "S1": "Violent Crimes.",
    "S2": "Non-Violent Crimes.",
    "S3": "Sex Crimes.",
    "S4": "Child Exploitation.",
    "S5": "Defamation.",
    "S6": "Specialized Advice.",
    "S7": "Privacy.",
    "S8": "Intellectual Property.",
    "S9": "Indiscriminate Weapons.",
    "S10": "Hate.",
    "S11": "Self-Harm.",
    "S12": "Sexual Content.",
    "S13": "Elections.",
    "S14": "Code Interpreter Abuse.",
}
SELF_HARM = "S11"

Role = Literal["User", "Agent"]


@dataclass
class ModerationResult:
    safe: bool
    categories: list[str] = field(default_factory=list)
    raw: str = ""
    error: str | None = None

    @property
    def self_harm(self) -> bool:
        return SELF_HARM in self.categories

    def to_dict(self) -> dict:
        return {
            "safe": self.safe,
            "categories": self.categories,
            "category_names": [CATEGORIES.get(c, c).rstrip(".") for c in self.categories],
            "error": self.error,
        }


def build_prompt(conversation: list[dict[str, str]], role: Role = "User", categories: dict[str, str] = CATEGORIES) -> str:
    """Build the Llama Guard 3 prompt. `conversation` uses user/assistant roles.

    The leading <|begin_of_text|> is omitted on purpose: vLLM adds BOS when it tokenizes
    a completion prompt.
    """
    cats = "\n".join(f"{k}: {v}" for k, v in categories.items())
    turns = "\n\n".join(
        f"{'User' if m['role'] == 'user' else 'Agent'}: {m['content'].strip()}" for m in conversation
    )
    return (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Task: Check if there is unsafe content in '{role}' messages in conversations "
        "according our safety policy with the below categories.\n\n"
        "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
        f"{cats}\n"
        "<END UNSAFE CONTENT CATEGORIES>\n\n"
        "<BEGIN CONVERSATION>\n\n"
        f"{turns}\n\n"
        "<END CONVERSATION>\n\n"
        f"Provide your safety assessment for ONLY THE LAST {role} message in the above conversation:\n"
        " - First line must read 'safe' or 'unsafe'.\n"
        " - If unsafe, a second line must include a comma-separated list of violated categories."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def parse_output(text: str) -> ModerationResult:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ModerationResult(safe=False, raw=text, error="empty moderation output")
    verdict = lines[0].lower()
    if verdict == "safe":
        return ModerationResult(safe=True, raw=text)
    if verdict == "unsafe":
        cats: list[str] = []
        if len(lines) > 1:
            cats = [c.strip().upper() for c in lines[1].split(",") if c.strip()]
        return ModerationResult(safe=False, categories=cats, raw=text)
    return ModerationResult(safe=False, raw=text, error=f"unparseable moderation output: {lines[0][:50]!r}")


class LlamaGuard:
    def __init__(self, llm, model: str, fail_closed: bool = True):
        self._llm = llm  # OpenAICompatLLM pointed at the Llama Guard vLLM server
        self._model = model
        self._fail_closed = fail_closed

    async def moderate(self, conversation: list[dict[str, str]], role: Role = "User") -> ModerationResult:
        prompt = build_prompt(conversation, role=role)
        try:
            text = await self._llm.complete(self._model, prompt, max_tokens=20, temperature=0.0)
        except Exception as exc:  # network / server errors
            logger.exception("Llama Guard call failed")
            return ModerationResult(safe=not self._fail_closed, error=f"guard unavailable: {exc.__class__.__name__}")
        result = parse_output(text)
        if result.error and not self._fail_closed:
            result.safe = True
        return result

    async def check_input(self, user_message: str, history: list[dict[str, str]] | None = None) -> ModerationResult:
        conv = list(history or [])[-4:] + [{"role": "user", "content": user_message}]
        return await self.moderate(conv, role="User")

    async def check_output(self, user_message: str, bot_message: str) -> ModerationResult:
        conv = [{"role": "user", "content": user_message}, {"role": "assistant", "content": bot_message}]
        return await self.moderate(conv, role="Agent")
