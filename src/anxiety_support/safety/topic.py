"""Second-tier topic steering: one short structured LLM call, run only on messages
that the embedding rails did not match (cost-aware cascade)."""

from __future__ import annotations

import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)

TOPIC_PROMPT = """You screen messages for a CBT-based anxiety support counselor.
on_topic = true if the message is something the counselor should engage with: feelings, stress, worries,
fears, mood, relationships, work or study pressure, sleep, body sensations of anxiety, coping, therapy
exercises, greetings, thanks, or any reply that continues the counseling conversation.
on_topic = false if it asks for an unrelated task or fact: coding, maths, trivia, news, shopping, travel,
writing or translating text, recipes, finance, entertainment and similar.

Counselor's previous message: "{previous}"
User message: "{message}"

Return JSON only."""


class TopicCheck(BaseModel):
    on_topic: bool


class TopicChecker:
    def __init__(self, llm, model: str):
        self.llm = llm
        self.model = model

    async def __call__(self, message: str, previous: str = "") -> bool:
        prompt = TOPIC_PROMPT.format(previous=previous[-300:], message=message)
        try:
            res = await self.llm.structured(self.model, [{"role": "user", "content": prompt}], TopicCheck, max_tokens=20)
        except Exception:
            logger.exception("topic check failed; failing open")
            return True
        return True if res is None else res.on_topic
