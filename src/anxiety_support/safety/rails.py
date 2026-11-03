"""The guarded conversation pipeline: NeMo Guardrails orchestrating Llama Guard and the CBT agent.

Request-scoped data (the session and a trace of what each layer decided) is passed to
NeMo actions through a ContextVar, so the API can report which rail handled a turn.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from nemoguardrails import LLMRails, RailsConfig

from ..agent.cbt_agent import CBTAgent
from ..agent.state import Session
from .crisis import crisis_message, detect_crisis
from .llama_guard import LlamaGuard
from .topic import TopicChecker

logger = logging.getLogger(__name__)

_TURN: ContextVar[dict[str, Any]] = ContextVar("turn")

# Routes whose reply is fixed text written by us, so output moderation is skipped.
CANNED_ROUTES = {"crisis", "medical", "medication", "diagnosis", "off_topic", "jailbreak", "blocked_input"}


@dataclass
class TurnResult:
    reply: str
    route: str
    input_moderation: dict | None = None
    output_moderation: dict | None = None
    agent: dict | None = None
    activated_rails: list[str] = field(default_factory=list)


class GuardedPipeline:
    def __init__(self, rails_config_path, guard: LlamaGuard, agent: CBTAgent, main_model: str | None = None,
                 main_base_url: str | None = None, crisis_region: str = "IN",
                 topic_checker: Callable[[str, str], Awaitable[bool]] | None = None):
        config = RailsConfig.from_path(str(rails_config_path))
        for m in config.models:
            if m.type == "main":
                if main_model:
                    m.model = main_model
                if main_base_url:
                    m.parameters["base_url"] = main_base_url
        self.rails = LLMRails(config)
        self.guard = guard
        self.agent = agent
        self.crisis_region = crisis_region
        self.topic_checker = topic_checker or TopicChecker(agent.llm, agent.planner_model)
        self._register_actions()

    # ---------------------------------------------------------------- actions
    def _register_actions(self) -> None:
        self.rails.register_action(self._check_user_input, name="check_user_input")
        self.rails.register_action(self._check_bot_output, name="check_bot_output")
        self.rails.register_action(self._crisis_response, name="crisis_response")
        self.rails.register_action(self._cbt_agent_respond, name="cbt_agent_respond")
        self.rails.register_action(self._set_route, name="set_route")
        self.rails.register_action(self._check_topic, name="check_topic")

    async def _set_route(self, route: str) -> None:
        _TURN.get()["route"] = route

    async def _check_topic(self) -> bool:
        turn = _TURN.get()
        prior = [m["content"] for m in turn["session"].messages[:-1] if m["role"] == "assistant"]
        on_topic = await self.topic_checker(turn["user_message"], prior[-1] if prior else "")
        turn["on_topic"] = on_topic
        return on_topic

    async def _check_user_input(self, context: dict | None = None) -> str:
        turn = _TURN.get()
        text = (context or {}).get("user_message") or turn["user_message"]
        if detect_crisis(text):
            turn.update(route="crisis", crisis_trigger="keyword")
            return "crisis"
        history = turn["session"].messages[:-1]
        result = await self.guard.check_input(text, history)
        turn["input_moderation"] = result.to_dict()
        if result.self_harm:
            turn.update(route="crisis", crisis_trigger="llama_guard_S11")
            return "crisis"
        if not result.safe and "S6" in result.categories and not result.error:
            turn["route"] = "medical"
            return "medical"
        if not result.safe:
            turn["route"] = "blocked_input"
            return "blocked"
        return "allow"

    async def _check_bot_output(self, context: dict | None = None) -> bool:
        turn = _TURN.get()
        if turn.get("route") in CANNED_ROUTES:
            return True
        bot_message = (context or {}).get("bot_message", "")
        result = await self.guard.check_output(turn["user_message"], bot_message)
        turn["output_moderation"] = result.to_dict()
        if not result.safe:
            turn["route"] = "blocked_output"
        return result.safe

    async def _crisis_response(self) -> str:
        turn = _TURN.get()
        turn["route"] = "crisis"
        turn["session"].crisis_flagged = True
        return crisis_message(self.crisis_region)

    async def _cbt_agent_respond(self) -> str:
        turn = _TURN.get()
        result = await self.agent.respond(turn["session"])
        turn["agent"] = {
            "phase": result.phase,
            "steps": [s.__dict__ for s in result.steps],
            "thought_record": turn["session"].record.model_dump(),
        }
        return result.reply

    # ---------------------------------------------------------------- entry point
    async def handle(self, session: Session, user_message: str) -> TurnResult:
        session.messages.append({"role": "user", "content": user_message})
        turn: dict[str, Any] = {"session": session, "user_message": user_message, "route": "unknown"}
        token = _TURN.set(turn)
        try:
            res = await self.rails.generate_async(
                messages=[{"role": "user", "content": user_message}],
                options={"log": {"activated_rails": True}},
            )
        finally:
            _TURN.reset(token)
        reply = res.response[0]["content"] if isinstance(res.response, list) else str(res.response)
        activated = [r.name for r in (res.log.activated_rails if res.log else [])]
        session.messages.append({"role": "assistant", "content": reply})
        return TurnResult(
            reply=reply,
            route=turn.get("route", "unknown"),
            input_moderation=turn.get("input_moderation"),
            output_moderation=turn.get("output_moderation"),
            agent=turn.get("agent"),
            activated_rails=activated,
        )
