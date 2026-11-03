"""Agentic CBT session controller.

Each user turn runs a bounded plan -> act -> respond loop:

1. Planner (base Gemma-2, JSON-schema-constrained decoding) picks the CBT phase and
   the next tool, given the transcript, thought record and observations so far.
2. The tool runs: distortion classifier, thought-record extractor, coping exercise or
   psychoeducation snippet. Its output becomes an observation.
3. Steps 1-2 repeat until the planner picks `respond` or `max_steps` is reached.
4. Responder (the QLoRA fine-tuned CBT model) writes the reply, conditioned on the
   phase, thought record and observations.

Every planner or tool failure degrades to "respond without a tool", so the user
always gets a reply.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from ..prompts import COUNSELOR_SYSTEM_PROMPT, PLANNER_PROMPT, RESPONDER_GUIDANCE
from . import tools
from .state import Phase, Session

logger = logging.getLogger(__name__)

ToolName = Literal["detect_distortions", "update_thought_record", "coping_exercise", "psychoeducation", "respond"]


class Plan(BaseModel):
    phase: Phase
    tool: ToolName
    tool_input: str = Field(default="", max_length=300)
    rationale: str = Field(default="", max_length=300)


@dataclass
class AgentStep:
    phase: str
    tool: str
    tool_input: str
    observation: str


@dataclass
class AgentResult:
    reply: str
    phase: str
    steps: list[AgentStep] = field(default_factory=list)


class CBTAgent:
    def __init__(self, llm, planner_model: str, responder_model: str, max_steps: int = 2, history_turns: int = 8):
        self.llm = llm
        self.planner_model = planner_model
        self.responder_model = responder_model
        self.max_steps = max_steps
        self.history_turns = history_turns

    async def plan(self, session: Session, observations: list[str]) -> Plan:
        prompt = PLANNER_PROMPT.format(
            phase=session.phase.value,
            thought_record=session.record.summary(),
            observations="\n".join(observations) or "(none)",
            transcript=tools.transcript(session.messages, self.history_turns),
        )
        try:
            plan = await self.llm.structured(self.planner_model, [{"role": "user", "content": prompt}], Plan, max_tokens=200)
        except Exception:
            logger.exception("planner call failed")
            plan = None
        return plan or Plan(phase=session.phase, tool="respond", rationale="fallback")

    async def run_tool(self, plan: Plan, session: Session) -> str:
        if plan.tool == "detect_distortions":
            thought = plan.tool_input or session.record.automatic_thought or session.messages[-1]["content"]
            found = await tools.detect_distortions(self.llm, self.planner_model, thought)
            if found:
                session.record.merge(type(session.record)(distortions=[d["label"] for d in found]))
                return "Possible distortions: " + "; ".join(f"{d['label']} (\"{d['evidence']}\")" for d in found)
            return "No clear distortion detected."
        if plan.tool == "update_thought_record":
            record = await tools.update_thought_record(self.llm, self.planner_model, session)
            return f"Thought record updated: {record.summary()}"
        if plan.tool == "coping_exercise":
            return "Exercise to offer: " + tools.coping_exercise(plan.tool_input)
        if plan.tool == "psychoeducation":
            return "Explain briefly: " + tools.psychoeducation(plan.tool_input)
        return ""

    async def respond(self, session: Session) -> AgentResult:
        """`session.messages` must already end with the new user message."""
        observations: list[str] = []
        steps: list[AgentStep] = []
        for _ in range(self.max_steps):
            plan = await self.plan(session, observations)
            session.phase = plan.phase
            if plan.tool == "respond":
                break
            try:
                obs = await self.run_tool(plan, session)
            except Exception:
                logger.exception("tool %s failed", plan.tool)
                obs = ""
            steps.append(AgentStep(plan.phase.value, plan.tool, plan.tool_input, obs))
            if obs:
                observations.append(obs)

        guidance = RESPONDER_GUIDANCE.format(
            phase=session.phase.value,
            thought_record=session.record.summary(),
            observations="\n".join(observations),
        )
        messages = [{"role": "system", "content": COUNSELOR_SYSTEM_PROMPT + guidance}]
        messages += session.messages[-self.history_turns :]
        reply = await self.llm.chat(self.responder_model, messages, max_tokens=300, temperature=0.7, top_p=0.9)
        return AgentResult(reply=reply, phase=session.phase.value, steps=steps)
