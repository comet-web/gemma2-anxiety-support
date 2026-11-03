"""Tools the CBT agent can call. Two are LLM-backed (structured outputs), two are curated content."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..prompts import DISTORTION_PROMPT, THOUGHT_RECORD_PROMPT
from .state import Session, ThoughtRecord

# Same labels as the CACTUS dataset's `patterns` field.
DISTORTIONS = [
    "all-or-nothing thinking",
    "overgeneralization",
    "mental filtering",
    "discounting the positive",
    "mind reading",
    "fortune-telling",
    "catastrophizing",
    "emotional reasoning",
    "should statements",
    "labeling",
    "personalization",
]

COPING_EXERCISES = {
    "box_breathing": (
        "Box breathing: breathe in slowly for 4 counts, hold for 4, breathe out for 4, hold for 4. "
        "Repeat 4 rounds, keeping the shoulders relaxed."
    ),
    "grounding_54321": (
        "5-4-3-2-1 grounding: notice 5 things you can see, 4 you can touch, 3 you can hear, "
        "2 you can smell and 1 you can taste. Say them slowly, out loud or in your head."
    ),
    "muscle_relaxation": (
        "Progressive muscle relaxation: tense one muscle group (hands, shoulders, face, legs) for "
        "5 seconds, then release for 10 and notice the difference. Move through the body slowly."
    ),
    "worry_time": (
        "Worry time: write worries down when they come up, then postpone them to a fixed 15-minute "
        "slot later in the day. In that slot, sort them into 'can act on' and 'can't control'."
    ),
}

PSYCHOEDUCATION = {
    "anxiety_cycle": (
        "Anxiety cycle: a trigger leads to an anxious thought, which creates body sensations and an "
        "urge to avoid. Avoiding brings short-term relief but teaches the brain the danger was real, "
        "so the anxiety returns stronger next time."
    ),
    "avoidance": (
        "Avoidance keeps anxiety going because we never get to learn that the feared outcome is less "
        "likely or more manageable than predicted. Small, gradual steps toward the feared situation help."
    ),
    "fight_or_flight": (
        "Fight-or-flight: a racing heart, fast breathing and tense muscles are the body's alarm system. "
        "They feel awful but are not dangerous, and they pass, especially with slow breathing."
    ),
    "thoughts_are_not_facts": (
        "Thoughts are not facts: an automatic thought is a guess the mind makes quickly. We can treat it "
        "as a hypothesis and test it against the evidence."
    ),
}


class Distortion(BaseModel):
    label: str
    evidence: str = Field(max_length=200)


class DistortionResult(BaseModel):
    distortions: list[Distortion] = Field(max_length=3)


def transcript(messages: list[dict[str, str]], last_n: int = 8) -> str:
    who = {"user": "Person", "assistant": "Counselor"}
    return "\n".join(f"{who.get(m['role'], m['role'])}: {m['content']}" for m in messages[-last_n:])


async def detect_distortions(llm, model: str, thought: str) -> list[dict]:
    prompt = DISTORTION_PROMPT.format(taxonomy=", ".join(DISTORTIONS), thought=thought)
    result = await llm.structured(model, [{"role": "user", "content": prompt}], DistortionResult, max_tokens=300)
    if result is None:
        return []
    allowed = set(DISTORTIONS)
    return [d.model_dump() for d in result.distortions if d.label in allowed]


async def update_thought_record(llm, model: str, session: Session) -> ThoughtRecord:
    prompt = THOUGHT_RECORD_PROMPT.format(record=session.record.model_dump_json(), transcript=transcript(session.messages))
    extracted = await llm.structured(model, [{"role": "user", "content": prompt}], ThoughtRecord, max_tokens=400)
    if extracted is not None:
        extracted.distortions = [d for d in extracted.distortions if d in DISTORTIONS]
        session.record.merge(extracted)
    return session.record


def coping_exercise(kind: str) -> str:
    return COPING_EXERCISES.get(kind, COPING_EXERCISES["box_breathing"])


def psychoeducation(topic: str) -> str:
    return PSYCHOEDUCATION.get(topic, PSYCHOEDUCATION["anxiety_cycle"])
