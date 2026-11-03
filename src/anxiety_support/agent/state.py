"""Per-session state: transcript, CBT phase and the evolving thought record.

In-memory with TTL eviction. Swap for Redis to run more than one API replica.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel


class Phase(str, Enum):
    rapport = "rapport"
    explore_situation = "explore_situation"
    identify_thought = "identify_thought"
    examine_distortion = "examine_distortion"
    reframe = "reframe"
    action_plan = "action_plan"
    coping_now = "coping_now"


class ThoughtRecord(BaseModel):
    situation: str | None = None
    automatic_thought: str | None = None
    emotion: str | None = None
    emotion_intensity: int | None = None
    distortions: list[str] = []
    evidence_for: str | None = None
    evidence_against: str | None = None
    balanced_thought: str | None = None

    def merge(self, other: "ThoughtRecord") -> None:
        for name, value in other.model_dump().items():
            if name == "distortions":
                self.distortions = sorted(set(self.distortions) | set(value or []))
            elif value not in (None, ""):
                setattr(self, name, value)

    def summary(self) -> str:
        filled = {k: v for k, v in self.model_dump().items() if v not in (None, "", [])}
        return "; ".join(f"{k}={v}" for k, v in filled.items()) or "(empty)"


@dataclass
class Session:
    id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    phase: Phase = Phase.rapport
    record: ThoughtRecord = field(default_factory=ThoughtRecord)
    crisis_flagged: bool = False
    updated_at: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, ttl_seconds: int = 3600):
        self._ttl = ttl_seconds
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str | None) -> Session:
        self._evict()
        if session_id and session_id in self._sessions:
            s = self._sessions[session_id]
        else:
            s = Session(id=session_id or uuid.uuid4().hex)
            self._sessions[s.id] = s
        s.updated_at = time.time()
        return s

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def _evict(self) -> None:
        cutoff = time.time() - self._ttl
        for sid in [sid for sid, s in self._sessions.items() if s.updated_at < cutoff]:
            del self._sessions[sid]
