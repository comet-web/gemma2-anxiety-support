"""FastAPI service.

POST /v1/chat            one guarded conversational turn
GET  /v1/sessions/{id}   transcript, CBT phase and thought record
DELETE /v1/sessions/{id} forget a session
GET  /healthz            liveness
GET  /readyz             checks the two vLLM backends are reachable
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ..agent.cbt_agent import CBTAgent
from ..agent.state import SessionStore
from ..config import Settings, get_settings
from ..llm_client import OpenAICompatLLM
from ..safety.llama_guard import LlamaGuard
from ..safety.rails import GuardedPipeline

logger = logging.getLogger("anxiety_support")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DISCLAIMER = (
    "This assistant is a self-help tool based on CBT techniques. It is not a substitute for a "
    "licensed mental-health professional. In an emergency in India call 112 or Tele-MANAS 14416."
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    route: str
    latency_ms: int
    input_moderation: dict | None = None
    output_moderation: dict | None = None
    agent: dict | None = None
    activated_rails: list[str] = []


def build_pipeline(settings: Settings) -> GuardedPipeline:
    llm = OpenAICompatLLM(settings.llm_base_url, settings.llm_api_key, settings.request_timeout_s)
    guard_llm = OpenAICompatLLM(settings.guard_base_url, settings.guard_api_key, settings.request_timeout_s)
    guard = LlamaGuard(guard_llm, settings.guard_model, fail_closed=settings.guard_fail_closed)
    agent = CBTAgent(llm, settings.planner_model, settings.responder_model, settings.agent_max_steps, settings.history_turns)
    return GuardedPipeline(
        settings.rails_config_path,
        guard=guard,
        agent=agent,
        main_model=settings.responder_model,
        main_base_url=settings.llm_base_url,
        crisis_region=settings.crisis_region,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not hasattr(app.state, "pipeline"):  # tests inject their own pipeline
        app.state.pipeline = build_pipeline(settings)
    app.state.sessions = SessionStore(settings.session_ttl_seconds)
    app.state.settings = settings
    yield


app = FastAPI(title="Gemma-2 Anxiety Support (CBT)", version="0.1.0", lifespan=lifespan, description=DISCLAIMER)


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    session = request.app.state.sessions.get_or_create(req.session_id)
    start = time.perf_counter()
    try:
        result = await request.app.state.pipeline.handle(session, req.message.strip())
    except Exception as exc:
        logger.exception("pipeline failed")
        raise HTTPException(status_code=503, detail="The assistant is temporarily unavailable.") from exc
    latency = int((time.perf_counter() - start) * 1000)
    logger.info("session=%s route=%s latency_ms=%d", session.id, result.route, latency)
    return ChatResponse(session_id=session.id, latency_ms=latency, **result.__dict__)


@app.get("/v1/sessions/{session_id}")
async def get_session(session_id: str, request: Request) -> dict:
    s = request.app.state.sessions.get(session_id)
    if s is None:
        raise HTTPException(404, "session not found")
    return {"session_id": s.id, "phase": s.phase.value, "thought_record": s.record.model_dump(),
            "crisis_flagged": s.crisis_flagged, "messages": s.messages}


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict:
    return {"deleted": request.app.state.sessions.delete(session_id)}


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request) -> dict:
    settings: Settings = request.app.state.settings
    status = {}
    async with httpx.AsyncClient(timeout=3) as client:
        for name, url in {"llm": settings.llm_base_url, "guard": settings.guard_base_url}.items():
            try:
                r = await client.get(url.rstrip("/") + "/models")
                status[name] = r.status_code == 200
            except httpx.HTTPError:
                status[name] = False
    if not all(status.values()):
        raise HTTPException(503, detail=status)
    return status
