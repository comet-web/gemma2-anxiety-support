"""Runtime settings, read from environment variables (or a .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # vLLM server hosting Gemma-2 (base model + the CBT LoRA adapter, multi-LoRA serving)
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "EMPTY"
    # Model used for the CBT reply. With vLLM `--lora-modules gemma-cbt=...` this is the adapter name.
    responder_model: str = "gemma-cbt"
    # Model used for planning / structured extraction (base instruct model is fine here).
    planner_model: str = "google/gemma-2-2b-it"

    # vLLM server hosting Llama Guard 3
    guard_base_url: str = "http://localhost:8001/v1"
    guard_api_key: str = "EMPTY"
    guard_model: str = "meta-llama/Llama-Guard-3-1B"
    # If the moderation model is unreachable: True = treat as unsafe (safe fallback reply).
    guard_fail_closed: bool = True

    # NeMo Guardrails config directory
    rails_config_path: Path = REPO_ROOT / "guardrails"

    # Agent
    agent_max_steps: int = 2
    history_turns: int = 8
    session_ttl_seconds: int = 60 * 60

    # Crisis resources shown to the user (see safety/crisis.py)
    crisis_region: str = "IN"

    request_timeout_s: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
