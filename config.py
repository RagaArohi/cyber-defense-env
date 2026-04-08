"""
config.py — Configuration loader for ACD-Env

Reads hackathon-required environment variables with correct defaults.

Hackathon spec:
  API_BASE_URL  — has default  (OpenAI-compatible endpoint)
  MODEL_NAME    — has default  (model identifier)
  HF_TOKEN      — NO default   (must be set in HF Space secrets)
  LOCAL_IMAGE_NAME — optional, only for from_docker_image()

Priority: shell env > .env file > coded defaults
"""

import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env parser — no external dependencies."""
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


class Config:
    """All runtime settings resolved once at import time."""

    def __init__(self):
        _load_dotenv(Path(__file__).parent / ".env")

        # ── Hackathon-required vars (exact names) ─────────────────────────────
        self.api_base_url: str       = os.environ.get("API_BASE_URL",
                                           "https://api-inference.huggingface.co/v1/")
        self.model_name: str         = os.environ.get("MODEL_NAME",
                                           "Qwen/Qwen2.5-72B-Instruct")
        self.hf_token: str | None    = os.environ.get("HF_TOKEN") or None
        self.local_image_name: str | None = os.environ.get("LOCAL_IMAGE_NAME") or None

        # ── Server config ─────────────────────────────────────────────────────
        self.host: str      = os.environ.get("HOST", "0.0.0.0")
        self.port: int      = int(os.environ.get("PORT", "7860"))
        self.log_level: str = os.environ.get("LOG_LEVEL", "info").lower()

    @property
    def using_llm(self) -> bool:
        """True when HF_TOKEN is set — enables LLM reasoning."""
        return self.hf_token is not None

    @property
    def llm_provider(self) -> str:
        return "openai-compatible" if self.using_llm else "none"

    @property
    def active_model(self) -> str:
        return self.model_name if self.using_llm else "rule-based"

    def describe(self) -> str:
        if self.using_llm:
            preview = self.hf_token[:8] + "..." if self.hf_token else ""
            return (f"provider=openai-compatible  model={self.model_name}  "
                    f"base_url={self.api_base_url}  token={preview}")
        return ("provider=none  agent=rule-based-fallback  "
                "(set HF_TOKEN in Space secrets to enable LLM)")

    def validate(self) -> None:
        """Print startup banner."""
        print("\n" + "─" * 60)
        print("  ACD-Env — Adaptive Cyber Defense Environment")
        print("─" * 60)
        print(f"  {self.describe()}")
        print(f"  server=http://{self.host}:{self.port}")
        if not self.using_llm:
            print()
            print("  TIP: Set HF_TOKEN in HuggingFace Space secrets to")
            print("  enable LLM reasoning via the OpenAI-compatible API.")
        print("─" * 60 + "\n")


cfg = Config()
