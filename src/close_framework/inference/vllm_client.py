"""vLLM-specific OpenAI-compatible inference client."""

from __future__ import annotations

from close_framework.config.settings import CloseSettings
from close_framework.inference.llm_client import LLMClient


class VLLMClient(LLMClient):
    """Client for a locally served vLLM endpoint."""

    def __init__(self, settings: CloseSettings) -> None:
        if not settings.llm_base_url:
            settings = type(settings)(**{**settings.__dict__, "llm_base_url": "http://127.0.0.1:8000/v1", "llm_api_key": "dummy"})
        super().__init__(settings)
