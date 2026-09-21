"""OpenAI-compatible and Ollama-native LLM client for CLOSE Capture."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from close_framework.config.settings import CloseSettings, InferenceMode

LOGGER = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Minimal normalized response from an inference backend."""

    text: str
    model: str
    backend: str
    attempts: int = 1
    error: Optional[str] = None


class LLMClient:
    """Client with retries, Ollama-native chat, and Qwen thinking-off support."""

    def __init__(self, settings: CloseSettings) -> None:
        self.settings = settings
        self.backend = settings.inference_mode.value
        self.model = settings.llm_model or ""
        self._client = None
        self.last_error: Optional[str] = None
        if settings.inference_mode in {
            InferenceMode.OPENAI_API,
            InferenceMode.AZURE_OPENAI,
            InferenceMode.HYBRID_LLM,
            InferenceMode.LOCAL_VLLM,
            InferenceMode.LOCAL_OLLAMA,
        }:
            try:
                from openai import OpenAI

                default_key = "ollama" if settings.inference_mode == InferenceMode.LOCAL_OLLAMA else "EMPTY"
                kwargs: Dict[str, Any] = {"api_key": settings.llm_api_key or default_key}
                if settings.llm_base_url:
                    kwargs["base_url"] = settings.llm_base_url
                elif settings.inference_mode == InferenceMode.LOCAL_OLLAMA:
                    kwargs["base_url"] = "http://127.0.0.1:11434/v1"
                self._client = OpenAI(**kwargs)
            except Exception as exc:  # pragma: no cover
                self.last_error = f"client_init_failed: {exc}"
                self._client = None

    @property
    def available(self) -> bool:
        if self.settings.inference_mode == InferenceMode.LOCAL_OLLAMA:
            return bool(self.model and self._ollama_base())
        return bool(self._client and self.model)

    def _ollama_base(self) -> str:
        base = (self.settings.llm_base_url or "http://127.0.0.1:11434/v1").rstrip("/")
        if base.endswith("/v1"):
            return base[: -len("/v1")]
        return base

    @staticmethod
    def _thinking_disabled() -> bool:
        return os.environ.get("CLOSE_DISABLE_THINKING", "").strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def _message_text(message: Any) -> str:
        """Prefer visible content; fall back to reasoning/thinking payloads."""

        if message is None:
            return ""
        if isinstance(message, dict):
            content = message.get("content")
            reasoning = message.get("reasoning") or message.get("thinking") or message.get("reasoning_content")
        else:
            content = getattr(message, "content", None)
            reasoning = (
                getattr(message, "reasoning", None)
                or getattr(message, "thinking", None)
                or getattr(message, "reasoning_content", None)
            )
            if content is None and hasattr(message, "model_dump"):
                dumped = message.model_dump()
                content = dumped.get("content")
                reasoning = reasoning or dumped.get("reasoning") or dumped.get("thinking")
        text = (content or "").strip()
        if text:
            return text
        return (reasoning or "").strip()

    def _complete_ollama_native(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        disable_thinking: bool,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": int(max_tokens),
            },
        }
        if disable_thinking:
            payload["think"] = False
        req = urllib.request.Request(
            f"{self._ollama_base()}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as handle:
            data = json.loads(handle.read().decode("utf-8"))
        return self._message_text(data.get("message") or {})

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        retries: Optional[int] = None,
    ) -> Optional[LLMResponse]:
        """Run a chat completion with retries; return None on hard failure."""

        self.last_error = None
        if not self.available:
            self.last_error = "llm_unavailable"
            return None

        max_tokens = int(max_tokens or self.settings.llm_max_tokens or 1024)
        retries = int(retries if retries is not None else getattr(self.settings, "llm_retries", 3))
        disable_thinking = self._thinking_disabled()

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        user_prompt = prompt
        if disable_thinking and "qwen" in (self.model or "").lower() and "/no_think" not in prompt:
            user_prompt = f"{prompt.rstrip()}\n\n/no_think"
        messages.append({"role": "user", "content": user_prompt})

        last_err = ""
        for attempt in range(1, retries + 1):
            try:
                if self.settings.inference_mode == InferenceMode.LOCAL_OLLAMA:
                    # Native /api/chat honors think=false; OpenAI-compat often fills
                    # reasoning and leaves content empty for Qwen3.x.
                    text = self._complete_ollama_native(
                        messages,
                        max_tokens=max_tokens,
                        disable_thinking=disable_thinking,
                    )
                else:
                    opts: Dict[str, Any] = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0,
                        "max_tokens": max_tokens,
                    }
                    if disable_thinking:
                        opts["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                    response = self._client.chat.completions.create(**opts)
                    text = self._message_text(response.choices[0].message)

                if not text:
                    last_err = "empty_response"
                    LOGGER.warning("LLM empty response attempt=%s model=%s", attempt, self.model)
                    time.sleep(min(2 * attempt, 6))
                    continue
                return LLMResponse(text=text, model=self.model, backend=self.backend, attempts=attempt)
            except Exception as exc:
                last_err = str(exc)
                msg = last_err.lower()
                if disable_thinking and (
                    "chat_template_kwargs" in msg
                    or "enable_thinking" in msg
                    or "extra_body" in msg
                    or '"think"' in msg
                    or "unknown field" in msg
                ):
                    disable_thinking = False
                    LOGGER.warning("Disabling thinking kwargs after server rejection: %s", exc)
                    continue
                if ("max_tokens" in msg or "context" in msg or "too large" in msg) and max_tokens > 256:
                    max_tokens = max(256, max_tokens // 2)
                    LOGGER.warning("Reducing max_tokens to %s after: %s", max_tokens, exc)
                    continue
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        detail = exc.read().decode("utf-8", errors="replace")[:400]
                        last_err = f"{exc} :: {detail}"
                    except Exception:
                        pass
                LOGGER.warning("LLM attempt %s/%s failed: %s", attempt, retries, last_err)
                time.sleep(min(2 * attempt, 8))

        self.last_error = last_err or "unknown_llm_failure"
        return None

    async def acomplete_many(self, prompts: List[str]) -> List[Optional[LLMResponse]]:
        return await asyncio.gather(*(asyncio.to_thread(self.complete, prompt) for prompt in prompts))
