"""
LLM factory with DummyLLM fallback for CI and tests.

DummyLLM exists so the full workflow can run without API keys — critical
for CI pipelines and local development. Each agent checks isinstance(llm, DummyLLM)
to short-circuit to deterministic hardcoded responses instead of calling a real model.
"""
from __future__ import annotations

import os
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = structlog.get_logger()


class DummyLLM(BaseChatModel):
    """
    Deterministic no-op LLM for tests and CI environments.

    Returns a fixed string so callers can exercise the full agent/graph
    logic without network calls. Agents detect this class via isinstance
    and substitute hardcoded, realistic-looking responses.
    """

    model_name: str = "dummy"
    default_response: str = "Placeholder response"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(content=self.default_response)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "dummy"


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> BaseChatModel:
    """
    Return the best available LLM given the current environment.

    Falls back to DummyLLM when OPENAI_API_KEY is absent or set to the
    placeholder value from .env.example — keeping CI green without secrets.
    Lazy-imports ChatOpenAI to avoid import cost in test environments.
    """
    _PLACEHOLDER_PREFIXES = ("sk-xxx", "sk-test", "sk-fake", "sk-placeholder", "sk-dummy")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not any(api_key.startswith(p) for p in _PLACEHOLDER_PREFIXES):
        from langchain_openai import ChatOpenAI

        logger.info("llm_initialized", model=model, provider="openai")
        return ChatOpenAI(model=model, temperature=temperature)

    logger.info("llm_initialized", provider="dummy", reason="no_api_key_configured")
    return DummyLLM()
