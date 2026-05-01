"""
Centralized model registry — single source of truth for model metadata.

Adding a new model requires one dict entry here. The gateway, cost tracker,
and rate limiter all read from this registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ModelInfo:
    """Immutable metadata for a single LLM model."""

    provider: str
    tier: int  # 1 = local/cheap, 2 = mid, 3 = premium
    cost_per_1k_input: Decimal
    cost_per_1k_output: Decimal
    default_max_tokens: int = 4096
    rpm: int = 60
    tpm: int = 40_000


MODEL_REGISTRY: dict[str, ModelInfo] = {
    # ── Tier 1: Local / free ──────────────────────────────────────────────
    "dummy": ModelInfo(
        provider="dummy",
        tier=0,
        cost_per_1k_input=Decimal("0"),
        cost_per_1k_output=Decimal("0"),
        rpm=10_000,
        tpm=10_000_000,
    ),
    "llama3.1:8b": ModelInfo(
        provider="ollama",
        tier=1,
        cost_per_1k_input=Decimal("0"),
        cost_per_1k_output=Decimal("0"),
        rpm=120,
        tpm=100_000,
    ),
    # ── Tier 2: Mid-range ─────────────────────────────────────────────────
    "gpt-4o-mini": ModelInfo(
        provider="openai",
        tier=2,
        cost_per_1k_input=Decimal("0.000150"),
        cost_per_1k_output=Decimal("0.000600"),
        rpm=500,
        tpm=200_000,
    ),
    "claude-3-5-haiku": ModelInfo(
        provider="anthropic",
        tier=2,
        cost_per_1k_input=Decimal("0.000800"),
        cost_per_1k_output=Decimal("0.004000"),
        rpm=100,
        tpm=100_000,
    ),
    # ── Tier 3: Premium ───────────────────────────────────────────────────
    "gpt-4o": ModelInfo(
        provider="openai",
        tier=3,
        cost_per_1k_input=Decimal("0.005000"),
        cost_per_1k_output=Decimal("0.015000"),
        rpm=100,
        tpm=30_000,
    ),
    "claude-3-5-sonnet": ModelInfo(
        provider="anthropic",
        tier=3,
        cost_per_1k_input=Decimal("0.003000"),
        cost_per_1k_output=Decimal("0.015000"),
        rpm=60,
        tpm=40_000,
    ),
}


def get_model_info(model: str) -> ModelInfo:
    """Look up model metadata, falling back to a sensible default."""
    return MODEL_REGISTRY.get(model, MODEL_REGISTRY["gpt-4o-mini"])


def resolve_provider(model: str) -> str:
    """Return the provider name for a model."""
    return get_model_info(model).provider


def list_models(provider: str | None = None, max_tier: int | None = None) -> list[str]:
    """List registered model names, optionally filtered."""
    results = []
    for name, info in MODEL_REGISTRY.items():
        if provider and info.provider != provider:
            continue
        if max_tier is not None and info.tier > max_tier:
            continue
        results.append(name)
    return results
