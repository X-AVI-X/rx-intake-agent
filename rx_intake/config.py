"""Environment-driven configuration and wiring."""

from __future__ import annotations

import os

from .extractors import ClaudeExtractor, Extractor, OllamaExtractor, RulesExtractor
from .formulary import Formulary
from .pipeline import IntakePipeline


def build_extractor(provider: str) -> Extractor:
    if provider == "claude":
        return ClaudeExtractor(
            model=os.getenv("RX_CLAUDE_MODEL", "claude-opus-5-5"),
            timeout_s=float(os.getenv("RX_LLM_TIMEOUT_S", "30")),
        )
    if provider == "ollama":
        return OllamaExtractor(
            model=os.getenv("RX_OLLAMA_MODEL", "qwen2.5-coder:7b"),
            base_url=os.getenv("RX_OLLAMA_URL", "http://localhost:11434"),
            timeout_s=float(os.getenv("RX_LLM_TIMEOUT_S", "120")),
        )
    if provider == "rules":
        return RulesExtractor()
    raise ValueError(f"Unknown provider {provider!r}; use claude, ollama or rules")


def build_pipeline(provider: str | None = None) -> IntakePipeline:
    provider = provider or os.getenv("RX_PROVIDER", "rules")
    primary = build_extractor(provider)
    fallback = RulesExtractor() if provider != "rules" else None
    return IntakePipeline(primary=primary, fallback=fallback, formulary=Formulary.load())
