"""
Query router for sheares-policy-copilot.

Classifies each incoming query as one of:
  "fast"  — simple factual policy lookup → handled by the vector RAG pipeline only
  "agent" — multi-step, calculation, eligibility, action, or personalised query
             → handed to the LangGraph agent

Two-tier approach:
  Tier 1 — rule-based fast path  (~0 ms, no LLM call)
  Tier 2 — LLM classifier        (only when rules are inconclusive)

Usage:
    from agent.router import QueryRouter

    router = QueryRouter()
    result = await router.classify("What is the guest policy?")
    # {"path": "fast", "method": "rules", "latency_ms": 0.12}
"""

from __future__ import annotations

import re
import time
from typing import Literal

import anthropic

from llm.settings import get_settings
from observability.logging_config import get_logger

log = get_logger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────────────
RoutePath = Literal["fast", "agent"]
RouteMethod = Literal["rules", "llm"]

# ── Tier-1 rule constants ─────────────────────────────────────────────────────

# Simple factual question openers — any of these alone push toward "fast"
_FACTUAL_PREFIXES: tuple[str, ...] = (
    "what is",
    "what are",
    "how much",
    "when is",
    "who is",
    "what does",
    "define",
    "explain",
)

# Signals that the query needs personalisation, calculation, or action.
# Keep these tight — only phrases that clearly imply a *personal* or *action* query.
_AGENT_SIGNALS: tuple[str, ...] = (
    "am i",
    " i have ",
    "how many points do i",
    "calculate my",
    "my subsidy",
    "my points",
    "my appeal",
    "draft ",
    "help me ",
    "escalate",
    "appeal my",
)

# Hard word-count threshold — queries at or below this are almost certainly
# simple factual lookups and can skip the LLM classifier
_SHORT_QUERY_WORDS = 10

# ── Tier-2 LLM classifier prompt ─────────────────────────────────────────────
_CLASSIFIER_SYSTEM = (
    'You are a query classifier. Answer with one word only: "fast" or "agent".\n'
    '"fast" = simple factual lookup (one policy fact, no personalisation, no calculation)\n'
    '"agent" = needs calculation, eligibility check, document drafting, or multi-step reasoning'
)

_CLASSIFIER_MODEL = "claude-haiku-4-5"  # cheapest model — classification needs no reasoning


class QueryRouter:
    """
    Classifies a query into "fast" (RAG only) or "agent" (LangGraph).

    Tier 1 is synchronous and free; Tier 2 calls the Anthropic API only when
    the rule-based pass is inconclusive.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key or None  # None → reads ANTHROPIC_API_KEY env var
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    async def classify(self, query: str) -> dict:
        """
        Classify *query* and return a routing decision dict.

        Returns:
            {
                "path":       "fast" | "agent",
                "method":     "rules" | "llm",
                "latency_ms": float,        # wall-clock ms for this call
            }
        """
        t0 = time.perf_counter()

        # Tier 1 — rule-based (no LLM, ~0 ms)
        rule_result = self._apply_rules(query)
        if rule_result is not None:
            latency_ms = (time.perf_counter() - t0) * 1000
            log.info(
                "router.decision",
                path=rule_result,
                method="rules",
                latency_ms=round(latency_ms, 3),
                query_preview=query[:80],
            )
            return {"path": rule_result, "method": "rules", "latency_ms": latency_ms}

        # Tier 2 — LLM classifier
        llm_result = await self._llm_classify(query)
        latency_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "router.decision",
            path=llm_result,
            method="llm",
            latency_ms=round(latency_ms, 3),
            query_preview=query[:80],
        )
        return {"path": llm_result, "method": "llm", "latency_ms": latency_ms}

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 1 — Rule-based classifier
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_rules(query: str) -> RoutePath | None:
        """
        Apply deterministic rules to the query.

        Returns:
            "fast"  — confident it is a simple factual lookup
            "agent" — confident it needs multi-step reasoning or personalisation
            None    — inconclusive; Tier 2 LLM classifier required
        """
        q_lower = query.lower().strip()

        # 1. Any agent-signal phrase present → always route to agent
        for signal in _AGENT_SIGNALS:
            if signal in q_lower:
                return "agent"

        # 2. Query is very short → lean fast, but only if no agent signal (already checked)
        word_count = len(q_lower.split())
        if word_count <= _SHORT_QUERY_WORDS:
            return "fast"

        # 3. Starts with a known factual opener → fast
        for prefix in _FACTUAL_PREFIXES:
            if q_lower.startswith(prefix):
                return "fast"

        # 4. Inconclusive — escalate to LLM
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 2 — LLM classifier
    # ─────────────────────────────────────────────────────────────────────────

    async def _llm_classify(self, query: str) -> RoutePath:
        """
        Ask the LLM to classify the query as "fast" or "agent".

        Defaults to "agent" on any error so we never under-serve a user.
        """
        try:
            response = await self._client.messages.create(
                model=_CLASSIFIER_MODEL,
                max_tokens=8,  # we only need a single word back
                system=_CLASSIFIER_SYSTEM,
                messages=[{"role": "user", "content": f"Query: {query}"}],
            )
            raw = next(
                (block.text for block in response.content if block.type == "text"),
                "",
            ).strip().lower()

            # Strip punctuation then match
            token = re.sub(r"[^a-z]", "", raw)
            if token == "fast":
                return "fast"
            if token == "agent":
                return "agent"

            # Unexpected response — safe default
            log.warning("router.llm.unexpected_response", raw=raw)
            return "agent"

        except Exception as exc:  # noqa: BLE001
            log.error("router.llm.error", error=str(exc))
            return "agent"
