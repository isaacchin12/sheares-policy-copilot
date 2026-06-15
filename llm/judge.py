"""
Groundedness judge — checks whether every factual claim in an answer
is supported by the provided retrieval contexts.

Usage:
    from llm.judge import GroundednessJudge

    judge = GroundednessJudge()
    result = await judge.check(
        answer="The curfew is 11 pm on weekdays.",
        contexts=["Sheares Hall curfew policy: residents must return by 11 pm Mon–Thu."],
        query="What is the curfew?",
    )
    # result == {"is_grounded": True, "confidence": 0.95, "reasoning": "..."}
"""

from __future__ import annotations

import json

from llm.client import LLMClient, get_client
from observability.logging_config import get_correlation_id, get_logger

log = get_logger(__name__)

_JUDGE_SYSTEM_PROMPT = """
You are a strict factual-grounding evaluator.

Your job is to assess whether EVERY factual claim in the ANSWER is directly
supported by at least one of the provided CONTEXT passages.

Rules:
- Consider a claim grounded only if it can be traced to explicit text in the
  contexts.  Plausible inferences or common knowledge do NOT count.
- Ignore stylistic or formatting differences between the answer and the context.
- If the answer says "I don't know" or makes no factual claims, treat it as
  grounded (confidence = 1.0).

Respond ONLY with a JSON object — no prose before or after — in this exact shape:
{
  "is_grounded": <true | false>,
  "confidence": <float 0.0–1.0>,
  "reasoning": "<one concise sentence explaining the verdict>"
}
""".strip()


class GroundednessJudge:
    """
    Uses an LLMClient to score whether an answer is grounded in the given contexts.

    Args:
        client: An LLMClient instance.  Defaults to the module-level singleton.
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or get_client()

    async def check(
        self,
        answer: str,
        contexts: list[str],
        query: str,
    ) -> dict:
        """
        Evaluate whether the answer is supported by the contexts.

        Args:
            answer:   The model-generated answer to evaluate.
            contexts: Retrieved passage strings that the answer should be based on.
            query:    The original user question (adds context for the judge).

        Returns:
            A dict with keys:
              - is_grounded (bool)
              - confidence (float, 0.0–1.0)
              - reasoning (str)
        """
        correlation_id = get_correlation_id()
        log.debug(
            "judge.check.start",
            correlation_id=correlation_id,
            n_contexts=len(contexts),
            answer_len=len(answer),
        )

        # Build the user turn sent to the judge model
        ctx_block = "\n\n".join(
            f"[Context {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )
        user_message = (
            f"QUERY:\n{query}\n\n"
            f"CONTEXTS:\n{ctx_block}\n\n"
            f"ANSWER:\n{answer}\n\n"
            "Now evaluate whether every factual claim in the ANSWER is supported "
            "by the CONTEXTS and respond with the JSON object."
        )

        raw: str = await self._client.chat(  # type: ignore[assignment]
            messages=[{"role": "user", "content": user_message}],
            system=_JUDGE_SYSTEM_PROMPT,
            max_tokens=256,
            stream=False,
        )

        verdict = self._parse_verdict(raw)

        log.info(
            "judge.check.done",
            correlation_id=correlation_id,
            is_grounded=verdict["is_grounded"],
            confidence=verdict["confidence"],
            reasoning=verdict["reasoning"],
        )
        return verdict

    # ── Parsing helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_verdict(raw: str) -> dict:
        """
        Attempt to parse the judge's JSON response.

        Falls back to a conservative "not grounded, low confidence" verdict if
        the model returns something unparseable.
        """
        # Strip markdown code fences the model might add (```json ... ```)
        stripped = raw.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            stripped = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        try:
            data = json.loads(stripped)
            return {
                "is_grounded": bool(data.get("is_grounded", False)),
                "confidence": float(data.get("confidence", 0.0)),
                "reasoning": str(data.get("reasoning", "")),
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            log.warning(
                "judge.parse_error",
                raw_response=raw[:200],
                fallback="returning safe defaults",
            )
            return {
                "is_grounded": False,
                "confidence": 0.0,
                "reasoning": "Judge response could not be parsed; defaulting to not grounded.",
            }
