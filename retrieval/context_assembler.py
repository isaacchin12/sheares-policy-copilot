"""Context assembler: deduplicate, truncate, and format chunks for the LLM prompt."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

# Rough token estimation: GPT/Claude tokenisers average ~4 chars per token.
_CHARS_PER_TOKEN = 4

_CHUNK_HEADER_TEMPLATE = "[SOURCE {i}: {source_doc} — {section}]"


class ContextAssembler:
    """Assemble a formatted context string from ranked retrieval chunks.

    Usage
    -----
    assembler = ContextAssembler()
    context_str, used_chunks = assembler.assemble(query, chunks, max_tokens=3000)
    # Pass context_str to the LLM; used_chunks to the citation tracker.
    """

    def assemble(
        self,
        query: str,
        chunks: list[dict],
        max_tokens: int = 3000,
    ) -> tuple[str, list[dict]]:
        """Deduplicate, truncate, and format chunks into an LLM-ready context.

        Parameters
        ----------
        query:
            The original user question (used for logging only).
        chunks:
            Ranked list of chunk dicts.  Each must contain at minimum:
            ``content``, ``source_doc``, ``section``.  May also have
            ``score``, ``rerank_score``, ``chunk_id``, ``subsection``.
        max_tokens:
            Approximate token budget for the assembled context.  Estimated as
            ``len(text) / 4`` (1 token ≈ 4 chars).

        Returns
        -------
        tuple[str, list[dict]]
            ``(context_str, used_chunks)`` where *context_str* is ready to be
            injected into a prompt and *used_chunks* is the ordered list of
            chunk dicts that were included (for citation generation).
        """
        # ------------------------------------------------------------------
        # 1. Deduplicate by content — keep the highest-scored version
        # ------------------------------------------------------------------
        seen_content: dict[str, dict] = {}
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if not content:
                continue
            existing = seen_content.get(content)
            if existing is None:
                seen_content[content] = chunk
            else:
                # Keep whichever has the higher effective score
                current_score = _effective_score(chunk)
                existing_score = _effective_score(existing)
                if current_score > existing_score:
                    seen_content[content] = chunk

        # Re-sort by effective score descending after dedup
        deduped = sorted(
            seen_content.values(),
            key=_effective_score,
            reverse=True,
        )

        # ------------------------------------------------------------------
        # 2. Truncate to max_tokens budget
        # ------------------------------------------------------------------
        max_chars = max_tokens * _CHARS_PER_TOKEN
        used_chunks: list[dict] = []
        total_chars = 0

        for chunk in deduped:
            content = chunk.get("content", "").strip()
            # Rough header cost (~50 chars) + content
            header_cost = 60
            chunk_chars = len(content) + header_cost

            if total_chars + chunk_chars > max_chars and used_chunks:
                # Budget exhausted — stop adding chunks
                break

            used_chunks.append(chunk)
            total_chars += chunk_chars

        # ------------------------------------------------------------------
        # 3. Format each chunk
        # ------------------------------------------------------------------
        parts: list[str] = []
        for i, chunk in enumerate(used_chunks, start=1):
            source_doc = chunk.get("source_doc", "Unknown source")
            section = chunk.get("section", "")
            content = chunk.get("content", "").strip()

            header = _CHUNK_HEADER_TEMPLATE.format(
                i=i,
                source_doc=source_doc,
                section=section,
            )
            parts.append(f"{header}\n{content}")

        context_str = "\n\n".join(parts)

        logger.info(
            "context_assembled",
            query=query[:120],
            n_input_chunks=len(chunks),
            n_deduped=len(deduped),
            n_used=len(used_chunks),
            estimated_tokens=total_chars // _CHARS_PER_TOKEN,
            max_tokens=max_tokens,
        )

        return context_str, used_chunks


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _effective_score(chunk: dict) -> float:
    """Return the best available score for a chunk (rerank > retrieval > 0)."""
    if "rerank_score" in chunk:
        return float(chunk["rerank_score"])
    return float(chunk.get("score", 0.0))
