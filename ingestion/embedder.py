"""
Embedder — adds vector embeddings to chunk dicts via the LLM client.

Batches requests in groups of 20 to stay within API rate limits.
Falls back to a deterministic stub embedding (numpy random seeded by
content hash) if the LLM embed call fails, so the pipeline never crashes
in local/offline development.
"""

from __future__ import annotations

import hashlib

from llm.client import get_client
from observability.logging_config import get_logger

log = get_logger(__name__)

_BATCH_SIZE = 20
_STUB_DIM = 1536  # matches text-embedding-3-large / voyage dimensions expected by Qdrant


class Embedder:
    """Enriches chunk dicts with an ``"embedding"`` field (list[float])."""

    async def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """Embed all *chunks* and return new dicts with ``"embedding"`` added.

        Processes chunks in batches of ``_BATCH_SIZE``.  If the LLM client
        raises for a batch, every chunk in that batch falls back to a
        deterministic stub vector so the rest of the pipeline can continue.
        """
        client = get_client()
        enriched: list[dict] = []
        total = len(chunks)

        for batch_start in range(0, total, _BATCH_SIZE):
            batch = chunks[batch_start : batch_start + _BATCH_SIZE]
            batch_end = min(batch_start + _BATCH_SIZE, total)
            log.info(
                "embedder.batch.start",
                batch=f"{batch_start + 1}-{batch_end}/{total}",
            )

            texts = [c["content"] for c in batch]
            vectors: list[list[float]] | None = None

            try:
                vectors = await client.embed(texts)
                log.info(
                    "embedder.batch.done",
                    batch=f"{batch_start + 1}-{batch_end}/{total}",
                    dim=len(vectors[0]) if vectors else 0,
                )
            except Exception as exc:
                log.warning(
                    "embedder.batch.fallback",
                    batch=f"{batch_start + 1}-{batch_end}/{total}",
                    error=str(exc),
                    reason="LLM embed failed; using deterministic stub vectors",
                )

            for i, chunk in enumerate(batch):
                if vectors is not None and i < len(vectors):
                    embedding = vectors[i]
                else:
                    embedding = self._stub_embedding(chunk["content"])

                enriched.append({**chunk, "embedding": embedding})

        log.info("embedder.complete", total_chunks=len(enriched))
        return enriched

    # ── Stub embedding ────────────────────────────────────────────────────────

    @staticmethod
    def _stub_embedding(content: str) -> list[float]:
        """Return a deterministic pseudo-random vector seeded by content hash.

        The vector has no semantic meaning — it exists only to let the
        pipeline run end-to-end in environments without an embedding API key.
        The same content string always produces the same vector.
        """
        try:
            import numpy as np  # type: ignore[import]

            digest = hashlib.sha256(content.encode("utf-8", errors="replace")).digest()
            seed = int.from_bytes(digest[:4], "big")
            rng = np.random.default_rng(seed)
            vec: list[float] = rng.standard_normal(_STUB_DIM).tolist()
            return vec

        except ImportError:
            # numpy not available — fall back to stdlib random
            import random

            digest = hashlib.sha256(content.encode("utf-8", errors="replace")).digest()
            seed = int.from_bytes(digest[:4], "big")
            rng = random.Random(seed)
            return [rng.gauss(0, 1) for _ in range(_STUB_DIM)]
