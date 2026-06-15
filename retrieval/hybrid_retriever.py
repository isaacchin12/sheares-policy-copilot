"""Hybrid retriever: dense (Qdrant) + sparse (BM25) with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — BM25 and Qdrant are listed in pyproject.toml, but we
# guard gracefully so the module can be imported even in a minimal env.
# ---------------------------------------------------------------------------
try:
    from rank_bm25 import BM25Okapi  # type: ignore[import]

    _BM25_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BM25_AVAILABLE = False
    logger.warning("rank_bm25 not installed — BM25 retrieval disabled")

try:
    from qdrant_client import AsyncQdrantClient  # type: ignore[import]
    from qdrant_client.http import models as qmodels  # type: ignore[import]

    _QDRANT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _QDRANT_AVAILABLE = False
    logger.warning("qdrant_client not installed — dense retrieval disabled")

from llm.settings import get_settings as _get_settings

_settings = _get_settings()
_QDRANT_URL = _settings.qdrant_url
_COLLECTION = _settings.qdrant_collection

# RRF constant — standard value from the original RRF paper
_RRF_K = 60


class HybridRetriever:
    """Retrieve policy chunks using dense + BM25 fusion via RRF.

    Parameters
    ----------
    bm25_cache_path:
        Path to a pickle file produced by the ingestion pipeline containing a
        dict ``{"bm25": BM25Okapi, "metadata": list[dict]}``.  If the file is
        missing the retriever falls back to dense-only mode.
    qdrant_url:
        URL of the Qdrant instance.  Defaults to ``settings.qdrant_url`` or
        ``http://localhost:6333``.
    collection_name:
        Qdrant collection to search.
    """

    def __init__(
        self,
        bm25_cache_path: str = "data/bm25_cache.pkl",
        qdrant_url: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.bm25: "BM25Okapi | None" = None
        self.metadata: list[dict] = []
        self._dense_only = False

        # BM25 setup
        self._try_load_bm25(bm25_cache_path)

        # Qdrant async client
        url = qdrant_url or _QDRANT_URL
        self.collection_name = collection_name or _COLLECTION

        if _QDRANT_AVAILABLE:
            self._qdrant: "AsyncQdrantClient | None" = AsyncQdrantClient(url=url)
        else:
            self._qdrant = None
            logger.warning("qdrant_client unavailable — dense retrieval disabled")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_load_bm25(self, cache_path: str) -> None:
        """Load BM25 index from pickle; log a warning if absent."""
        path = Path(cache_path)
        if not path.exists():
            logger.warning(
                "bm25_cache_not_found",
                path=str(path),
                mode="dense-only",
            )
            return

        if not _BM25_AVAILABLE:
            logger.warning("rank_bm25_missing — skipping BM25 load")
            return

        try:
            with path.open("rb") as fh:
                cache = pickle.load(fh)
            self.metadata = cache["metadata"]

            if "bm25" in cache:
                # Pre-built BM25 object (future format)
                self.bm25 = cache["bm25"]
            elif "texts" in cache:
                # Raw texts from ingestion pipeline — build BM25 here
                texts: list[str] = cache["texts"]
                tokenized = [t.lower().split() for t in texts]
                self.bm25 = BM25Okapi(tokenized)
            else:
                logger.warning("bm25_cache_unknown_format", keys=list(cache.keys()))
                return

            logger.info(
                "bm25_index_loaded",
                n_docs=len(self.metadata),
                path=str(path),
            )
        except Exception as exc:
            logger.warning("bm25_load_failed", error=str(exc), path=str(path))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace tokeniser (matches ingestion pipeline default)."""
        return text.lower().split()

    def _build_qdrant_filter(self, filters: dict) -> "qmodels.Filter | None":
        """Convert a flat ``{field: value}`` dict into a Qdrant Filter."""
        if not filters or not _QDRANT_AVAILABLE:
            return None

        must_conditions: list[Any] = []
        for field, value in filters.items():
            must_conditions.append(
                qmodels.FieldCondition(
                    key=field,
                    match=qmodels.MatchValue(value=value),
                )
            )
        return qmodels.Filter(must=must_conditions)

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------

    async def _dense_retrieve(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None,
    ) -> list[dict]:
        """Return ``top_k`` results from Qdrant ordered by cosine similarity."""
        if self._qdrant is None:
            return []

        qdrant_filter = self._build_qdrant_filter(filters or {})
        hits = await self._qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "chunk_id": str(hit.id),
                    "content": payload.get("content", ""),
                    "source_doc": payload.get("source_doc", ""),
                    "domain": payload.get("domain", ""),
                    "section": payload.get("section", ""),
                    "subsection": payload.get("subsection", ""),
                    "score": hit.score,
                }
            )
        return results

    # ------------------------------------------------------------------
    # BM25 retrieval
    # ------------------------------------------------------------------

    def _bm25_retrieve(self, query: str, top_k: int) -> list[dict]:
        """Return ``top_k`` results from the in-memory BM25 index."""
        if self.bm25 is None or not self.metadata:
            return []

        tokenized_query = self._tokenize(query)
        scores: list[float] = self.bm25.get_scores(tokenized_query).tolist()

        # Pair each score with its index, sort descending
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = scored[:top_k]

        results = []
        for idx, score in top_indices:
            if idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append(
                {
                    "chunk_id": meta.get("chunk_id", str(idx)),
                    "content": meta.get("content", ""),
                    "source_doc": meta.get("source_doc", ""),
                    "domain": meta.get("domain", ""),
                    "section": meta.get("section", ""),
                    "subsection": meta.get("subsection", ""),
                    "score": score,
                }
            )
        return results

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_merge(
        lists: list[list[dict]],
        top_k: int,
        k: int = _RRF_K,
    ) -> list[dict]:
        """Merge ranked lists with Reciprocal Rank Fusion.

        Each result list is assumed to be ordered best-first (rank 0 = best).
        The RRF score for a chunk across all lists is::

            rrf_score = sum(1 / (k + rank_i))

        where ``rank_i`` is the 0-based rank in list *i* (or the chunk is
        absent from that list, contributing 0).
        """
        rrf_scores: dict[str, float] = {}
        # We also keep a representative payload for each chunk_id
        payloads: dict[str, dict] = {}

        for ranked_list in lists:
            for rank, item in enumerate(ranked_list):
                cid = item["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
                if cid not in payloads:
                    payloads[cid] = item

        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for cid, rrf_score in merged:
            entry = dict(payloads[cid])
            entry["score"] = rrf_score
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
        embed_fn: Any = None,
    ) -> list[dict]:
        """Retrieve the top-k most relevant chunks for *query*.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Number of results to return after fusion.
        filters:
            Optional payload filters forwarded to Qdrant as ``must`` conditions.
            Example: ``{"domain": "academic", "section": "appeals"}``.
        embed_fn:
            Async callable ``(text: str) -> list[float]`` that returns a query
            embedding.  Required for dense retrieval.  If ``None`` and Qdrant
            is available, dense results are skipped.

        Returns
        -------
        list[dict]
            Each element contains: ``chunk_id``, ``content``, ``source_doc``,
            ``domain``, ``section``, ``subsection``, ``score``.
        """
        ranked_lists: list[list[dict]] = []

        # 1. Dense retrieval
        if self._qdrant is not None and embed_fn is not None:
            try:
                query_vector = await embed_fn(query)
                dense_results = await self._dense_retrieve(query_vector, top_k, filters)
                if dense_results:
                    ranked_lists.append(dense_results)
            except Exception as exc:
                logger.warning("dense_retrieval_failed", error=str(exc))

        # 2. BM25 retrieval
        if self.bm25 is not None:
            bm25_results = self._bm25_retrieve(query, top_k)
            if bm25_results:
                ranked_lists.append(bm25_results)

        if not ranked_lists:
            logger.warning("no_retrieval_sources_available", query=query)
            return []

        # 3. RRF fusion
        if len(ranked_lists) == 1:
            # Only one source — return it directly (already sorted)
            merged = ranked_lists[0][:top_k]
        else:
            merged = self._rrf_merge(ranked_lists, top_k)

        top_score = merged[0]["score"] if merged else 0.0
        logger.info(
            "retrieval_complete",
            query=query[:120],
            n_results=len(merged),
            top_score=round(top_score, 4),
            sources=[f"list_{i}" for i in range(len(ranked_lists))],
        )

        return merged
