"""Cross-encoder reranker using sentence-transformers ms-marco-MiniLM-L-6-v2."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Rerank retrieval candidates with a cross-encoder model.

    The cross-encoder scores each (query, passage) pair jointly, which is
    more accurate than bi-encoder cosine similarity but slower.  When the
    model cannot be loaded (no sentence-transformers, no GPU, offline, etc.)
    the reranker falls back to returning candidates unchanged.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Defaults to ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model = None
        self._fallback = False
        self._try_load_model(model_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_load_model(self, model_name: str) -> None:
        """Attempt to load the cross-encoder; set fallback flag on failure."""
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]

            self._model = CrossEncoder(model_name)
            logger.info("cross_encoder_loaded", model=model_name)
        except Exception as exc:
            self._fallback = True
            logger.warning(
                "cross_encoder_load_failed",
                model=model_name,
                error=str(exc),
                mode="passthrough",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        """Score candidates and return the top-n by cross-encoder score.

        Parameters
        ----------
        query:
            The user's question.
        candidates:
            List of chunk dicts as returned by :class:`HybridRetriever`.
            Each must contain a ``"content"`` key.
        top_n:
            Number of results to return.

        Returns
        -------
        list[dict]
            Subset of *candidates*, sorted best-first.  Each item gains a
            ``"rerank_score"`` field (float).  In fallback mode the original
            retrieval score is copied into ``rerank_score`` unchanged.
        """
        if not candidates:
            return []

        # Fallback: model unavailable — return top_n as-is with passthrough score
        if self._fallback or self._model is None:
            logger.warning(
                "reranker_fallback_active",
                n_candidates=len(candidates),
                top_n=top_n,
            )
            results = []
            for item in candidates[:top_n]:
                entry = dict(item)
                # Preserve original retrieval score under the rerank key
                entry["rerank_score"] = float(item.get("score", 0.0))
                results.append(entry)
            return results

        # Build (query, passage) pairs for batch inference
        pairs = [(query, item["content"]) for item in candidates]

        try:
            raw_scores: list[float] = self._model.predict(pairs).tolist()
        except Exception as exc:
            logger.warning(
                "cross_encoder_predict_failed",
                error=str(exc),
                mode="passthrough",
            )
            # Graceful degradation on inference error
            results = []
            for item in candidates[:top_n]:
                entry = dict(item)
                entry["rerank_score"] = float(item.get("score", 0.0))
                results.append(entry)
            return results

        # Attach scores and sort descending
        scored: list[tuple[float, dict]] = []
        for score, item in zip(raw_scores, candidates):
            entry = dict(item)
            entry["rerank_score"] = float(score)
            scored.append((float(score), entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [entry for _, entry in scored[:top_n]]

        logger.info(
            "reranking_complete",
            n_candidates=len(candidates),
            top_n=top_n,
            top_rerank_score=round(top[0]["rerank_score"], 4) if top else None,
        )

        return top
