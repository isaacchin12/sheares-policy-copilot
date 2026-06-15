"""
IndexBuilder — upserts chunk embeddings into Qdrant.

Creates the collection if it does not exist (vector_size=1536, Cosine distance).
Upserts in batches of 100 to stay within Qdrant payload limits.
"""

from __future__ import annotations

from llm.settings import get_settings
from observability.logging_config import get_logger

log = get_logger(__name__)

_UPSERT_BATCH_SIZE = 100
_DISTANCE = "Cosine"


class IndexBuilder:
    """Manages a Qdrant collection for chunk vectors."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = self._make_client()
        self._collection = self._settings.qdrant_collection

    # ── Public API ────────────────────────────────────────────────────────────

    async def build(self, chunks_with_embeddings: list[dict]) -> int:
        """Upsert *chunks_with_embeddings* into Qdrant.

        Creates the collection if it does not already exist.

        Returns:
            Number of chunks successfully indexed.
        """
        if not chunks_with_embeddings:
            log.warning("index_builder.build.empty")
            return 0

        if self._client is None:
            log.warning("index_builder.qdrant.skipped", reason="Qdrant not available — dense index not built; BM25 cache will still be saved")
            return 0

        # Detect vector dimension from first embedding (supports both 384-stub and 1536-azure)
        first_embedding = next(
            (c["embedding"] for c in chunks_with_embeddings if c.get("embedding")), None
        )
        vector_size = len(first_embedding) if first_embedding else 1536
        self._ensure_collection(vector_size)

        total = len(chunks_with_embeddings)
        indexed = 0

        for batch_start in range(0, total, _UPSERT_BATCH_SIZE):
            batch = chunks_with_embeddings[batch_start : batch_start + _UPSERT_BATCH_SIZE]
            points = self._make_points(batch)

            try:
                self._client.upsert(
                    collection_name=self._collection,
                    points=points,
                    wait=True,
                )
                indexed += len(batch)
                log.info(
                    "index_builder.upsert.done",
                    batch=f"{batch_start + 1}-{batch_start + len(batch)}/{total}",
                    indexed_so_far=indexed,
                )
            except Exception as exc:
                log.error(
                    "index_builder.upsert.error",
                    batch_start=batch_start,
                    error=str(exc),
                )

        log.info("index_builder.build.complete", total_indexed=indexed, collection=self._collection)
        return indexed

    def get_all_texts(self) -> list[str]:
        """Scroll through the entire collection and return all chunk content strings.

        Used to build a BM25 index over the same corpus stored in Qdrant.
        """
        try:
            texts: list[str] = []
            offset = None

            while True:
                results, next_offset = self._client.scroll(
                    collection_name=self._collection,
                    limit=_UPSERT_BATCH_SIZE,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for record in results:
                    content = (record.payload or {}).get("content", "")
                    texts.append(content)

                if next_offset is None:
                    break
                offset = next_offset

            log.info("index_builder.get_all_texts.done", count=len(texts))
            return texts

        except Exception as exc:
            log.error("index_builder.get_all_texts.error", error=str(exc))
            return []

    # ── Private helpers ───────────────────────────────────────────────────────

    def _make_client(self):
        """Construct a synchronous Qdrant client (None if unavailable)."""
        try:
            from qdrant_client import QdrantClient  # type: ignore[import]
            client = QdrantClient(url=self._settings.qdrant_url, timeout=3)
            log.info("index_builder.qdrant.connected", url=self._settings.qdrant_url)
            return client
        except ImportError:
            log.warning("index_builder.qdrant.missing_dep", hint="pip install qdrant-client")
            return None
        except Exception as exc:
            log.warning("index_builder.qdrant.unavailable", error=str(exc))
            return None

    def _ensure_collection(self, vector_size: int = 1536) -> None:
        """Create the Qdrant collection if it does not already exist."""
        if self._client is None:
            return
        from qdrant_client.models import Distance, VectorParams  # type: ignore[import]

        try:
            existing = {c.name for c in self._client.get_collections().collections}
        except Exception as exc:
            log.warning("index_builder.collection.check_failed", error=str(exc))
            return
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            log.info(
                "index_builder.collection.created",
                collection=self._collection,
                vector_size=vector_size,
                distance=_DISTANCE,
            )
        else:
            log.debug("index_builder.collection.exists", collection=self._collection)

    def _make_points(self, batch: list[dict]) -> list:
        """Convert chunk dicts to Qdrant PointStruct objects."""
        from qdrant_client.models import PointStruct  # type: ignore[import]
        import uuid as _uuid

        points = []
        for chunk in batch:
            embedding = chunk.get("embedding")
            if not embedding:
                log.warning("index_builder.missing_embedding", chunk_id=chunk.get("chunk_id"))
                continue

            # Use the chunk_id (UUID4 string) as Qdrant point ID
            chunk_id = chunk.get("chunk_id") or str(_uuid.uuid4())

            # Payload: everything except the embedding
            payload = {k: v for k, v in chunk.items() if k != "embedding"}

            points.append(
                PointStruct(
                    id=str(chunk_id),
                    vector=embedding,
                    payload=payload,
                )
            )

        return points
