"""
Ingestion orchestration script — run this to build (or rebuild) the vector index.

Usage:
    python -m ingestion.build_index
    # or directly:
    python ingestion/build_index.py

Steps:
  1. Load all documents from the configured corpus directory.
  2. Redact PII from every document.
  3. Chunk each document into semantic pieces.
  4. Embed each chunk via the LLM client.
  5. Upsert embeddings into Qdrant.
  6. Persist a BM25 cache (pickle) for hybrid retrieval.
"""

from __future__ import annotations

import asyncio
import pathlib
import pickle

from ingestion.chunker import SemanticChunker
from ingestion.embedder import Embedder
from ingestion.index_builder import IndexBuilder
from ingestion.loader import DocumentLoader
from ingestion.redaction import get_redactor
from llm.settings import get_settings
from observability.logging_config import configure_logging, get_logger


async def main() -> None:
    configure_logging()
    log = get_logger("ingestion.main")
    settings = get_settings()

    log.info("ingestion.start", corpus_path=settings.corpus_path)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    docs = DocumentLoader().load_corpus(settings.corpus_path)
    log.info("ingestion.loaded", doc_count=len(docs))

    if not docs:
        log.warning("ingestion.no_docs", corpus_path=settings.corpus_path)
        return

    # ── 2. Redact PII ─────────────────────────────────────────────────────────
    redactor = get_redactor()
    for doc in docs:
        doc["content"] = redactor.redact(doc["content"])

    # ── 3. Chunk ──────────────────────────────────────────────────────────────
    chunker = SemanticChunker()
    all_chunks: list[dict] = []
    for doc in docs:
        all_chunks.extend(chunker.chunk(doc))
    log.info("ingestion.chunked", chunk_count=len(all_chunks))

    if not all_chunks:
        log.warning("ingestion.no_chunks")
        return

    # ── 4. Embed ──────────────────────────────────────────────────────────────
    embedder = Embedder()
    chunks_with_embeddings = await embedder.embed_chunks(all_chunks)

    # ── 5. Index into Qdrant ──────────────────────────────────────────────────
    builder = IndexBuilder()
    count = await builder.build(chunks_with_embeddings)
    log.info("ingestion.indexed", indexed=count)

    # ── 6. BM25 cache (pickle) for hybrid retrieval ───────────────────────────
    texts = [c["content"] for c in chunks_with_embeddings]
    metadata = [{k: v for k, v in c.items() if k != "embedding"} for c in chunks_with_embeddings]
    bm25_cache = {"texts": texts, "metadata": metadata}

    cache_path = pathlib.Path("data/bm25_cache.pkl")
    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(bm25_cache, f)
    log.info("ingestion.bm25_cache_saved", path=str(cache_path))


if __name__ == "__main__":
    asyncio.run(main())
