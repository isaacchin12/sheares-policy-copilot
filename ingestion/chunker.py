"""
SemanticChunker — splits documents into semantically meaningful chunks.

Strategy (in order of preference):
  1. Heading-based split (lines starting with #, ##, ###).
  2. Paragraph-based split (double newline) grouped into ~500-char chunks.

Post-processing:
  - Chunks smaller than MIN_CHARS (100) are merged with the next chunk.
  - Chunks larger than MAX_CHARS (1500) are split on sentence boundaries.
"""

from __future__ import annotations

import re
import uuid
from typing import Iterator

from observability.logging_config import get_logger

log = get_logger(__name__)

MIN_CHARS = 100
MAX_CHARS = 1500
TARGET_PARAGRAPH_CHARS = 500

# Matches a Markdown heading line (level 1–3)
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)", re.MULTILINE)

# Sentence boundary — period/!/? followed by whitespace or end-of-string
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker:
    """Splits a document dict into a list of chunk dicts."""

    def chunk(self, doc: dict) -> list[dict]:
        """Split *doc["content"]* into semantic chunks.

        Returns a list of dicts::

            {
                "chunk_id":   str,   # UUID4
                "source_doc": str,   # original filename
                "domain":     str,
                "section":    str,   # top-level heading (or "")
                "subsection": str,   # nested heading (or "")
                "content":    str,
                "char_count": int,
            }
        """
        content: str = doc.get("content", "")
        filename: str = doc.get("filename", "")
        domain: str = doc.get("domain", "")

        if not content.strip():
            log.debug("chunker.empty_doc", filename=filename)
            return []

        # Choose split strategy
        heading_positions = list(_HEADING_RE.finditer(content))
        if heading_positions:
            raw_chunks = list(self._split_by_headings(content, heading_positions))
        else:
            raw_chunks = list(self._split_by_paragraphs(content))

        # Post-process: enforce MIN/MAX sizes
        raw_chunks = self._enforce_min(raw_chunks)
        raw_chunks = self._enforce_max(raw_chunks)

        # Build output dicts
        chunks: list[dict] = []
        section = ""
        subsection = ""

        for item in raw_chunks:
            heading_level = item.get("heading_level", 0)
            heading_text = item.get("heading_text", "")
            body = item.get("body", "").strip()

            if not body:
                # Still update tracking headings
                if heading_level == 1:
                    section = heading_text
                    subsection = ""
                elif heading_level in (2, 3):
                    subsection = heading_text
                continue

            # Update section/subsection trackers
            if heading_level == 1:
                section = heading_text
                subsection = ""
            elif heading_level in (2, 3):
                subsection = heading_text

            # Prepend heading to content for context
            header_prefix = ""
            if heading_text:
                prefix_hashes = "#" * heading_level if heading_level else ""
                header_prefix = f"{prefix_hashes} {heading_text}\n\n" if prefix_hashes else ""

            full_content = (header_prefix + body).strip()

            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "source_doc": filename,
                "domain": domain,
                "section": section,
                "subsection": subsection,
                "content": full_content,
                "char_count": len(full_content),
            })

        log.info(
            "chunker.done",
            filename=filename,
            strategy="heading" if heading_positions else "paragraph",
            chunk_count=len(chunks),
        )
        return chunks

    # ── Splitting strategies ──────────────────────────────────────────────────

    def _split_by_headings(
        self, content: str, heading_positions: list[re.Match]
    ) -> Iterator[dict]:
        """Yield raw chunk dicts split at every heading."""
        # Collect start positions of each heading match
        boundaries = [(m.start(), m) for m in heading_positions]

        for i, (start, match) in enumerate(boundaries):
            # Body text runs from end of this heading line to start of next
            body_start = match.end()
            body_end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(content)

            level = len(match.group(1))  # number of # chars
            heading_text = match.group(2).strip()
            body = content[body_start:body_end]

            yield {
                "heading_level": level,
                "heading_text": heading_text,
                "body": body,
            }

        # Text before the first heading (preamble)
        if boundaries:
            preamble = content[: boundaries[0][0]].strip()
            if preamble:
                yield {"heading_level": 0, "heading_text": "", "body": preamble}

    def _split_by_paragraphs(self, content: str) -> Iterator[dict]:
        """Group double-newline paragraphs into ~TARGET_PARAGRAPH_CHARS chunks."""
        paragraphs = [p.strip() for p in re.split(r"\n\n+", content) if p.strip()]

        buffer: list[str] = []
        buffer_len = 0

        for para in paragraphs:
            para_len = len(para)
            if buffer and buffer_len + para_len > TARGET_PARAGRAPH_CHARS:
                yield {"heading_level": 0, "heading_text": "", "body": "\n\n".join(buffer)}
                buffer = [para]
                buffer_len = para_len
            else:
                buffer.append(para)
                buffer_len += para_len

        if buffer:
            yield {"heading_level": 0, "heading_text": "", "body": "\n\n".join(buffer)}

    # ── Post-processing ───────────────────────────────────────────────────────

    def _enforce_min(self, chunks: list[dict]) -> list[dict]:
        """Merge chunks shorter than MIN_CHARS into the following chunk."""
        result: list[dict] = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            body = chunk.get("body", "").strip()
            if len(body) < MIN_CHARS and i + 1 < len(chunks):
                # Merge into next
                next_chunk = chunks[i + 1]
                merged_body = body + "\n\n" + next_chunk.get("body", "").strip()
                chunks[i + 1] = {**next_chunk, "body": merged_body}
                i += 1
                continue
            result.append(chunk)
            i += 1
        return result

    def _enforce_max(self, chunks: list[dict]) -> list[dict]:
        """Split chunks longer than MAX_CHARS on sentence boundaries."""
        result: list[dict] = []
        for chunk in chunks:
            body = chunk.get("body", "").strip()
            if len(body) <= MAX_CHARS:
                result.append(chunk)
                continue
            # Split on sentence boundaries
            sub_chunks = self._split_on_sentences(body, chunk)
            result.extend(sub_chunks)
        return result

    def _split_on_sentences(self, body: str, template: dict) -> list[dict]:
        """Split *body* into pieces no larger than MAX_CHARS at sentence boundaries."""
        sentences = _SENTENCE_SPLIT_RE.split(body)
        pieces: list[dict] = []
        buffer: list[str] = []
        buffer_len = 0

        for sentence in sentences:
            slen = len(sentence)
            if buffer and buffer_len + slen + 1 > MAX_CHARS:
                pieces.append({**template, "body": " ".join(buffer)})
                buffer = [sentence]
                buffer_len = slen
            else:
                buffer.append(sentence)
                buffer_len += slen + 1  # +1 for the space re-join

        if buffer:
            pieces.append({**template, "body": " ".join(buffer)})

        return pieces
