"""
DocumentLoader — walks a corpus directory and returns raw document dicts.

Supported formats:
  - .md / .txt  : read as UTF-8 text
  - .docx       : extracted via python-docx
  - .pdf        : extracted via pypdf

Skipped:
  - Images / video / audio
  - .env files
  - .git and node_modules directories
"""

from __future__ import annotations

import os
from pathlib import Path

from observability.logging_config import get_logger

log = get_logger(__name__)

# Extensions we know how to read
_TEXT_EXTS = {".md", ".txt"}
_DOCX_EXT = ".docx"
_PDF_EXT = ".pdf"

# Extensions / directory names to skip entirely
_SKIP_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".ico",
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".mp3", ".wav", ".aac", ".flac",
    ".env",
    ".pkl", ".bin", ".pyc",
    ".zip", ".tar", ".gz", ".rar",
}
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache"}


class DocumentLoader:
    """Recursively loads all supported documents from a corpus directory."""

    def load_corpus(self, corpus_path: str) -> list[dict]:
        """Walk *corpus_path* and return a list of document dicts.

        Each dict has:
            {
                "filename": str,   # basename of the file
                "content":  str,   # extracted plain text
                "domain":   str,   # inferred from immediate subdirectory name
            }

        Files that cannot be read are logged and skipped (not raised).
        """
        root = Path(corpus_path).resolve()
        if not root.exists():
            log.warning("loader.corpus_not_found", path=str(root))
            return []

        docs: list[dict] = []

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune directories we never want to descend into (mutate in-place)
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]

            current_dir = Path(dirpath)

            # Infer domain from the immediate child of *root* that we're under.
            # If we're in the root itself, use the root directory name.
            try:
                rel = current_dir.relative_to(root)
                parts = rel.parts
                domain = parts[0] if parts else root.name
            except ValueError:
                domain = root.name

            for filename in filenames:
                file_path = current_dir / filename
                ext = file_path.suffix.lower()

                if ext in _SKIP_EXTS or filename.startswith("."):
                    log.debug("loader.skip", file=str(file_path), reason="extension/hidden")
                    continue

                content: str | None = None

                if ext in _TEXT_EXTS:
                    content = self._read_text(file_path)
                elif ext == _DOCX_EXT:
                    content = self._read_docx(file_path)
                elif ext == _PDF_EXT:
                    content = self._read_pdf(file_path)
                else:
                    log.debug("loader.skip", file=str(file_path), reason="unsupported_extension")
                    continue

                if content is None:
                    continue  # error already logged in the helper

                log.info(
                    "loader.loaded",
                    file=filename,
                    domain=domain,
                    chars=len(content),
                )
                docs.append({
                    "filename": filename,
                    "content": content,
                    "domain": domain,
                })

        log.info("loader.corpus_done", total_docs=len(docs), corpus_path=str(root))
        return docs

    # ── Private helpers ───────────────────────────────────────────────────────

    def _read_text(self, path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            log.error("loader.read_text.error", file=str(path), error=str(exc))
            return None

    def _read_docx(self, path: Path) -> str | None:
        try:
            from docx import Document  # type: ignore[import]
            doc = Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs]
            return "\n".join(paragraphs)
        except ImportError:
            log.error(
                "loader.read_docx.missing_dep",
                file=str(path),
                hint="pip install python-docx",
            )
            return None
        except Exception as exc:
            log.error("loader.read_docx.error", file=str(path), error=str(exc))
            return None

    def _read_pdf(self, path: Path) -> str | None:
        try:
            from pypdf import PdfReader  # type: ignore[import]
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except ImportError:
            log.error(
                "loader.read_pdf.missing_dep",
                file=str(path),
                hint="pip install pypdf",
            )
            return None
        except Exception as exc:
            log.error("loader.read_pdf.error", file=str(path), error=str(exc))
            return None
