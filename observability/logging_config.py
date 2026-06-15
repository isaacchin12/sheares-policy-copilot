"""
Shared structured logging setup for sheares-policy-copilot.

Every module imports get_logger() from here.  Each HTTP request is assigned
a correlation_id (UUID4) that is:
  - attached to every log line via structlog's context_var binding
  - set as the Langfuse trace ID for that turn

Usage:
    from observability.logging_config import get_logger, bind_correlation_id

    log = get_logger(__name__)
    bind_correlation_id("abc-123")     # call once at request start
    log.info("retrieval.done", n_chunks=5, latency_ms=42)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path

import structlog

# ── Context variable: correlation/trace ID per request ───────────────────────
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def new_correlation_id() -> str:
    """Generate a new UUID4 correlation ID."""
    return str(uuid.uuid4())


def bind_correlation_id(cid: str | None = None) -> str:
    """
    Bind a correlation ID to the current async context.
    Generates a new UUID if none is provided.
    Also tells structlog to include it in every subsequent log call.
    Returns the id so callers can pass it to Langfuse as the trace ID.
    """
    cid = cid or new_correlation_id()
    _correlation_id.set(cid)
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    return cid


def get_correlation_id() -> str:
    """Return the correlation ID for the current context (empty string if unset)."""
    return _correlation_id.get()


# ── Log level from environment ────────────────────────────────────────────────
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOGS_DIR = Path("logs")
_LOGS_DIR.mkdir(exist_ok=True)


def _add_correlation_id(
    logger: logging.Logger,  # noqa: ARG001
    method: str,             # noqa: ARG001
    event_dict: dict,
) -> dict:
    """structlog processor: inject current correlation_id if not already present."""
    event_dict.setdefault("correlation_id", _correlation_id.get() or "—")
    return event_dict


def configure_logging() -> None:
    """
    Call once at application startup (api/main.py, ingestion/__main__, etc.).
    Sets up:
      - structlog with JSON renderer for files, ConsoleRenderer for terminal
      - A rotating file handler at logs/app.log (10 MB × 5 backups)
      - Standard library `logging` routed through structlog
    """
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        _add_correlation_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # ── Formatter: JSON for file, pretty for stdout ───────────────────────────
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        foreign_pre_chain=shared_processors,
    )

    # ── Handlers ─────────────────────────────────────────────────────────────
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(console_formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        _LOGS_DIR / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(json_formatter)

    root = logging.getLogger()
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    root.setLevel(_LOG_LEVEL)

    # Quieten noisy third-party loggers
    for name in ("httpx", "httpcore", "urllib3", "openai", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to the given module name."""
    return structlog.get_logger(name)
