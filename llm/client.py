"""
Provider-agnostic LLM client for sheares-policy-copilot.

Supports:
  - Anthropic  (chat via AsyncAnthropic,  embeddings via Voyage or stub)
  - Azure OpenAI (chat via AsyncAzureOpenAI, embeddings via text-embedding-3-large)

Usage:
    from llm.client import get_client

    client = get_client()
    text = await client.chat([{"role": "user", "content": "Hello"}])

    # streaming
    async for chunk in await client.chat([...], stream=True):
        print(chunk, end="", flush=True)

    vectors = await client.embed(["some text", "another text"])
"""

from __future__ import annotations

import random
import json
from functools import lru_cache
from typing import AsyncIterator, Union

import anthropic
import openai

from llm.settings import Settings, get_settings
from observability.logging_config import get_logger

log = get_logger(__name__)

# ── Anthropic model to use ────────────────────────────────────────────────────
_ANTHROPIC_CHAT_MODEL = "claude-sonnet-4-6"

# ── Stub embedding dimension (used when Voyage is unavailable) ────────────────
_STUB_EMBED_DIM = 384


class LLMClient:
    """
    Thin wrapper that normalises Anthropic and Azure OpenAI behind one interface.

    Constructor is synchronous; all heavy I/O lives in async methods.
    """

    def __init__(self, settings: Settings) -> None:
        self._cfg = settings

        if settings.llm_provider == "anthropic":
            self._anthropic = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key or None  # None → reads ANTHROPIC_API_KEY
            )
            self._azure: openai.AsyncAzureOpenAI | None = None
            log.info("llm.client.init", provider="anthropic", model=_ANTHROPIC_CHAT_MODEL)

        elif settings.llm_provider == "azure_openai":
            self._anthropic = None  # type: ignore[assignment]
            self._azure = openai.AsyncAzureOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
            )
            log.info(
                "llm.client.init",
                provider="azure_openai",
                chat_deployment=settings.azure_openai_chat_deployment,
                embed_deployment=settings.azure_openai_embedding_deployment,
            )
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")

    # ─────────────────────────────────────────────────────────────────────────
    # Chat
    # ─────────────────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict],
        *,
        system: str = "",
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Send a chat request and return either a full string or an async generator
        of text chunks (when stream=True).

        Args:
            messages:   OpenAI-style list of {"role": ..., "content": ...} dicts.
            system:     Optional system prompt (Anthropic) / injected as system
                        message at index 0 (Azure OpenAI).
            max_tokens: Maximum tokens in the completion.
            stream:     If True, returns an AsyncIterator[str] of text chunks.

        Returns:
            str when stream=False, AsyncIterator[str] when stream=True.
        """
        provider = self._cfg.llm_provider

        if provider == "anthropic":
            return await self._anthropic_chat(messages, system=system, max_tokens=max_tokens, stream=stream)
        else:
            return await self._azure_chat(messages, system=system, max_tokens=max_tokens, stream=stream)

    # ── Anthropic implementation ──────────────────────────────────────────────

    async def _anthropic_chat(
        self,
        messages: list[dict],
        *,
        system: str,
        max_tokens: int,
        stream: bool,
    ) -> Union[str, AsyncIterator[str]]:
        kwargs: dict = dict(
            model=_ANTHROPIC_CHAT_MODEL,
            messages=messages,
            max_tokens=max_tokens,
        )
        if system:
            kwargs["system"] = system

        if stream:
            log.debug("llm.anthropic.stream.start", n_messages=len(messages))
            return self._anthropic_stream_gen(**kwargs)
        else:
            log.debug("llm.anthropic.chat.start", n_messages=len(messages))
            response = await self._anthropic.messages.create(**kwargs)
            text = next(
                (block.text for block in response.content if block.type == "text"),
                "",
            )
            log.info(
                "llm.anthropic.chat.done",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            return text

    async def _anthropic_stream_gen(self, **kwargs) -> AsyncIterator[str]:
        """Async generator that yields text chunks from an Anthropic stream."""
        async with self._anthropic.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
        final = await stream.get_final_message()
        log.info(
            "llm.anthropic.stream.done",
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
        )

    # ── Azure OpenAI implementation ───────────────────────────────────────────

    async def _azure_chat(
        self,
        messages: list[dict],
        *,
        system: str,
        max_tokens: int,
        stream: bool,
    ) -> Union[str, AsyncIterator[str]]:
        # Prepend system message if provided
        az_messages: list[dict] = []
        if system:
            az_messages.append({"role": "system", "content": system})
        az_messages.extend(messages)

        deployment = self._cfg.azure_openai_chat_deployment

        if stream:
            log.debug("llm.azure.stream.start", deployment=deployment, n_messages=len(az_messages))
            return self._azure_stream_gen(az_messages, deployment=deployment, max_tokens=max_tokens)
        else:
            log.debug("llm.azure.chat.start", deployment=deployment, n_messages=len(az_messages))
            response = await self._azure.chat.completions.create(
                model=deployment,
                messages=az_messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            log.info(
                "llm.azure.chat.done",
                deployment=deployment,
                input_tokens=getattr(response.usage, "prompt_tokens", None),
                output_tokens=getattr(response.usage, "completion_tokens", None),
            )
            return text

    async def _azure_stream_gen(
        self,
        messages: list[dict],
        *,
        deployment: str,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Async generator that yields text chunks from an Azure OpenAI stream."""
        stream = await self._azure.chat.completions.create(
            model=deployment,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
        log.info("llm.azure.stream.done", deployment=deployment)

    # ─────────────────────────────────────────────────────────────────────────
    # Embeddings
    # ─────────────────────────────────────────────────────────────────────────

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings and return a list of float vectors.

        Anthropic provider:
          - Uses the `voyageai` package if installed (model voyage-3-lite, 512-dim).
          - Falls back to a STUB that returns random 384-dim vectors seeded by
            text length — suitable only for local development/testing without a
            Voyage API key.  Do NOT use in production.

        Azure OpenAI provider:
          - Uses text-embedding-3-large (1536-dim) via the Azure deployment
            configured in AZURE_OPENAI_EMBEDDING_DEPLOYMENT.
        """
        if self._cfg.llm_provider == "anthropic":
            return await self._anthropic_embed(texts)
        else:
            return await self._azure_embed(texts)

    async def _anthropic_embed(self, texts: list[str]) -> list[list[float]]:
        """
        Attempt Voyage embeddings; fall back to a deterministic random stub.
        The stub is intentionally unseeded-per-text (uses text length as seed)
        so identical strings produce the same vector within a single process,
        which is good enough for smoke-testing retrieval pipelines locally.
        """
        try:
            import voyageai  # type: ignore[import]

            vo = voyageai.AsyncClient()
            result = await vo.embed(texts, model="voyage-3-lite")
            log.info("llm.anthropic.embed.voyage", n=len(texts))
            return result.embeddings

        except ImportError:
            # ── STUB — local dev only; not suitable for production ────────────
            # voyageai package not installed.  Returns random 384-dimensional
            # unit-scale vectors seeded by the length of each text so the same
            # string always maps to the same vector within a single Python session.
            # These vectors carry NO semantic meaning.
            log.warning(
                "llm.anthropic.embed.stub",
                reason="voyageai not installed; returning random 384-dim stub vectors",
                n=len(texts),
            )
            rng = random.Random()
            result_vecs: list[list[float]] = []
            for t in texts:
                rng.seed(len(t))
                result_vecs.append([rng.gauss(0, 1) for _ in range(_STUB_EMBED_DIM)])
            return result_vecs

        except Exception as exc:
            log.error("llm.anthropic.embed.error", error=str(exc))
            raise

    async def _azure_embed(self, texts: list[str]) -> list[list[float]]:
        deployment = self._cfg.azure_openai_embedding_deployment
        log.debug("llm.azure.embed.start", deployment=deployment, n=len(texts))
        response = await self._azure.embeddings.create(
            model=deployment,
            input=texts,
        )
        vecs = [item.embedding for item in response.data]
        log.info("llm.azure.embed.done", deployment=deployment, n=len(vecs), dim=len(vecs[0]) if vecs else 0)
        return vecs


# ─────────────────────────────────────────────────────────────────────────────
# Module-level factory (cached singleton)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_client() -> LLMClient:
    """
    Return the cached LLMClient singleton.

    The first call reads settings from the environment / .env and constructs
    the underlying async HTTP clients.  Subsequent calls return the same object.
    """
    return LLMClient(get_settings())
