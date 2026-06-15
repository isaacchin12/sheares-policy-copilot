"""
LangGraph StateGraph for sheares-policy-copilot.

Graph topology:
    route ──► retrieve ──► [conditional]
                               ├── fast_rag    ──► groundedness_gate ──► END
                               ├── agent_llm   ──► groundedness_gate ──► END
                               └── refusal                            ──► END

Entry point: run_query(query, history, correlation_id) → dict

Nodes:
    route              — classifies the query as "fast" or "agent"
    retrieve           — hybrid retrieval + cross-encoder rerank + context assembly
    fast_rag           — single LLM call with retrieved context
    agent_llm          — tool-using LLM loop (up to 3 rounds)
    groundedness_gate  — LLM judge; appends disclaimer if not grounded
    refusal            — polite refusal when retrieval yields nothing useful
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from agent.router import QueryRouter
from agent.tools.escalate import EscalationInput, escalate_to_human
from agent.tools.points_eligibility import PointsInput, check_points_eligibility
from agent.tools.subsidy_calc import SubsidyInput, calculate_subsidy
from llm.client import LLMClient, get_client
from llm.judge import GroundednessJudge
from llm.settings import get_settings
from observability.logging_config import bind_correlation_id, get_logger
from retrieval.context_assembler import ContextAssembler
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

log = get_logger(__name__)

# ── Load system prompt once at import time ────────────────────────────────────
_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.txt"
_SYSTEM_PROMPT: str = (
    _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    if _SYSTEM_PROMPT_PATH.exists()
    else "You are the Sheares Hall Policy Copilot."
)

# ── Retrieval quality threshold ───────────────────────────────────────────────
# Cross-encoder (ms-marco-MiniLM) outputs raw logits (typically -10 to +10).
# BM25 scores depend on corpus size and can be 0–30+.
# We refuse only when ALL scores are exactly 0 (no retrieval hits at all).
_MIN_RETRIEVAL_SCORE = 0.0

# ── Max agentic tool-call rounds ─────────────────────────────────────────────
_MAX_TOOL_ROUNDS = 3

# ── Groundedness disclaimer suffix ───────────────────────────────────────────
_DISCLAIMER = (
    "\n\n---\n*Note: This answer could not be fully verified against the "
    "available policy documents. Please confirm with the JCRC before acting.*"
)


# ─────────────────────────────────────────────────────────────────────────────
# State schema
# ─────────────────────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    query: str
    conversation_history: list[dict]
    route: str                  # "fast" | "agent"
    retrieved_chunks: list[dict]
    context_str: str
    tool_results: list[dict]
    draft_answer: str
    is_grounded: bool
    final_answer: str
    citations: list[dict]
    correlation_id: str


# ─────────────────────────────────────────────────────────────────────────────
# Shared singletons (lazy-init on first use)
# ─────────────────────────────────────────────────────────────────────────────

_router: QueryRouter | None = None
_retriever: HybridRetriever | None = None
_reranker: CrossEncoderReranker | None = None
_assembler: ContextAssembler | None = None
_llm: LLMClient | None = None
_judge: GroundednessJudge | None = None


def _get_router() -> QueryRouter:
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def _get_assembler() -> ContextAssembler:
    global _assembler
    if _assembler is None:
        _assembler = ContextAssembler()
    return _assembler


def _get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = get_client()
    return _llm


def _get_judge() -> GroundednessJudge:
    global _judge
    if _judge is None:
        _judge = GroundednessJudge(_get_llm())
    return _judge


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry — maps tool name → (input_model, async_callable)
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_REGISTRY: dict = {
    "check_points_eligibility": (PointsInput, check_points_eligibility),
    "calculate_subsidy": (SubsidyInput, calculate_subsidy),
    "escalate_to_human": (EscalationInput, escalate_to_human),
}

# ── Anthropic tool definitions ────────────────────────────────────────────────
_ANTHROPIC_TOOLS: list[dict] = [
    {
        "name": "check_points_eligibility",
        "description": (
            "Check a Sheares Hall resident's room retention priority tier "
            "based on their activity points total. Returns tier name, "
            "eligibility status, and gap to next tier."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "total_points": {
                    "type": "integer",
                    "description": "The resident's base activity points total.",
                },
                "has_supplementary": {
                    "type": "boolean",
                    "description": "Whether supplementary points apply.",
                    "default": False,
                },
                "supplementary_points": {
                    "type": "integer",
                    "description": "Number of supplementary points to add.",
                    "default": 0,
                },
            },
            "required": ["total_points"],
        },
    },
    {
        "name": "calculate_subsidy",
        "description": (
            "Calculate the exam period accommodation subsidy for a resident. "
            "Returns eligibility, daily rate, and total subsidy amount."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days_staying": {
                    "type": "integer",
                    "description": "Number of days staying during exam period.",
                },
                "points_total": {
                    "type": "integer",
                    "description": "Resident's total activity points.",
                },
                "applied_before_deadline": {
                    "type": "boolean",
                    "description": (
                        "Whether the application was submitted before the "
                        "last Friday of the exam period."
                    ),
                },
            },
            "required": ["days_staying", "points_total", "applied_before_deadline"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Escalate an issue to the JCRC for human follow-up. "
            "Creates a support ticket and returns the ticket ID and contact email. "
            "Use when policy documents do not contain enough information to resolve the issue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "issue_type": {
                    "type": "string",
                    "description": "Short category label, e.g. 'room allocation', 'finance appeal'.",
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of the issue.",
                },
                "resident_query": {
                    "type": "string",
                    "description": "The original question from the resident.",
                },
            },
            "required": ["issue_type", "description", "resident_query"],
        },
    },
]

# ── Azure OpenAI function definitions ─────────────────────────────────────────
_AZURE_FUNCTIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "check_points_eligibility",
            "description": (
                "Check a Sheares Hall resident's room retention priority tier "
                "based on their activity points total."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "total_points": {"type": "integer"},
                    "has_supplementary": {"type": "boolean", "default": False},
                    "supplementary_points": {"type": "integer", "default": 0},
                },
                "required": ["total_points"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_subsidy",
            "description": "Calculate the exam period accommodation subsidy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_staying": {"type": "integer"},
                    "points_total": {"type": "integer"},
                    "applied_before_deadline": {"type": "boolean"},
                },
                "required": ["days_staying", "points_total", "applied_before_deadline"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate an issue to the JCRC for human follow-up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_type": {"type": "string"},
                    "description": {"type": "string"},
                    "resident_query": {"type": "string"},
                },
                "required": ["issue_type", "description", "resident_query"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build citations from used chunks
# ─────────────────────────────────────────────────────────────────────────────


def _build_citations(chunks: list[dict]) -> list[dict]:
    """Convert retrieved chunks into citation objects numbered from 1."""
    citations = []
    for i, chunk in enumerate(chunks, start=1):
        citations.append(
            {
                "number": i,
                "source_doc": chunk.get("source_doc", "Unknown"),
                "section": chunk.get("section", ""),
                "subsection": chunk.get("subsection", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "score": round(
                    float(chunk.get("rerank_score", chunk.get("score", 0.0))), 4
                ),
            }
        )
    return citations


# ─────────────────────────────────────────────────────────────────────────────
# Helper: detect if we should refuse (empty or low-quality retrieval)
# ─────────────────────────────────────────────────────────────────────────────


def _should_refuse(state: AgentState) -> bool:
    """Return True if no chunks were retrieved (empty retrieval = out-of-corpus)."""
    chunks = state.get("retrieved_chunks", [])
    return len(chunks) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: route_node
# ─────────────────────────────────────────────────────────────────────────────


async def route_node(state: AgentState) -> AgentState:
    """Classify the query as 'fast' or 'agent'."""
    router = _get_router()
    result = await router.classify(state["query"])
    log.info(
        "graph.route",
        correlation_id=state.get("correlation_id"),
        path=result["path"],
        method=result["method"],
        latency_ms=round(result["latency_ms"], 2),
    )
    return {**state, "route": result["path"]}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: retrieve_node
# ─────────────────────────────────────────────────────────────────────────────


async def retrieve_node(state: AgentState) -> AgentState:
    """
    Hybrid retrieval → cross-encoder reranking → context assembly.
    Sets: retrieved_chunks, context_str, citations.
    """
    query = state["query"]
    llm = _get_llm()

    # Embed function: takes a single string, returns a flat list[float]
    async def embed_fn(text: str) -> list[float]:
        vecs = await llm.embed([text])
        return vecs[0] if vecs else []

    retriever = _get_retriever()
    reranker = _get_reranker()
    assembler = _get_assembler()

    # 1. Retrieve (hybrid dense + BM25)
    raw_chunks = await retriever.retrieve(query, top_k=10, embed_fn=embed_fn)

    # 2. Rerank with cross-encoder
    reranked = reranker.rerank(query, raw_chunks, top_n=5)

    # 3. Assemble context string
    context_str, used_chunks = assembler.assemble(query, reranked, max_tokens=3000)

    citations = _build_citations(used_chunks)

    log.info(
        "graph.retrieve",
        correlation_id=state.get("correlation_id"),
        n_raw=len(raw_chunks),
        n_reranked=len(reranked),
        n_used=len(used_chunks),
    )

    return {
        **state,
        "retrieved_chunks": used_chunks,
        "context_str": context_str,
        "citations": citations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3a: fast_rag_node
# ─────────────────────────────────────────────────────────────────────────────


async def fast_rag_node(state: AgentState) -> AgentState:
    """Single LLM call with retrieved context. Sets: draft_answer."""
    llm = _get_llm()
    query = state["query"]
    context_str = state.get("context_str", "")
    history = state.get("conversation_history", [])

    # Build the user message with context injected
    user_content = (
        f"RETRIEVED POLICY CONTEXT:\n{context_str}\n\n"
        f"RESIDENT QUESTION:\n{query}"
        if context_str
        else query
    )

    messages = list(history) + [{"role": "user", "content": user_content}]

    draft = await llm.chat(
        messages=messages,
        system=_SYSTEM_PROMPT,
        max_tokens=1024,
        stream=False,
    )

    log.info(
        "graph.fast_rag",
        correlation_id=state.get("correlation_id"),
        answer_len=len(draft),
    )

    return {**state, "draft_answer": draft}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3b: agent_node
# ─────────────────────────────────────────────────────────────────────────────


async def agent_node(state: AgentState) -> AgentState:
    """
    Tool-using LLM loop (up to _MAX_TOOL_ROUNDS rounds).
    Handles both Anthropic (tools format) and Azure OpenAI (function calling).
    Sets: draft_answer, tool_results.
    """
    llm = _get_llm()
    cfg = get_settings()
    query = state["query"]
    context_str = state.get("context_str", "")
    history = list(state.get("conversation_history", []))
    tool_results: list[dict] = []

    user_content = (
        f"RETRIEVED POLICY CONTEXT:\n{context_str}\n\n"
        f"RESIDENT QUESTION:\n{query}"
        if context_str
        else query
    )
    messages = history + [{"role": "user", "content": user_content}]

    provider = cfg.llm_provider
    draft = ""

    # ── Anthropic tool loop ───────────────────────────────────────────────────
    if provider == "anthropic":
        import anthropic as anthropic_sdk

        client = llm._anthropic  # type: ignore[attr-defined]
        for round_idx in range(_MAX_TOOL_ROUNDS):
            response = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                tools=_ANTHROPIC_TOOLS,  # type: ignore[arg-type]
                messages=messages,
            )

            # Collect any text from this response turn
            text_blocks = [
                b.text for b in response.content if b.type == "text"
            ]
            if text_blocks:
                draft = "\n".join(text_blocks)

            # Check stop reason
            if response.stop_reason != "tool_use":
                # No more tool calls — we have a final answer
                break

            # Process tool_use blocks
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                break

            # Append assistant message with full content
            messages.append(
                {"role": "assistant", "content": response.content}
            )

            # Execute each tool and build tool_result content
            tool_result_content: list[dict] = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block.name
                tool_input = tool_block.input or {}

                if tool_name in _TOOL_REGISTRY:
                    input_model_cls, tool_fn = _TOOL_REGISTRY[tool_name]
                    try:
                        parsed_input = input_model_cls(**tool_input)
                        result_obj = await tool_fn(parsed_input)
                        result_dict = result_obj.model_dump()
                        result_str = json.dumps(result_dict)
                        tool_results.append(
                            {
                                "tool": tool_name,
                                "input": tool_input,
                                "output": result_dict,
                            }
                        )
                        log.info(
                            "graph.agent.tool_call",
                            correlation_id=state.get("correlation_id"),
                            tool=tool_name,
                            round=round_idx,
                        )
                    except Exception as exc:
                        result_str = json.dumps({"error": str(exc)})
                        log.error(
                            "graph.agent.tool_error",
                            correlation_id=state.get("correlation_id"),
                            tool=tool_name,
                            error=str(exc),
                        )
                else:
                    result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})

                tool_result_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": result_str,
                    }
                )

            # Append tool results as a user message
            messages.append({"role": "user", "content": tool_result_content})

        # If no text was collected (only tool calls), run one more pass for final answer
        if not draft:
            final_response = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                messages=messages,
            )
            draft = next(
                (b.text for b in final_response.content if b.type == "text"),
                "I was unable to generate a response.",
            )

    # ── Azure OpenAI tool loop ────────────────────────────────────────────────
    elif provider == "azure_openai":
        import openai as openai_sdk

        azure_client = llm._azure  # type: ignore[attr-defined]
        deployment = cfg.azure_openai_chat_deployment

        # Convert messages to Azure format (with system prepended)
        az_messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        az_messages.extend(messages)

        for round_idx in range(_MAX_TOOL_ROUNDS):
            response = await azure_client.chat.completions.create(
                model=deployment,
                messages=az_messages,  # type: ignore[arg-type]
                tools=_AZURE_FUNCTIONS,  # type: ignore[arg-type]
                tool_choice="auto",
                max_tokens=2048,
            )

            choice = response.choices[0]
            msg = choice.message

            # Collect text content
            if msg.content:
                draft = msg.content

            # Check if there are tool calls
            if not msg.tool_calls or choice.finish_reason == "stop":
                break

            # Append assistant message
            az_messages.append(msg.model_dump(exclude_none=True))

            # Execute each tool call
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    tool_input = {}

                if tool_name in _TOOL_REGISTRY:
                    input_model_cls, tool_fn = _TOOL_REGISTRY[tool_name]
                    try:
                        parsed_input = input_model_cls(**tool_input)
                        result_obj = await tool_fn(parsed_input)
                        result_dict = result_obj.model_dump()
                        result_str = json.dumps(result_dict)
                        tool_results.append(
                            {
                                "tool": tool_name,
                                "input": tool_input,
                                "output": result_dict,
                            }
                        )
                        log.info(
                            "graph.agent.tool_call",
                            correlation_id=state.get("correlation_id"),
                            tool=tool_name,
                            round=round_idx,
                        )
                    except Exception as exc:
                        result_str = json.dumps({"error": str(exc)})
                        log.error(
                            "graph.agent.tool_error",
                            correlation_id=state.get("correlation_id"),
                            tool=tool_name,
                            error=str(exc),
                        )
                else:
                    result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})

                az_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    }
                )

        if not draft:
            # One final non-tool call to get the summary answer
            final_resp = await azure_client.chat.completions.create(
                model=deployment,
                messages=az_messages,  # type: ignore[arg-type]
                max_tokens=2048,
            )
            draft = final_resp.choices[0].message.content or "I was unable to generate a response."

    log.info(
        "graph.agent",
        correlation_id=state.get("correlation_id"),
        n_tool_calls=len(tool_results),
        answer_len=len(draft),
    )

    return {**state, "draft_answer": draft, "tool_results": tool_results}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: groundedness_gate_node
# ─────────────────────────────────────────────────────────────────────────────


async def groundedness_gate_node(state: AgentState) -> AgentState:
    """
    Run the groundedness judge.
    Appends a disclaimer if not grounded; sets is_grounded and final_answer.
    """
    judge = _get_judge()
    draft = state.get("draft_answer", "")
    chunks = state.get("retrieved_chunks", [])
    query = state["query"]

    contexts = [c.get("content", "") for c in chunks if c.get("content")]

    verdict = await judge.check(answer=draft, contexts=contexts, query=query)
    is_grounded = verdict["is_grounded"]

    final_answer = draft if is_grounded else draft + _DISCLAIMER

    log.info(
        "graph.groundedness",
        correlation_id=state.get("correlation_id"),
        is_grounded=is_grounded,
        confidence=verdict.get("confidence"),
        reasoning=verdict.get("reasoning", "")[:120],
    )

    return {**state, "is_grounded": is_grounded, "final_answer": final_answer}


# ─────────────────────────────────────────────────────────────────────────────
# Node 5: refusal_node
# ─────────────────────────────────────────────────────────────────────────────


async def refusal_node(state: AgentState) -> AgentState:
    """Return a polite refusal when retrieval yields no useful results."""
    final_answer = (
        "I don't have enough policy information to answer that reliably. "
        "Please contact the JCRC directly at jcrc@sheares.nus.edu.sg."
    )
    log.info(
        "graph.refusal",
        correlation_id=state.get("correlation_id"),
        n_chunks=len(state.get("retrieved_chunks", [])),
    )
    return {
        **state,
        "draft_answer": final_answer,
        "final_answer": final_answer,
        "is_grounded": True,  # refusals are always "safe" — no ungrounded claims
    }


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge: retrieve → fast_rag | agent_llm | refusal
# ─────────────────────────────────────────────────────────────────────────────


def _route_after_retrieve(state: AgentState) -> str:
    """
    Decide which node follows retrieval:
      - "refusal" if no useful chunks found
      - state["route"] ("fast" or "agent") otherwise
    """
    if _should_refuse(state):
        return "refusal"
    return state["route"]


# ─────────────────────────────────────────────────────────────────────────────
# Build and compile the graph
# ─────────────────────────────────────────────────────────────────────────────

_graph_builder = StateGraph(AgentState)

_graph_builder.add_node("route", route_node)
_graph_builder.add_node("retrieve", retrieve_node)
_graph_builder.add_node("fast_rag", fast_rag_node)
_graph_builder.add_node("agent_llm", agent_node)
_graph_builder.add_node("groundedness_gate", groundedness_gate_node)
_graph_builder.add_node("refusal", refusal_node)

_graph_builder.set_entry_point("route")
_graph_builder.add_edge("route", "retrieve")

_graph_builder.add_conditional_edges(
    "retrieve",
    _route_after_retrieve,
    {
        "fast": "fast_rag",
        "agent": "agent_llm",
        "refusal": "refusal",
    },
)

_graph_builder.add_edge("fast_rag", "groundedness_gate")
_graph_builder.add_edge("agent_llm", "groundedness_gate")
_graph_builder.add_edge("groundedness_gate", END)
_graph_builder.add_edge("refusal", END)

compiled_graph = _graph_builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


async def run_query(
    query: str,
    history: list[dict] | None = None,
    correlation_id: str | None = None,
) -> dict:
    """
    Execute the full RAG/agent pipeline for a single user query.

    Args:
        query:          The resident's question.
        history:        Prior conversation turns in OpenAI message format.
        correlation_id: Optional trace ID; a new UUID is generated if omitted.

    Returns:
        {
            "answer":       str   — final answer shown to the user,
            "citations":    list  — [{number, source_doc, section, score, ...}],
            "route":        str   — "fast" | "agent",
            "is_grounded":  bool  — verdict from the groundedness judge,
            "tool_results": list  — [{tool, input, output}] for agent calls,
        }
    """
    cid = bind_correlation_id(correlation_id)
    log.info("graph.run_query.start", correlation_id=cid, query=query[:120])

    initial_state: AgentState = {
        "query": query,
        "conversation_history": history or [],
        "route": "fast",
        "retrieved_chunks": [],
        "context_str": "",
        "tool_results": [],
        "draft_answer": "",
        "is_grounded": False,
        "final_answer": "",
        "citations": [],
        "correlation_id": cid,
    }

    final_state: AgentState = await compiled_graph.ainvoke(initial_state)

    result = {
        "answer": final_state.get("final_answer", ""),
        "citations": final_state.get("citations", []),
        "route": final_state.get("route", "fast"),
        "is_grounded": final_state.get("is_grounded", False),
        "tool_results": final_state.get("tool_results", []),
    }

    log.info(
        "graph.run_query.done",
        correlation_id=cid,
        route=result["route"],
        is_grounded=result["is_grounded"],
        n_citations=len(result["citations"]),
        n_tool_results=len(result["tool_results"]),
        answer_len=len(result["answer"]),
    )

    return result
