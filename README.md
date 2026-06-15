# Sheares Hall Policy Copilot

> **v1 → v2:** A ground-up rebuild of the Sheares Hall JCRC chatbot — from a single-file TF-IDF keyword bot to a production agentic RAG platform with hybrid retrieval, multi-step tool use, structured outputs, a CI-gated eval harness, and full observability.

---

## What it does

A policy assistant for Sheares Hall residents and JCRC members that:
- Answers hall policy questions with **grounded, cited answers** (constitution, finance procedures, SOPs, social media policy, award criteria, IHG rules)
- Runs structured hall workflows — **points eligibility checks**, **subsidy calculations**, **claim drafting**, **human escalation**
- Routes queries adaptively: **simple factual → fast vector-RAG path**, **complex/action → agentic LangGraph path**
- Refuses to answer out-of-corpus questions rather than hallucinate
- Streams answers in a chat UI with citations, and exposes a REST API for integration

---

## Architecture

```
                 ┌─────────────── Streamlit chat UI ────────────────┐
                 │   streaming answers · citations · 👍/👎 feedback   │
                 └───────────────────────┬──────────────────────────┘
                                         │ HTTP/SSE
                 ┌───────────────────────▼──────────────────────────┐
                 │  FastAPI  /chat (stream) · /feedback · /health    │
                 └───────────────────────┬──────────────────────────┘
                                         │
                 ┌───────────────────────▼──────────────────────────┐
                 │  Adaptive Router (query classifier)               │
                 │  simple → FAST PATH │ complex/action → AGENT      │
                 └──────┬─────────────────────────┬─────────────────┘
                        │ Fast path               │ Agent path
        ┌───────────────▼──────────┐  ┌───────────▼────────────────────────┐
        │ Vector RAG               │  │ LangGraph agent                     │
        │ retrieve → rerank        │  │ plan → tools → synthesize           │
        │ → grounded, cited answer │  │ + conversation + profile memory     │
        └───────────────┬──────────┘  └──┬──────────┬──────────┬───────────┘
                        │               │          │          │
                        │    points_eligibility  subsidy  escalate_to_human
                        │    (structured calc)   _calc    (Telegram/email)
                        │              (both paths share retrieval + LLM)
   ┌────────────────────▼──────────────▼───────────┐  provider-agnostic LLM
   │  Qdrant vector DB · BM25 hybrid · bge-reranker │  (Azure OpenAI ↔ Anthropic)
   └───────────────────────────────────────────────┘  Langfuse traces · RAGAS eval
```

---

## Milestones

| # | Title | Status |
|---|-------|--------|
| M0 | Foundations: repo, logging, PII pipeline, ingestion | 🟡 In progress |
| M1 | Vector RAG fast path (beats v1 TF-IDF) | ⬜ Pending |
| M2 | Adaptive router + LangGraph agentic path | ⬜ Pending |
| M3 | Eval harness: RAGAS + golden set + CI gate | ⬜ Pending |
| M4 | FastAPI + Streamlit UI + Langfuse observability | ⬜ Pending |
| M5 | Docker Compose + Azure Container Apps deploy | ⬜ Pending |

---

## Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Agent orchestration | LangGraph | Explicit multi-step graph + memory; LangChain ecosystem (named in JD) |
| RAG framework | LlamaIndex | Document loaders + retrievers (named in JD) |
| Vector DB | Qdrant | Dockerable, metadata filtering, hybrid BM25+dense |
| Reranking | bge-reranker (cross-encoder) | Local, fast, strong precision |
| LLM | Azure OpenAI (gpt-4o) / Anthropic (Claude) | Provider-agnostic; swap via `.env` |
| Eval | RAGAS + LLM-as-judge | Faithfulness, context precision, answer relevancy, MRR |
| Backend | FastAPI + SSE | Streaming, typed endpoints |
| Frontend | Streamlit | Fast credible chat UI |
| Observability | Langfuse (self-hosted) + structlog | Trace per turn, correlated with JSON structured logs |
| Deploy | Docker Compose → Azure Container Apps | Fully containerized |

---

## Data governance

The real Sheares Hall corpus contains resident PII (namelists, contracts, scholarship applications, SEP endorsements). This project:
1. **Allowlists only prose policy documents** (constitution, finance procedures, SOPs, award proposals, sponsorship guidelines) — personal records are excluded at ingestion
2. **Runs a PII redaction pass** (Presidio + spaCy NER + regex) before any text reaches the vector index; a CI test asserts zero names/emails/matric IDs in indexed text
3. **Uses synthetic residents/rooms/rate-tables** for structured-tool demos — no real individual data
4. **Commits only the synthetic + redacted sample corpus** — the real corpus mount is `.gitignore`d

---

## Quick start (local dev)

```bash
# 1. Clone and install
git clone git@github.com:isaacchin12/sheares-policy-copilot.git
cd sheares-policy-copilot
cp .env.example .env     # fill in your keys
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Build the index (over the synthetic/redacted sample corpus)
python ingestion/build_index.py

# 3. Start services
docker compose up -d qdrant langfuse

# 4. Run the API
uvicorn api.main:app --reload

# 5. Open the chat UI
streamlit run ui/app.py
```

---

## Evaluation

```bash
# Run the full eval suite (also runs in CI on every PR)
python eval/run_ragas.py

# Outputs: faithfulness, answer_relevancy, context_precision, context_recall, MRR, p50/p95 latency
```

---

## v1 → v2

| | v1 (sheares_chatbot) | v2 (this project) |
|---|---|---|
| Retrieval | TF-IDF keyword search | Hybrid BM25 + dense + cross-encoder rerank |
| Knowledge base | 9 hand-curated docs | Redacted/synthetic policy corpus (100+ docs) |
| Architecture | Single script | Modular pipeline (ingestion / retrieval / agent / api / ui) |
| LLM | Claude (single provider) | Provider-agnostic (Azure OpenAI ↔ Anthropic) |
| Workflows | Q&A only | Agentic tools (points calc, subsidy calc, claim drafting, human escalation) |
| Routing | Always LLM | Adaptive: fast vector-RAG path vs agentic path |
| Eval | None | RAGAS + golden set + LLM-judge + CI gate |
| Observability | Log file | Langfuse traces + structlog (correlated by trace ID) |
| Deploy | Flask + Heroku | FastAPI + Docker Compose + Azure Container Apps |

---

*Built by Isaac Chin (45th JCRC President, Sheares Hall, NUS) · 2026*
