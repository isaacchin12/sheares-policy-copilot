"""
Sheares Hall Policy Copilot — Demo Script

Tests 12 prompts across all capabilities:
- Fast vector RAG path (simple policy facts)
- Agent path (eligibility, calculations, escalation)
- Refusal (out-of-corpus questions)

Usage:
    python demo.py               # run all demos (requires API running)
    python demo.py --offline     # run directly against agent.graph (no API needed)
    python demo.py --prompt 3    # run only prompt #3
"""
import asyncio, argparse, json, time, sys

DEMO_PROMPTS = [
    # ── Fast path: simple facts ──────────────────────────────────────────
    {
        "id": 1,
        "label": "Points tiers",
        "path": "fast",
        "prompt": "What are the Shearite Points tiers for room retention priority?",
        "expect_keywords": ["Priority A", "Priority B", "Priority C", "80", "50", "30"],
    },
    {
        "id": 2,
        "label": "Finance claim deadline",
        "path": "fast",
        "prompt": "How long do I have to submit a finance claim after my event?",
        "expect_keywords": ["calendar days", "Claim Form"],
    },
    {
        "id": 3,
        "label": "Social media rules",
        "path": "fast",
        "prompt": "What are the social media rules for sponsorship logos?",
        "expect_keywords": ["24 hours", "Director of Communications", "official post"],
    },
    {
        "id": 4,
        "label": "Constitution quorum",
        "path": "fast",
        "prompt": "What is the quorum for a JCRC meeting?",
        "expect_keywords": ["2/3", "attendance"],
    },
    {
        "id": 5,
        "label": "Room swap window",
        "path": "fast",
        "prompt": "Can I swap rooms after moving in? How long do I have?",
        "expect_keywords": ["2 weeks", "swap"],
    },
    {
        "id": 6,
        "label": "IHG training attendance",
        "path": "fast",
        "prompt": "What is the minimum IHG training attendance to be eligible for a match?",
        "expect_keywords": ["60%"],
    },
    # ── Agent path: calculations + eligibility ───────────────────────────
    {
        "id": 7,
        "label": "Points eligibility check",
        "path": "agent",
        "prompt": "I have 45 points total — what tier am I in and how many more points do I need for the next tier?",
        "expect_keywords": ["Priority C", "5", "Priority B"],
    },
    {
        "id": 8,
        "label": "Subsidy calculation",
        "path": "agent",
        "prompt": "Calculate my holiday subsidy if I'm staying for 10 days. I have 35 points and applied before the deadline.",
        "expect_keywords": ["$7", "10", "$70"],
    },
    {
        "id": 9,
        "label": "Room retention eligibility",
        "path": "agent",
        "prompt": "I have 28 points. Am I eligible for room retention priority?",
        "expect_keywords": ["Not Eligible", "2 points", "30"],
    },
    {
        "id": 10,
        "label": "Supplementary points",
        "path": "agent",
        "prompt": "I have 72 base points plus 10 supplementary points. What tier am I in for room retention?",
        "expect_keywords": ["Priority A", "82"],
    },
    # ── Escalation ───────────────────────────────────────────────────────
    {
        "id": 11,
        "label": "Human escalation",
        "path": "agent",
        "prompt": "I think there's an error in my points calculation and I want to formally appeal. Can you help me escalate this to the JCRC?",
        "expect_keywords": ["SH-", "ticket", "working days"],
    },
    # ── Out-of-corpus refusal ────────────────────────────────────────────
    {
        "id": 12,
        "label": "Out-of-corpus refusal",
        "path": "fast",
        "prompt": "What is the NUS Office of Student Affairs policy on overnight guests in halls?",
        "expect_keywords": ["don't have", "contact", "JCRC"],
    },
]


async def run_offline(prompt_data: dict) -> dict:
    """Run directly against the agent graph (no API required)."""
    from observability import configure_logging, bind_correlation_id
    from agent.graph import run_query
    configure_logging()
    cid = bind_correlation_id()
    start = time.perf_counter()
    result = await run_query(prompt_data["prompt"], correlation_id=cid)
    latency = (time.perf_counter() - start) * 1000
    return {**result, "latency_ms": latency, "correlation_id": cid}


async def run_via_api(prompt_data: dict, api_url: str = "http://localhost:8000") -> dict:
    """Run via the FastAPI endpoint (non-streaming for simplicity in demo)."""
    import httpx
    async with httpx.AsyncClient(timeout=30) as client:
        start = time.perf_counter()
        resp = await client.post(f"{api_url}/chat", json={"message": prompt_data["prompt"], "stream": False})
        latency = (time.perf_counter() - start) * 1000
        result = resp.json()
        return {**result, "latency_ms": latency}


def evaluate_result(result: dict, prompt_data: dict) -> dict:
    answer = result.get("answer", "").lower()
    keywords = prompt_data.get("expect_keywords", [])
    hits = [kw for kw in keywords if kw.lower() in answer]
    return {
        "hit_rate": len(hits) / len(keywords) if keywords else 1.0,
        "hits": hits,
        "misses": [kw for kw in keywords if kw.lower() not in answer],
        "route_correct": result.get("route") == prompt_data.get("path"),
    }


async def main():
    parser = argparse.ArgumentParser(description="Sheares Policy Copilot demo")
    parser.add_argument("--offline", action="store_true", help="Skip API, run graph directly")
    parser.add_argument("--prompt", type=int, help="Run only this prompt number")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    prompts = [p for p in DEMO_PROMPTS if args.prompt is None or p["id"] == args.prompt]

    print("\n" + "="*70)
    print("  SHEARES HALL POLICY COPILOT — DEMO")
    print("  v1 keyword bot → v2 adaptive RAG + agentic AI")
    print("="*70)

    overall_scores = []

    for p in prompts:
        print(f"\n[{p['id']:02d}] {p['label'].upper()} ({p['path']} path)")
        print(f"  Q: {p['prompt']}")

        try:
            if args.offline:
                result = await run_offline(p)
            else:
                result = await run_via_api(p, args.api_url)

            answer = result.get("answer", "(no answer)")
            route = result.get("route", "?")
            latency = result.get("latency_ms", 0)
            citations = result.get("citations", [])
            grounded = result.get("is_grounded", True)

            eval_r = evaluate_result(result, p)
            overall_scores.append(eval_r["hit_rate"])

            print(f"  A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            print(f"  Route: {route} | Latency: {latency:.0f}ms | Grounded: {grounded}")
            if citations:
                print(f"  Citations: {[c.get('source_doc','?') for c in citations[:3]]}")

            status = "✅" if eval_r["hit_rate"] >= 0.6 else "⚠️ "
            print(f"  {status} Keyword check: {eval_r['hit_rate']:.0%} ({len(eval_r['hits'])}/{len(eval_r['hits'])+len(eval_r['misses'])} hits)")
            if eval_r["misses"]:
                print(f"     Missing: {eval_r['misses']}")
            if not eval_r["route_correct"]:
                print(f"     ⚠️  Expected route '{p['path']}', got '{result.get('route')}'")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            overall_scores.append(0.0)

    print("\n" + "="*70)
    avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    print(f"  OVERALL: {avg:.0%} keyword hit rate across {len(prompts)} prompts")
    print("="*70 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
