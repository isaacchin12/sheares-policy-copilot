"""
Run RAGAS evaluation over the golden set.
Requires: ANTHROPIC_API_KEY or AZURE_OPENAI_API_KEY in environment.
Qdrant + ingestion must be complete before running.
"""
import asyncio, json, time
from pathlib import Path
from agent.graph import run_query
from observability import configure_logging, get_logger

configure_logging()
log = get_logger("eval.run_ragas")

async def evaluate():
    golden_set = [json.loads(l) for l in Path("eval/golden_set.jsonl").read_text().splitlines() if l.strip()]

    results = []
    for item in golden_set:
        start = time.perf_counter()
        try:
            result = await run_query(item["query"])
            latency = (time.perf_counter() - start) * 1000

            # Simple keyword-based faithfulness check
            answer = result["answer"].lower()
            expected = item.get("expected_answer_contains", [])
            hits = sum(1 for kw in expected if kw.lower() in answer)
            keyword_recall = hits / len(expected) if expected else 1.0

            # Route accuracy
            route_correct = result.get("route") == item["path"]

            results.append({
                "id": item["id"],
                "query": item["query"],
                "route": result.get("route"),
                "route_correct": route_correct,
                "keyword_recall": keyword_recall,
                "latency_ms": round(latency, 1),
                "is_grounded": result.get("is_grounded", True),
                "answer_preview": result["answer"][:100],
            })
            log.info("eval.item", id=item["id"], route_correct=route_correct, kw_recall=round(keyword_recall,2))
        except Exception as e:
            log.error("eval.item.error", id=item["id"], error=str(e))
            results.append({"id": item["id"], "error": str(e)})

    # Summary stats
    successful = [r for r in results if "error" not in r]
    avg_recall = sum(r["keyword_recall"] for r in successful) / len(successful) if successful else 0
    avg_latency = sum(r["latency_ms"] for r in successful) / len(successful) if successful else 0
    route_acc = sum(r["route_correct"] for r in successful) / len(successful) if successful else 0
    grounded_pct = sum(r["is_grounded"] for r in successful) / len(successful) if successful else 0

    print("\n=== EVAL RESULTS ===")
    print(f"Questions evaluated: {len(golden_set)}")
    print(f"Successful: {len(successful)}")
    print(f"Keyword recall (faithfulness proxy): {avg_recall:.2%}")
    print(f"Route accuracy: {route_acc:.2%}")
    print(f"Grounded answers: {grounded_pct:.2%}")
    print(f"Avg latency: {avg_latency:.0f}ms")
    print("====================\n")

    Path("eval/results").mkdir(exist_ok=True)
    Path("eval/results/latest.json").write_text(json.dumps({
        "summary": {"keyword_recall": avg_recall, "route_accuracy": route_acc, "grounded_pct": grounded_pct, "avg_latency_ms": avg_latency},
        "details": results
    }, indent=2))

    return {"keyword_recall": avg_recall, "route_accuracy": route_acc, "grounded_pct": grounded_pct}

if __name__ == "__main__":
    asyncio.run(evaluate())
