import asyncio, sys
from eval.run_ragas import evaluate

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-faithfulness", type=float, default=0.75)
    parser.add_argument("--min-recall-at-5", type=float, default=0.70)
    args = parser.parse_args()

    metrics = await evaluate()

    passed = True
    if metrics["keyword_recall"] < args.min_faithfulness:
        print(f"FAIL: keyword_recall {metrics['keyword_recall']:.2%} < threshold {args.min_faithfulness:.2%}")
        passed = False
    if metrics["route_accuracy"] < 0.80:
        print(f"FAIL: route_accuracy {metrics['route_accuracy']:.2%} < 80%")
        passed = False

    if passed:
        print("EVAL GATE: PASS")
        sys.exit(0)
    else:
        print("EVAL GATE: FAIL")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
