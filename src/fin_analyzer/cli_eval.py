"""`uv run eval` — recall@5 over the 15 approved retrieval questions."""

from fin_analyzer.eval import run_eval


def main() -> None:
    results = run_eval()

    hits = sum(1 for r in results if r.hit)
    total = len(results)
    print(f"\nrecall@5: {hits}/{total} ({hits / total:.0%})")

    failures = [r for r in results if not r.hit]
    if not failures:
        print("\nAll questions passed.")
        return

    print(f"\n--- {len(failures)} failure(s) ---")
    for r in failures:
        q = r.question
        print(f"\n[{q['ticker']} / {q['section']}] {q['question']}")
        print(f"  expected phrase: {q['expected_phrase']!r}")
        print("  got instead:")
        for chunk in r.results:
            print(f"    [{chunk.score:.4f}] {chunk.ticker} chunk #{chunk.chunk_index}: {chunk.text[:150]}")


if __name__ == "__main__":
    main()
