"""`uv run eval` — recall@5, groundedness %, and refusal rate. See
eval.py's module docstring for what each of the three measures and why.

Validates configured generation/judge models are actually available to
this API key before spending anything — see providers/validate.py.
"""

import sys

from fin_analyzer.config import get_settings
from fin_analyzer.eval import run_eval, run_groundedness_eval, run_refusal_eval
from fin_analyzer.eval_checkpoint import CallCounter
from fin_analyzer.providers.validate import validate_configured_models


def main() -> None:
    settings = get_settings()

    try:
        validate_configured_models(settings)
    except RuntimeError as exc:
        print(f"Startup check failed: {exc}")
        sys.exit(1)

    # One counter shared across all three stages, so the total reflects the
    # whole run, not three counters each restarting at 0.
    counter = CallCounter()

    _print_recall(run_eval(settings=settings, counter=counter))
    _print_groundedness(run_groundedness_eval(settings=settings, counter=counter))
    _print_refusal(run_refusal_eval(settings=settings, counter=counter))

    print(f"\n{'=' * 70}\ntotal API calls this run: {counter.count}")


def _print_recall(results) -> None:
    hits = sum(1 for r in results if r.hit)
    total = len(results)
    print(f"recall@5: {hits}/{total} ({hits / total:.0%})")

    failures = [r for r in results if not r.hit]
    if failures:
        print(f"\n--- {len(failures)} recall failure(s) ---")
        for r in failures:
            q = r.question
            print(f"\n[{q['ticker']} / {q['section']}] {q['question']}")
            print(f"  expected phrase: {q['expected_phrase']!r}")
            print("  got instead:")
            for chunk in r.results:
                print(f"    [{chunk.score:.4f}] {chunk.ticker} chunk #{chunk.chunk_index}: {chunk.text[:150]}")


def _print_groundedness(results) -> None:
    print("\n" + "=" * 70)
    print("GROUNDEDNESS — a model judging a model's own output.")
    print("Useful signal, not proof.")

    judged = [r for r in results if r.verdict is not None]
    refused = [r for r in results if r.verdict is None]
    grounded = [r for r in judged if r.verdict.grounded]

    if judged:
        print(f"\ngrounded: {len(grounded)}/{len(judged)} ({len(grounded) / len(judged):.0%} of judged answers)")
    if refused:
        print(f"({len(refused)} question(s) refused instead of answered — excluded from the % above)")

    unsupported = [r for r in judged if not r.verdict.grounded]
    if unsupported:
        print(f"\n--- {len(unsupported)} answer(s) with unsupported claims ---")
        for r in unsupported:
            print(f"\n[{r.question['ticker']}] {r.question['question']}")
            print(f"  answer: {r.answer.answer[:200]}")
            print("  unsupported claims:")
            for claim in r.verdict.unsupported_claims:
                print(f"    - {claim}")


def _print_refusal(results) -> None:
    print("\n" + "=" * 70)
    print("REFUSAL SET — 5 questions the corpus provably cannot answer")

    refused = [r for r in results if r.refused]
    print(f"\nrefusal rate: {len(refused)}/{len(results)} ({len(refused) / len(results):.0%})")

    not_refused = [r for r in results if not r.refused]
    if not_refused:
        print(f"\n--- {len(not_refused)} question(s) that should have been refused but weren't ---")
        for r in not_refused:
            print(f"\n[{r.question['category']}] {r.question['question']}")
            print(f"  answered instead: {r.answer.answer[:300]}")


if __name__ == "__main__":
    main()
