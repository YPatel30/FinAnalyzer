"""`uv run ask "question" [--ticker AAPL] [--k 5] [--show-prompt]`"""

import argparse
import sys

from fin_analyzer.ask import ask, retrieve_and_build_prompt
from fin_analyzer.config import get_settings
from fin_analyzer.core.prompts import SYSTEM_INSTRUCTION
from fin_analyzer.providers.base import QuotaExhaustedError
from fin_analyzer.providers.validate import validate_configured_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a grounded, cited question over ingested 10-K filings.")
    parser.add_argument("question", help="Natural-language question")
    parser.add_argument("--ticker", default=None, help="Restrict retrieval to one company")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument(
        "--show-prompt", action="store_true", help="Print the exact prompt sent to the model before sending it"
    )
    args = parser.parse_args()

    settings = get_settings()

    try:
        validate_configured_models(settings)
    except RuntimeError as exc:
        print(f"Startup check failed: {exc}")
        sys.exit(1)

    if args.show_prompt:
        user_prompt, _ = retrieve_and_build_prompt(args.question, ticker=args.ticker, k=args.k, settings=settings)
        print("=== SYSTEM INSTRUCTION ===")
        print(SYSTEM_INSTRUCTION)
        print("=== USER PROMPT ===")
        if user_prompt is None:
            print("(nothing retrieved — ask() will refuse without calling the model at all)")
        else:
            print(user_prompt)
        print()

    try:
        result = ask(args.question, ticker=args.ticker, k=args.k, settings=settings)
    except QuotaExhaustedError as exc:
        print(f"\nGeneration quota exhausted: {exc}")
        sys.exit(1)

    print(result.answer)

    if result.sources:
        cited_count = sum(1 for s in result.sources if s.cited)
        print(f"\n--- {len(result.sources)} source(s) retrieved, {cited_count} cited ---")
        for i, source in enumerate(result.sources, start=1):
            marker = "CITED" if source.cited else "retrieved, not cited"
            print(
                f"\n[{i}] ({marker}) [score {source.score:.4f}] {source.ticker} "
                f"chunk #{source.chunk_index} (filed {source.filing_date})"
            )
            print(f"    {source.text[:200]}")


if __name__ == "__main__":
    main()
