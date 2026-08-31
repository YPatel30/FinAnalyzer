"""`uv run search "query" [--ticker AAPL] [--limit 5]`"""

import argparse

from fin_analyzer.search import search


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search over ingested 10-K chunks.")
    parser.add_argument("query", help="Natural-language question")
    parser.add_argument("--ticker", default=None, help="Restrict results to one company")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    results = search(args.query, ticker=args.ticker, limit=args.limit)
    if not results:
        print("No results.")
        return

    for chunk in results:
        print(f"\n[score {chunk.score:.4f}] {chunk.ticker} chunk #{chunk.chunk_index} (filed {chunk.filing_date})")
        print(chunk.text[:300])


if __name__ == "__main__":
    main()
