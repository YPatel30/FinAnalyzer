"""`uv run ingest TICKER [TICKER ...]` — the Phase 1 entry point."""

import argparse
import sys

from pymongo.database import Database

from fin_analyzer.config import get_settings
from fin_analyzer.db import ensure_indexes, get_db
from fin_analyzer.edgar_client import fetch_ticker_map
from fin_analyzer.ingest import ingest_ticker


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest SEC 10-K filings into MongoDB.")
    parser.add_argument("tickers", nargs="+", help="Stock tickers, e.g. AAPL MSFT GOOGL")
    args = parser.parse_args()

    settings = get_settings()
    db = get_db(settings)
    ensure_indexes(db)

    # Fetched once per run and reused for every ticker in the batch, rather
    # than once per ticker — it's the same ~10MB file regardless of which
    # tickers we're looking up.
    ticker_map = fetch_ticker_map(settings)

    succeeded = []
    failed = []
    for ticker in args.tickers:
        try:
            succeeded.append(ingest_ticker(ticker, ticker_map, settings, db))
        except Exception as exc:
            print(f"[{ticker}] FAILED: {exc}")
            failed.append(ticker)

    print(f"\nDone. Ingested {len(succeeded)}/{len(args.tickers)} tickers.")
    filing_ids = [result["filing_id"] for result in succeeded]
    _print_sample_chunks(db, filing_ids)

    if failed:
        sys.exit(1)


def _print_sample_chunks(db: Database, filing_ids: list, count: int = 3) -> None:
    """Sample only from filings ingested *this run* — sampling the whole
    `chunks` collection would mostly show older tickers from past runs
    once a few have accumulated, defeating the point of eyeballing what
    was just ingested."""
    if not filing_ids:
        print("\nNo chunks to sample.")
        return

    pipeline = [
        {"$match": {"filing_id": {"$in": filing_ids}}},
        {"$sample": {"size": count}},
    ]
    sample = list(db.chunks.aggregate(pipeline))
    if not sample:
        print("\nNo chunks in the database to sample.")
        return

    print(f"\n--- {len(sample)} random chunk(s) ---")
    for doc in sample:
        print(f"\n[{doc['ticker']} chunk #{doc['chunk_index']}, {doc['char_count']} chars]")
        print(doc["text"][:800])


if __name__ == "__main__":
    main()
