"""Pure logic for resolving a ticker to a CIK.

Kept separate from edgar_client.py (which does the actual HTTP fetch) so this
can be unit-tested against a small in-memory dict, with no network involved.
"""


class TickerNotFoundError(Exception):
    pass


def resolve_cik(ticker: str, ticker_map: dict) -> dict:
    """Look up `ticker` in the SEC ticker map.

    `ticker_map` is the parsed company_tickers.json: a dict of
    {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    (the outer keys are just row numbers as strings, not meaningful).

    Returns {"cik": "0000320193", "ticker": "AAPL", "name": "Apple Inc."}
    with the CIK zero-padded to 10 digits, since that's the form every SEC
    URL we need expects.
    """
    ticker_upper = ticker.upper()
    for row in ticker_map.values():
        if row["ticker"].upper() == ticker_upper:
            return {
                "cik": str(row["cik_str"]).zfill(10),
                "ticker": ticker_upper,
                "name": row["title"],
            }
    raise TickerNotFoundError(f"No CIK found for ticker {ticker!r}")
