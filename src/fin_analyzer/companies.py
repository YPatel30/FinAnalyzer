"""Reading back what's been ingested. Nothing queries the `companies`
collection today — ingest.py only writes to it — so this is new, and used
by both `search.py` (ticker validation) and the Phase 4 API's /companies.
"""

from dataclasses import dataclass

from pymongo.database import Database


@dataclass
class Company:
    ticker: str
    name: str
    sector: str | None
    cik: str
    latest_filing_date: str | None


def list_companies(db: Database) -> list[Company]:
    """One row per ingested company, with its most recent filing date
    joined in — small enough (a handful of companies, not thousands) that a
    plain per-company lookup is simpler than an aggregation pipeline, and
    just as fast at this scale.
    """
    companies = []
    for doc in db.companies.find().sort("ticker", 1):
        latest_filing = db.filings.find_one({"ticker": doc["ticker"]}, sort=[("filing_date", -1)])
        companies.append(
            Company(
                ticker=doc["ticker"],
                name=doc["name"],
                sector=doc.get("sector"),
                cik=doc["cik"],
                latest_filing_date=latest_filing["filing_date"] if latest_filing else None,
            )
        )
    return companies


def ticker_exists(db: Database, ticker: str) -> bool:
    return db.companies.count_documents({"ticker": ticker.upper()}, limit=1) > 0
