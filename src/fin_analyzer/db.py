"""MongoDB connection and index setup.

Plain sync pymongo — one client per process is enough for a CLI tool like
this, so there's no connection-pooling logic to think about here.
"""

from pymongo import ASCENDING, MongoClient
from pymongo.database import Database

from fin_analyzer.config import Settings


def get_client(settings: Settings) -> MongoClient:
    return MongoClient(settings.mongodb_uri)


def get_db(settings: Settings) -> Database:
    client = get_client(settings)
    return client[settings.mongodb_db]


def ensure_indexes(db: Database) -> None:
    """Create indexes if they don't already exist. Safe to call every run.

    - companies.ticker: unique, so upsert-by-ticker is the natural idempotency key.
    - filings.accession_no: unique, so upsert-by-accession is the idempotency key
      for filings (an accession number is SEC's permanent ID for one filing).
    - chunks {ticker, filing_id}: compound, not unique — dedup for chunks is
      handled at write time (delete-then-insert in ingest.py), this index just
      makes "give me all chunks for this filing" fast.
    - jobs.job_id: unique — every GET /ingest/{job_id} poll looks up by this
      field (Phase 4); without an index that's a full collection scan on
      every poll, which only gets worse as more ingests accumulate.
    """
    db.companies.create_index([("ticker", ASCENDING)], unique=True)
    db.filings.create_index([("accession_no", ASCENDING)], unique=True)
    db.chunks.create_index([("ticker", ASCENDING), ("filing_id", ASCENDING)])
    db.jobs.create_index([("job_id", ASCENDING)], unique=True)
