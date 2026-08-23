"""Orchestrates ingestion of one ticker: CIK -> filing -> text -> chunks -> Mongo.

The Mongo write functions (upsert_company, upsert_filing, write_chunks) are
kept separate from ingest_ticker() so each one is a small, obvious unit —
and so test_ingest_idempotent.py can call write_chunks() directly without
needing a real SEC HTTP round trip.
"""

from pymongo import ReturnDocument
from pymongo.database import Database

from fin_analyzer.chunk import chunk_text
from fin_analyzer.cik_lookup import resolve_cik
from fin_analyzer.config import Settings
from fin_analyzer.edgar_client import build_filing_url, fetch_filing_html, fetch_submissions
from fin_analyzer.extract import extract_text


class NoTenKFoundError(Exception):
    pass


def find_latest_10k(submissions: dict) -> dict:
    """Pick the most recent 10-K out of a submissions.json payload.

    submissions["filings"]["recent"] is a set of parallel arrays (one entry
    per filing, newest first) rather than a list of per-filing dicts — that's
    SEC's format, not a choice made here. Only the last ~1000 filings are in
    "recent"; anything older lives in a separate paginated "files" list that
    Phase 1 doesn't fetch, since every real company's *most recent* 10-K is
    always within that window.
    """
    recent = submissions["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return {
                "form_type": form,
                "filing_date": recent["filingDate"][i],
                "accession_no": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
            }
    raise NoTenKFoundError("No 10-K found in this company's recent filings")


def upsert_company(db: Database, ticker: str, cik: str, name: str, sector: str | None) -> None:
    db.companies.update_one(
        {"ticker": ticker},
        {"$set": {"cik": cik, "name": name, "sector": sector}},
        upsert=True,
    )


def upsert_filing(
    db: Database, ticker: str, form_type: str, filing_date: str, accession_no: str, source_url: str
):
    """Upsert keyed on accession_no (SEC's permanent ID for one filing).
    find_one_and_update with return_document=AFTER gets us the filing's _id
    in the same round trip, whether this run just created it or it already
    existed from a previous run — that's what write_chunks() needs to key on.
    """
    doc = db.filings.find_one_and_update(
        {"accession_no": accession_no},
        {
            "$set": {
                "ticker": ticker,
                "form_type": form_type,
                "filing_date": filing_date,
                "accession_no": accession_no,
                "source_url": source_url,
            }
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return doc["_id"]


def write_chunks(db: Database, ticker: str, filing_id, chunk_texts: list[str]) -> int:
    """Replace all chunks for this (ticker, filing_id) with `chunk_texts`.

    Delete-then-insert, rather than diffing against what's already there, is
    what makes reruns idempotent: run this twice with the same input and the
    collection ends up with exactly one copy of each chunk, not two.
    """
    db.chunks.delete_many({"ticker": ticker, "filing_id": filing_id})
    if not chunk_texts:
        return 0
    docs = [
        {
            "ticker": ticker,
            "filing_id": filing_id,
            "chunk_index": i,
            "text": text,
            "char_count": len(text),
        }
        for i, text in enumerate(chunk_texts)
    ]
    db.chunks.insert_many(docs)
    return len(docs)


def ingest_ticker(ticker: str, ticker_map: dict, settings: Settings, db: Database) -> dict:
    resolved = resolve_cik(ticker, ticker_map)
    cik = resolved["cik"]
    print(f"[{resolved['ticker']}] CIK {cik}")

    submissions = fetch_submissions(cik, settings)
    # submissions.json's own "name"/"sicDescription" are EDGAR's official
    # values, more authoritative than the ticker map's "title".
    company_name = submissions.get("name", resolved["name"])
    sector = submissions.get("sicDescription")
    upsert_company(db, ticker=resolved["ticker"], cik=cik, name=company_name, sector=sector)

    filing = find_latest_10k(submissions)
    print(f"[{resolved['ticker']}] most recent 10-K: {filing['filing_date']} (accession {filing['accession_no']})")

    html = fetch_filing_html(
        cik=cik,
        accession_no=filing["accession_no"],
        primary_document=filing["primary_document"],
        ticker=resolved["ticker"],
        settings=settings,
    )
    source_url = build_filing_url(cik, filing["accession_no"], filing["primary_document"])
    filing_id = upsert_filing(
        db,
        ticker=resolved["ticker"],
        form_type=filing["form_type"],
        filing_date=filing["filing_date"],
        accession_no=filing["accession_no"],
        source_url=source_url,
    )

    text = extract_text(html)
    print(f"[{resolved['ticker']}] extracted {len(text):,} characters of prose")

    chunks = chunk_text(text, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    chunk_count = write_chunks(db, resolved["ticker"], filing_id, chunks)
    print(f"[{resolved['ticker']}] wrote {chunk_count} chunks")

    return {"ticker": resolved["ticker"], "cik": cik, "filing_id": filing_id, "chunk_count": chunk_count}
