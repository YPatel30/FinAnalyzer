"""Idempotency of the Mongo write path — no SEC network calls.

Calls upsert_filing()/write_chunks() directly (the functions ingest_ticker()
itself uses) rather than running the full CLI, so this doesn't depend on
network access or real filing data.
"""

from fin_analyzer.ingest import upsert_filing, write_chunks


def test_upsert_filing_does_not_duplicate(test_db):
    kwargs = dict(
        ticker="TEST",
        form_type="10-K",
        filing_date="2024-01-01",
        accession_no="0000000000-24-000001",
        source_url="https://example.com/filing.html",
    )

    first_id = upsert_filing(test_db, **kwargs)
    second_id = upsert_filing(test_db, **kwargs)

    assert first_id == second_id
    assert test_db.filings.count_documents({"accession_no": kwargs["accession_no"]}) == 1


def test_write_chunks_does_not_duplicate_on_rerun(test_db):
    filing_id = upsert_filing(
        test_db,
        ticker="TEST",
        form_type="10-K",
        filing_date="2024-01-01",
        accession_no="0000000000-24-000002",
        source_url="https://example.com/filing2.html",
    )
    chunks = [f"chunk number {i} of a fake filing" for i in range(5)]

    first_count = write_chunks(test_db, "TEST", filing_id, chunks)
    second_count = write_chunks(test_db, "TEST", filing_id, chunks)

    assert first_count == 5
    assert second_count == 5
    stored = test_db.chunks.count_documents({"ticker": "TEST", "filing_id": filing_id})
    assert stored == 5  # rerunning must replace, not append
