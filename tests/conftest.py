"""Shared pytest fixtures.

Tests that need Mongo talk to the real Atlas cluster from .env, but always
against the `finanalyzer_test` database — never `finanalyzer` — regardless of
what MONGODB_DB is set to. No SEC network calls happen in tests.
"""

import pytest

from fin_analyzer.config import Settings
from fin_analyzer.db import ensure_indexes, get_db


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    return Settings(mongodb_db="finanalyzer_test")


@pytest.fixture
def test_db(test_settings: Settings):
    db = get_db(test_settings)
    ensure_indexes(db)
    yield db
    # Leave the test database clean for the next test / run.
    db.companies.delete_many({})
    db.filings.delete_many({})
    db.chunks.delete_many({})
