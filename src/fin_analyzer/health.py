"""Liveness + Mongo connectivity + vector index status — GET /health's
business logic. Reports degradation in its return value rather than
raising: a health check's whole job is to say what's wrong, not to fail
itself the moment something is.
"""

from dataclasses import dataclass

from pymongo.database import Database
from pymongo.errors import PyMongoError

from fin_analyzer.config import Settings
from fin_analyzer.vector_index import probe_once


@dataclass
class HealthStatus:
    mongo_connected: bool
    vector_index_status: str  # "ready" | "not_ready" | "unknown"

    @property
    def ok(self) -> bool:
        return self.mongo_connected and self.vector_index_status != "not_ready"


def check_health(db: Database, settings: Settings) -> HealthStatus:
    try:
        db.command("ping")
        mongo_connected = True
    except PyMongoError:
        return HealthStatus(mongo_connected=False, vector_index_status="unknown")

    hit = probe_once(db, settings)
    if hit is None:
        vector_index_status = "unknown"
    else:
        vector_index_status = "ready" if hit else "not_ready"

    return HealthStatus(mongo_connected=mongo_connected, vector_index_status=vector_index_status)
