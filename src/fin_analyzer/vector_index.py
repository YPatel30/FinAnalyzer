"""Create and confirm readiness of the Atlas Vector Search index on `chunks`.

Atlas restricts *driver-level* search-index management (create/list) to
M10+ clusters — M0 (free tier) rejects create_search_index()/
list_search_indexes() with an OperationFailure, even though $vectorSearch
*querying* works fine on M0 once an index exists. So ensure_vector_index()
tries the driver path first and, if that's not permitted here, prints the
index definition and manual Atlas UI steps instead.

Either way — driver-created or hand-created — readiness is confirmed the
same way: wait_until_ready() doesn't trust a keypress or a listed "status"
field (list_search_indexes() is off-limits on M0 too). It runs an actual
trivial $vectorSearch query in a retry loop and treats "at least one hit
came back" as the only proof that matters. A missing index raises
OperationFailure, a still-building index returns zero hits, a ready index
returns hits — all three are handled explicitly below.
"""

import time

from pymongo.database import Database
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from fin_analyzer.config import Settings

VECTOR_FIELD = "embedding"
# Fields search() may filter on — Atlas requires each declared in the index
# definition up front, it can't filter on arbitrary fields at query time.
FILTER_FIELDS = ["ticker"]


def index_definition(settings: Settings) -> dict:
    return {
        "fields": [
            {
                "type": "vector",
                "path": VECTOR_FIELD,
                "numDimensions": settings.embedding_dimensions,
                "similarity": "cosine",
            },
            *({"type": "filter", "path": field} for field in FILTER_FIELDS),
        ]
    }


def ensure_vector_index(db: Database, settings: Settings) -> None:
    """Create the index via the driver if this cluster tier allows it;
    otherwise print manual setup steps and pause for confirmation that
    they've been started. Either way, this function doesn't itself prove
    the index is ready — call wait_until_ready() next for that.
    """
    if _index_already_exists(db, settings):
        print(f"'{settings.vector_index_name}' already exists.")
        return

    model = SearchIndexModel(
        definition=index_definition(settings),
        name=settings.vector_index_name,
        type="vectorSearch",
    )
    try:
        db.chunks.create_search_index(model=model)
        print(f"Submitted '{settings.vector_index_name}' for creation via the driver.")
    except OperationFailure as exc:
        if _already_exists(exc):
            print(f"'{settings.vector_index_name}' already exists.")
            return
        print(_manual_setup_instructions(settings, exc))
        input("Press Enter once you've created it in the Atlas UI (this doesn't trust the keypress — readiness is verified programmatically next)... ")


def wait_until_ready(
    db: Database, settings: Settings, timeout_seconds: float = 120.0, initial_delay: float = 5.0
) -> None:
    """Poll with a real $vectorSearch probe query until it returns a hit.

    Requires at least one chunk to already have an embedding — that's the
    probe vector, taken straight from the database so this needs zero extra
    Gemini calls. (Sequencing note: embed.py always embeds every chunk
    *before* calling this, both because the index shouldn't be built against
    a field that's still being populated, and because this function needs a
    real vector to exist already.)
    """
    probe_vector = _get_probe_vector(db)
    deadline = time.monotonic() + timeout_seconds
    delay = initial_delay
    attempt = 0

    while True:
        attempt += 1
        if _probe_returns_a_hit(db, settings, probe_vector):
            print(f"'{settings.vector_index_name}' is READY (probe #{attempt} returned a hit).")
            return

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"'{settings.vector_index_name}' still isn't returning hits after "
                f"{timeout_seconds:.0f}s. Check Atlas UI -> Search for its build status/errors."
            )

        print(f"  ...not ready yet (probe #{attempt}), retrying in {delay:.0f}s")
        time.sleep(delay)
        delay = min(delay * 1.5, 20.0)


def probe_once(db: Database, settings: Settings) -> bool | None:
    """A single, non-retrying readiness check — unlike wait_until_ready(),
    this must return fast (Phase 4's GET /health calls it, and a health
    check shouldn't block a caller for up to 2 minutes). Returns None if
    there's no embedded chunk yet to probe with — that's "unknown", not
    "not ready": the index might be perfectly fine, there's just nothing to
    test it with yet.
    """
    try:
        probe_vector = _get_probe_vector(db)
    except RuntimeError:
        return None
    return _probe_returns_a_hit(db, settings, probe_vector)


def _probe_returns_a_hit(db: Database, settings: Settings, probe_vector: list[float]) -> bool:
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": settings.vector_index_name,
                    "path": VECTOR_FIELD,
                    "queryVector": probe_vector,
                    "numCandidates": 10,
                    "limit": 1,
                }
            },
            {"$project": {"_id": 1}},
        ]
        return len(list(db.chunks.aggregate(pipeline))) > 0
    except OperationFailure:
        # Index doesn't exist yet at all — same "not ready" bucket as a
        # still-building index returning zero hits; the caller just retries.
        return False


def _get_probe_vector(db: Database) -> list[float]:
    doc = db.chunks.find_one({"embedding": {"$exists": True}}, {"embedding": 1})
    if doc is None:
        raise RuntimeError("No embedded chunks found to use as a readiness probe — run embed_missing_chunks() first.")
    return doc["embedding"]


def _index_already_exists(db: Database, settings: Settings) -> bool:
    """Best-effort pre-check so a rerun's log doesn't claim to be "creating"
    an index that's already there. If listing isn't permitted on this
    cluster tier either, we just don't know yet — fall through to attempting
    creation, whose own OperationFailure handling (_already_exists) covers
    the same case.
    """
    try:
        return any(idx.get("name") == settings.vector_index_name for idx in db.chunks.list_search_indexes())
    except OperationFailure:
        return False


def _already_exists(exc: OperationFailure) -> bool:
    message = str(exc).lower()
    return "already exists" in message or "duplicate index" in message


def _manual_setup_instructions(settings: Settings, exc: Exception) -> str:
    import json

    definition_json = json.dumps(
        {"name": settings.vector_index_name, "type": "vectorSearch", "definition": index_definition(settings)},
        indent=2,
    )
    return f"""
Could not create the index via the driver ({exc}).
This is expected on Atlas M0/M2/M5 — search index *management* is
restricted to M10+ clusters; querying still works fine on M0 once the
index exists, you just have to create it by hand once.

Create it manually:
  1. Atlas UI -> your cluster -> Search tab -> Create Search Index
  2. Choose "Atlas Vector Search" -> JSON Editor
  3. Database: {settings.mongodb_db}   Collection: chunks
  4. Paste this definition:

{definition_json}

  5. Click Create.
"""
