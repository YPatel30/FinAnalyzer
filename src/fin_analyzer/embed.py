"""Finds chunks without an embedding, embeds them via Gemini, writes the
vectors back, and ensures the Atlas Vector Search index exists and is ready.
"""

import time

import tiktoken
from pymongo import UpdateOne
from pymongo.database import Database

from fin_analyzer.config import Settings
from fin_analyzer.embeddings import BATCH_SIZE, embed_texts
from fin_analyzer.vector_index import ensure_vector_index, wait_until_ready

RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"

# Deliberate pause between batches, on top of embeddings.py's reactive
# retry-on-429 — proactive pacing so a multi-batch run doesn't even approach
# the free tier's per-minute embedding limit in the first place, rather than
# relying entirely on backing off after tripping it (which is what happened
# the first time this ran for real).
BATCH_PAUSE_SECONDS = 15.0

# The free tier's embedding limit turned out to be *tokens*-per-minute, not
# requests-per-minute — discovered empirically: a 10-chunk call (~8K tokens)
# succeeded, a 50-chunk call (~45K tokens) immediately hit 429. Batches are
# grouped by this token budget (comfortably under the failure point, well
# above the success point) rather than by a fixed chunk count, since actual
# chunk size varies. Still capped at embeddings.BATCH_SIZE per call too,
# since that's a separate, hard per-call item limit.
MAX_TOKENS_PER_BATCH = 20_000

# gemini-embedding-001 caps input at 2048 tokens/text, and (per Google's own
# docs) newer embedding models tend to *truncate silently* rather than error
# on an over-limit input — meaning a bug here wouldn't announce itself, it'd
# just quietly embed a chopped-off chunk. So this is a hard pre-check, not a
# try/except: catch it before the API call, not after. 2000 (not 2048)
# leaves a small margin. Counted with the same cl100k_base tiktoken encoder
# chunk.py already targets ~1000 tokens with — an approximation of Gemini's
# actual tokenizer, but good enough for a safety trip-wire, and it costs no
# extra API calls to check.
MAX_SAFE_TOKENS = 2000
_ENCODER = tiktoken.get_encoding("cl100k_base")


class OversizedChunksError(Exception):
    pass


def embed_missing_chunks(
    db: Database,
    settings: Settings,
    embed_fn=embed_texts,
    ensure_index_fn=ensure_vector_index,
    wait_ready_fn=wait_until_ready,
) -> int:
    """Embed every chunk that doesn't have an `embedding` field yet, then
    make sure the vector index exists and is ready.

    Idempotent by construction: the query only selects chunks missing the
    field, so a chunk that was already embedded on a previous run is never
    re-sent to the API — that's what keeps reruns free. `embed_fn` is
    injectable so tests can supply a fake embedder and assert it's never
    called at all when nothing is pending. `ensure_index_fn`/`wait_ready_fn`
    are injectable too, so a test against finanalyzer_test can no-op them
    instead of creating a second real Atlas search index just for tests —
    M0 caps out at 3 total, see CLAUDE.md.

    Written to Mongo incrementally, one token-budgeted batch at a time (see
    MAX_TOKENS_PER_BATCH), rather than all at once at the end — so if a run
    stops partway (rate limit, Ctrl-C, laptop sleep), whatever was already
    embedded is already saved, and a rerun's `pending` query picks up
    exactly where it left off instead of redoing (or losing) that work.
    """
    pending = list(db.chunks.find({"embedding": {"$exists": False}}, {"text": 1, "ticker": 1, "chunk_index": 1}))
    token_counts = [len(_ENCODER.encode(doc["text"])) for doc in pending]

    oversized = [(doc, tc) for doc, tc in zip(pending, token_counts) if tc > MAX_SAFE_TOKENS]
    if oversized:
        _report_oversized(oversized)
        raise OversizedChunksError(
            f"{len(oversized)} chunk(s) exceed the ~{MAX_SAFE_TOKENS}-token safety threshold. "
            "That's a Phase 1 chunking issue to fix at the source (chunk.py / chunk_size_tokens), "
            "not something to paper over here — aborting before any embedding calls."
        )

    batches = _group_by_token_budget(pending, token_counts)

    total_written = 0
    total_skipped = 0
    for i, batch_docs in enumerate(batches):
        texts = [doc["text"] for doc in batch_docs]
        labels = [f"{doc['ticker']} chunk #{doc['chunk_index']}" for doc in batch_docs]
        vectors = embed_fn(texts, RETRIEVAL_DOCUMENT, settings, labels=labels)

        writes = []
        for doc, vector in zip(batch_docs, vectors):
            if vector is None:
                total_skipped += 1
                continue
            writes.append(UpdateOne({"_id": doc["_id"]}, {"$set": {"embedding": vector}}))

        if writes:
            db.chunks.bulk_write(writes)
            total_written += len(writes)
        print(f"  ...batch {i + 1}/{len(batches)}: {total_written}/{len(pending)} chunks embedded so far")

        if i < len(batches) - 1:
            time.sleep(BATCH_PAUSE_SECONDS)

    if total_skipped:
        print(f"Skipped {total_skipped} chunk(s) that failed to embed even individually — see log above.")

    # Sequenced deliberately after embedding, not before: the index
    # shouldn't be built against a field that's still being populated, and
    # wait_until_ready()'s readiness probe needs a real embedded chunk to
    # exist already. Runs every call (not just when something was embedded)
    # so a rerun still catches an index that failed to finish setting up.
    ensure_index_fn(db, settings)
    wait_ready_fn(db, settings)

    return total_written


def _group_by_token_budget(pending: list[dict], token_counts: list[int]) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_tokens = 0

    for doc, token_count in zip(pending, token_counts):
        over_budget = current_tokens + token_count > MAX_TOKENS_PER_BATCH
        over_item_limit = len(current_batch) >= BATCH_SIZE
        if current_batch and (over_budget or over_item_limit):
            batches.append(current_batch)
            current_batch, current_tokens = [], 0
        current_batch.append(doc)
        current_tokens += token_count

    if current_batch:
        batches.append(current_batch)
    return batches


def _report_oversized(oversized: list[tuple[dict, int]]) -> None:
    print(f"\n{len(oversized)} chunk(s) exceed the ~{MAX_SAFE_TOKENS}-token safety threshold:\n")
    for doc, token_count in oversized:
        print(f"  {doc['ticker']} chunk #{doc['chunk_index']} (_id={doc['_id']}): {token_count} tokens")
