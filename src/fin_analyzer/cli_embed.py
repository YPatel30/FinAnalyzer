"""`uv run embed` — embed any chunks missing vectors, then ensure the
Atlas Vector Search index exists and is ready."""

import sys

from google.genai import errors

from fin_analyzer.config import get_settings
from fin_analyzer.db import get_db
from fin_analyzer.embed import OversizedChunksError, embed_missing_chunks


def main() -> None:
    settings = get_settings()
    db = get_db(settings)

    try:
        count = embed_missing_chunks(db, settings)
    except OversizedChunksError as exc:
        print(f"\n{exc}")
        sys.exit(1)
    except errors.APIError as exc:
        if exc.code == 429:
            print(
                f"\nStill rate limited after retries ({exc}). Whatever embedded "
                "before this point is already saved — wait a bit and rerun "
                "`uv run embed`, it'll pick up exactly where this left off."
            )
        else:
            print(f"\nEmbedding failed: {exc}")
        sys.exit(1)

    if count:
        print(f"\nEmbedded {count} chunk(s).")
    else:
        print("\nNothing to embed — every chunk already has a vector.")


if __name__ == "__main__":
    main()
