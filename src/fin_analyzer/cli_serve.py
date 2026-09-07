"""`uv run serve` — starts the FastAPI server."""

import uvicorn


def main() -> None:
    uvicorn.run("fin_analyzer.api.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
