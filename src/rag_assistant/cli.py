from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Persian RAG Assistant API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    """Run Uvicorn with an event loop compatible with async psycopg on Windows."""
    args = _parser().parse_args(argv)
    uvicorn.run(
        "rag_assistant.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        loop="rag_assistant.core.event_loop:compatible_event_loop_factory",
        access_log=False,
    )
