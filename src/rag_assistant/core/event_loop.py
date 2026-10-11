from __future__ import annotations

import asyncio
import sys


def compatible_event_loop_factory() -> asyncio.AbstractEventLoop:
    """Create an event loop supported by psycopg async on every platform."""
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()
