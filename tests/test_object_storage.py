from __future__ import annotations

from pathlib import Path

import pytest

from rag_assistant.indexing.storage import LocalObjectStore, ObjectStoreError


async def test_local_object_store_is_confined_to_its_root(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    root.mkdir()
    source = root / "sample.pdf"
    source.write_bytes(b"fixture")
    store = LocalObjectStore(root)

    assert await store.get("sample.pdf") == b"fixture"
    with pytest.raises(ObjectStoreError):
        await store.get("../outside.pdf")
