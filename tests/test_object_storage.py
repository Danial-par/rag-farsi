from __future__ import annotations

from datetime import UTC
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


async def test_local_object_store_writes_immutable_objects_and_lists_them(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    await store.put(
        "documents/user/document/version.pdf", b"%PDF-test", content_type="application/pdf"
    )

    assert await store.get("documents/user/document/version.pdf") == b"%PDF-test"
    listed = await store.list("documents/", limit=10)
    assert [item.key for item in listed.objects] == ["documents/user/document/version.pdf"]
    assert listed.objects[0].last_modified.tzinfo == UTC
    with pytest.raises(ObjectStoreError):
        await store.put(
            "documents/user/document/version.pdf", b"replacement", content_type="application/pdf"
        )

    await store.delete("documents/user/document/version.pdf")
    await store.delete("documents/user/document/version.pdf")
    assert (await store.list("documents/")).objects == []


async def test_local_store_rejects_path_traversal_for_all_write_operations(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    with pytest.raises(ObjectStoreError):
        await store.put("../outside.pdf", b"%PDF-test", content_type="application/pdf")
    with pytest.raises(ObjectStoreError):
        await store.delete("../outside.pdf")
    with pytest.raises(ObjectStoreError):
        await store.list("../")


async def test_local_object_listing_supports_bounded_continuation(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        await store.put(f"documents/{name}", b"fixture", content_type="application/pdf")

    first = await store.list("documents/", limit=1)
    second = await store.list("documents/", limit=1, continuation_token=first.continuation_token)
    third = await store.list("documents/", limit=1, continuation_token=second.continuation_token)

    assert [first.objects[0].key, second.objects[0].key, third.objects[0].key] == [
        "documents/a.pdf",
        "documents/b.pdf",
        "documents/c.pdf",
    ]
    assert third.continuation_token is None
