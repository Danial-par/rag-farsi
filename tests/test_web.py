from __future__ import annotations

from pathlib import Path

from httpx import AsyncClient

WEB_ROOT = Path(__file__).parents[1] / "src" / "rag_assistant" / "web"


async def test_local_interface_is_server_rendered_and_available_without_auth(
    client: AsyncClient,
) -> None:
    response = await client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert '<html lang="fa" dir="rtl">' in response.text
    assert 'id="auth-view"' in response.text
    assert 'id="app-view"' in response.text


async def test_static_assets_are_served_by_fastapi(client: AsyncClient) -> None:
    stylesheet = await client.get("/static/app.css")
    script = await client.get("/static/app.js")

    assert stylesheet.status_code == script.status_code == 200
    assert "@media(max-width:760px)" in stylesheet.text
    assert "document-list" in script.text


def test_browser_client_obeys_security_and_chat_contracts() -> None:
    script = (WEB_ROOT / "static" / "app.js").read_text(encoding="utf-8")

    assert 'credentials: "same-origin"' in script
    assert 'headers.set("X-CSRF-Token", token)' in script
    assert '"Idempotency-Key": idempotencyKey' in script
    assert "[...(page.items || [])].reverse()" in script
    assert "node.textContent" in script
    assert 'marker.dir = "ltr"' in script
    assert "appendAnswerText(content" in script
    assert "message.answer_outcome" in script
    assert 'outcome === "abstained"' in script
    assert 'message.execution_status === "failed"' in script
    assert 'error.payload?.execution_status === "pending"' in script
    assert "citation.original_text" in script
    assert 'version?.status === "failed" && !doc.active_version_id' in script
