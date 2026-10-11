from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.exc import OperationalError


class _SessionContext:
    def __init__(self, session: object) -> None:
        self.session = session

    async def __aenter__(self) -> object:
        return self.session

    async def __aexit__(self, *args: object) -> None:
        return None


class _HealthySession:
    async def execute(self, statement: object) -> None:
        return None


class _UnavailableSession:
    async def execute(self, statement: object) -> None:
        raise OperationalError("SELECT 1", {}, ConnectionError("offline"))


def _session_factory(session: object) -> Callable[[], _SessionContext]:
    return lambda: _SessionContext(session)


async def test_liveness_does_not_require_the_database(client: AsyncClient) -> None:
    response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readiness_succeeds_when_database_answers(
    client: AsyncClient,
    started_application: FastAPI,
) -> None:
    started_application.state.session_factory = _session_factory(_HealthySession())

    response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


async def test_readiness_fails_closed_without_database(
    client: AsyncClient,
    started_application: FastAPI,
) -> None:
    started_application.state.session_factory = _session_factory(_UnavailableSession())

    response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}


async def test_safe_request_id_is_propagated(client: AsyncClient) -> None:
    response = await client.get("/health/live", headers={"X-Request-ID": "gateway-123"})

    assert response.headers["X-Request-ID"] == "gateway-123"


async def test_unsafe_request_id_is_replaced(client: AsyncClient) -> None:
    response = await client.get("/health/live", headers={"X-Request-ID": "unsafe request id"})

    UUID(response.headers["X-Request-ID"])
