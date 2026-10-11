from __future__ import annotations

from datetime import timedelta
from http.cookies import SimpleCookie
from typing import cast
from uuid import UUID

import jwt
from httpx import AsyncClient
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from rag_assistant.api.auth import (
    ACCESS_COOKIE_NAME,
    CSRF_COOKIE_NAME,
    REFRESH_COOKIE_NAME,
    REFRESH_COOKIE_PATH,
)
from rag_assistant.core.config import Settings
from rag_assistant.core.security import TokenManager
from rag_assistant.db.base import utc_now
from rag_assistant.db.models import AuthSession, User

PASSWORD = "correct horse battery staple"


async def _register(client: AsyncClient, email: str = "User@Example.com") -> dict[str, str]:
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": PASSWORD},
    )
    assert response.status_code == 201
    return cast(dict[str, str], response.json())


def _csrf_header(client: AsyncClient) -> dict[str, str]:
    return {"X-CSRF-Token": client.cookies[CSRF_COOKIE_NAME]}


def _replace_cookie(client: AsyncClient, name: str, value: str, *, path: str) -> None:
    for cookie in list(client.cookies.jar):
        if cookie.name == name:
            client.cookies.delete(name, domain=cookie.domain, path=cookie.path)
    client.cookies.set(name, value, path=path)


async def test_registration_normalizes_email_and_rejects_duplicates(
    database_client: AsyncClient,
    database_connection: AsyncConnection,
) -> None:
    registered = await _register(database_client)

    duplicate = await database_client.post(
        "/api/v1/auth/register",
        json={"email": " user@example.COM ", "password": PASSWORD},
    )

    assert duplicate.status_code == 409
    assert duplicate.json() == {"detail": "Unable to register account"}
    async with AsyncSession(bind=database_connection) as session:
        user = await session.scalar(select(User).where(User.id == UUID(registered["id"])))
        assert user is not None
        assert user.normalized_email == "user@example.com"
        assert user.password_hash != PASSWORD
        assert user.password_hash.startswith("$argon2id$")


async def test_login_accepts_correct_password_and_conceals_failures(
    database_client: AsyncClient,
) -> None:
    await _register(database_client, "login@example.com")
    database_client.cookies.clear()

    wrong_password = await database_client.post(
        "/api/v1/auth/login",
        json={"email": "login@example.com", "password": "incorrect password value"},
    )
    missing_user = await database_client.post(
        "/api/v1/auth/login",
        json={"email": "missing@example.com", "password": "incorrect password value"},
    )
    success = await database_client.post(
        "/api/v1/auth/login",
        json={"email": "LOGIN@example.com", "password": PASSWORD},
    )

    assert wrong_password.status_code == missing_user.status_code == 401
    assert wrong_password.json() == missing_user.json() == {"detail": "Invalid email or password"}
    assert PASSWORD not in wrong_password.text
    assert success.status_code == 200
    assert "token" not in success.json()

    exposed_password = "sekret"
    invalid = await database_client.post(
        "/api/v1/auth/login",
        json={"email": "login@example.com", "password": exposed_password},
    )
    assert invalid.status_code == 422
    assert exposed_password not in invalid.text


async def test_access_cookie_authenticates_me_and_rejects_invalid_tokens(
    database_client: AsyncClient,
) -> None:
    registered = await _register(database_client, "me@example.com")
    authenticated = await database_client.get("/api/v1/auth/me")
    assert authenticated.status_code == 200
    assert authenticated.json() == registered

    _replace_cookie(database_client, ACCESS_COOKIE_NAME, "malformed", path="/")
    malformed = await database_client.get("/api/v1/auth/me")
    assert malformed.status_code == 401
    assert "malformed" not in malformed.text


async def test_refresh_rotates_tokens_and_requires_csrf(database_client: AsyncClient) -> None:
    await _register(database_client, "refresh@example.com")
    old_refresh = database_client.cookies[REFRESH_COOKIE_NAME]

    missing_csrf = await database_client.post("/api/v1/auth/refresh")
    invalid_csrf = await database_client.post(
        "/api/v1/auth/refresh", headers={"X-CSRF-Token": "wrong"}
    )
    rotated = await database_client.post(
        "/api/v1/auth/refresh", headers=_csrf_header(database_client)
    )

    assert missing_csrf.status_code == invalid_csrf.status_code == 403
    assert rotated.status_code == 204
    assert database_client.cookies[REFRESH_COOKIE_NAME] != old_refresh
    assert database_client.cookies[ACCESS_COOKIE_NAME]


async def test_refresh_grace_then_reuse_revokes_only_affected_session(
    database_client: AsyncClient,
    database_connection: AsyncConnection,
    settings: Settings,
) -> None:
    await _register(database_client, "sessions@example.com")
    first_refresh = database_client.cookies[REFRESH_COOKIE_NAME]
    csrf = database_client.cookies[CSRF_COOKIE_NAME]
    first_claims = TokenManager(settings).decode(first_refresh, expected_type="refresh")

    rotated = await database_client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": csrf})
    current_refresh = database_client.cookies[REFRESH_COOKIE_NAME]
    assert rotated.status_code == 204

    _replace_cookie(database_client, REFRESH_COOKIE_NAME, first_refresh, path=REFRESH_COOKIE_PATH)
    concurrent = await database_client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": csrf})
    assert concurrent.status_code == 409

    async with AsyncSession(bind=database_connection) as session, session.begin():
        await session.execute(
            update(AuthSession)
            .where(AuthSession.id == first_claims.session_id)
            .values(previous_refresh_token_valid_until=utc_now() - timedelta(seconds=1))
        )
    reused = await database_client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": csrf})
    assert reused.status_code == 401

    _replace_cookie(database_client, REFRESH_COOKIE_NAME, current_refresh, path=REFRESH_COOKIE_PATH)
    revoked_family = await database_client.post(
        "/api/v1/auth/refresh", headers={"X-CSRF-Token": csrf}
    )
    assert revoked_family.status_code == 401

    database_client.cookies.clear()
    second_login = await database_client.post(
        "/api/v1/auth/login",
        json={"email": "sessions@example.com", "password": PASSWORD},
    )
    assert second_login.status_code == 200
    assert (await database_client.get("/api/v1/auth/me")).status_code == 200


async def test_logout_revokes_session_and_clears_cookies(database_client: AsyncClient) -> None:
    await _register(database_client, "logout@example.com")
    access_token = database_client.cookies[ACCESS_COOKIE_NAME]

    logged_out = await database_client.post(
        "/api/v1/auth/logout", headers=_csrf_header(database_client)
    )
    _replace_cookie(database_client, ACCESS_COOKIE_NAME, access_token, path="/")
    after_logout = await database_client.get("/api/v1/auth/me")

    assert logged_out.status_code == 204
    assert after_logout.status_code == 401


async def test_wrong_type_expired_and_inactive_tokens_are_rejected(
    database_client: AsyncClient,
    database_connection: AsyncConnection,
    settings: Settings,
) -> None:
    user = await _register(database_client, "inactive@example.com")
    refresh_token = database_client.cookies[REFRESH_COOKIE_NAME]
    refresh_claims = TokenManager(settings).decode(refresh_token, expected_type="refresh")

    _replace_cookie(database_client, ACCESS_COOKIE_NAME, refresh_token, path="/")
    assert (await database_client.get("/api/v1/auth/me")).status_code == 401

    now = utc_now()
    expired = jwt.encode(
        {
            "sub": user["id"],
            "sid": str(refresh_claims.session_id),
            "typ": "access",
            "iat": now - timedelta(hours=2),
            "exp": now - timedelta(hours=1),
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
        },
        settings.jwt_secret.get_secret_value(),
        algorithm="HS256",
    )
    _replace_cookie(database_client, ACCESS_COOKIE_NAME, expired, path="/")
    assert (await database_client.get("/api/v1/auth/me")).status_code == 401

    async with AsyncSession(bind=database_connection) as session, session.begin():
        await session.execute(
            update(User).where(User.id == UUID(user["id"])).values(is_active=False)
        )
    valid_access = TokenManager(settings).create_access_token(
        user_id=UUID(user["id"]),
        session_id=refresh_claims.session_id,
        issued_at=now,
        session_expires_at=now + timedelta(minutes=30),
    )
    _replace_cookie(database_client, ACCESS_COOKIE_NAME, valid_access, path="/")
    assert (await database_client.get("/api/v1/auth/me")).status_code == 401


async def test_cookie_attributes_are_scoped_and_environment_aware(
    database_client: AsyncClient,
) -> None:
    response = await database_client.post(
        "/api/v1/auth/register",
        json={"email": "cookies@example.com", "password": PASSWORD},
    )
    set_cookie_headers = response.headers.get_list("set-cookie")
    parsed = SimpleCookie()
    for header in set_cookie_headers:
        parsed.load(header)

    assert parsed[ACCESS_COOKIE_NAME]["path"] == "/"
    assert parsed[ACCESS_COOKIE_NAME]["httponly"] is True
    assert parsed[REFRESH_COOKIE_NAME]["path"] == REFRESH_COOKIE_PATH
    assert parsed[REFRESH_COOKIE_NAME]["httponly"] is True
    assert parsed[CSRF_COOKIE_NAME]["httponly"] == ""
    assert all(cookie["samesite"].lower() == "lax" for cookie in parsed.values())
    assert all(cookie["secure"] == "" for cookie in parsed.values())


async def test_multiple_users_remain_isolated(database_client: AsyncClient) -> None:
    first = await _register(database_client, "first@example.com")
    first_access = database_client.cookies[ACCESS_COOKIE_NAME]
    database_client.cookies.clear()
    second = await _register(database_client, "second@example.com")
    assert first["id"] != second["id"]

    _replace_cookie(database_client, ACCESS_COOKIE_NAME, first_access, path="/")
    current = await database_client.get("/api/v1/auth/me")
    assert current.status_code == 200
    assert current.json()["id"] == first["id"]
