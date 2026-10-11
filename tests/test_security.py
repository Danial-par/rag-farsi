from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest
from fastapi import Response
from httpx import AsyncClient

from rag_assistant.api.auth import _set_cookie
from rag_assistant.core.config import Settings
from rag_assistant.core.security import InvalidTokenError, PasswordManager, TokenManager
from rag_assistant.db.base import utc_now


def test_argon2_password_hashing_and_verification() -> None:
    passwords = PasswordManager()
    password_hash = passwords.hash("a sufficiently long password")

    assert password_hash.startswith("$argon2id$")
    assert passwords.verify(password_hash, "a sufficiently long password") is True
    assert passwords.verify(password_hash, "a different long password") is False


def test_jwt_validation_pins_type_issuer_audience_and_expiry(settings: Settings) -> None:
    tokens = TokenManager(settings)
    now = utc_now()
    access = tokens.create_access_token(
        user_id=uuid4(),
        session_id=uuid4(),
        issued_at=now,
        session_expires_at=now + timedelta(hours=1),
    )

    claims = tokens.decode(access, expected_type="access")
    assert claims.token_type == "access"
    with pytest.raises(InvalidTokenError):
        tokens.decode(access, expected_type="refresh")

    expired = tokens.create_access_token(
        user_id=uuid4(),
        session_id=uuid4(),
        issued_at=now - timedelta(hours=2),
        session_expires_at=now - timedelta(hours=1),
    )
    with pytest.raises(InvalidTokenError):
        tokens.decode(expired, expected_type="access")


async def test_validation_errors_do_not_echo_passwords(client: AsyncClient) -> None:
    password = "sekret"
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "user@example.com", "password": password},
    )

    assert response.status_code == 422
    assert password not in response.text
    assert response.headers["Cache-Control"] == "no-store"


def test_production_cookie_has_secure_browser_attributes(settings: Settings) -> None:
    production = settings.model_copy(update={"app_env": "production"})
    response = Response()

    _set_cookie(
        response,
        key="test_cookie",
        value="value",
        max_age=60,
        path="/",
        http_only=True,
        settings=production,
    )

    header = response.headers["set-cookie"]
    assert "HttpOnly" in header
    assert "Path=/" in header
    assert "SameSite=lax" in header
    assert "Secure" in header
