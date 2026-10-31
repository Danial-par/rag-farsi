from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, cast
from uuid import UUID

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from rag_assistant.core.config import Settings

TokenType = Literal["access", "refresh"]


class InvalidTokenError(ValueError):
    """Raised when a JWT cannot be trusted for the requested operation."""


class PasswordManager:
    """Hash and verify passwords with Argon2id."""

    def __init__(self) -> None:
        self._hasher = PasswordHasher()
        self._dummy_hash = self._hasher.hash(secrets.token_urlsafe(32))

    def hash(self, password: str) -> str:
        return self._hasher.hash(password)

    def verify(self, password_hash: str, password: str) -> bool:
        try:
            return self._hasher.verify(password_hash, password)
        except (InvalidHashError, VerificationError, VerifyMismatchError):
            return False

    def verify_dummy(self, password: str) -> None:
        """Spend a normal verification cost when the account does not exist."""
        self.verify(self._dummy_hash, password)

    def needs_rehash(self, password_hash: str) -> bool:
        try:
            return self._hasher.check_needs_rehash(password_hash)
        except InvalidHashError:
            return True


@dataclass(frozen=True, slots=True)
class TokenClaims:
    user_id: UUID
    session_id: UUID
    token_type: TokenType
    issued_at: datetime
    expires_at: datetime
    token_id: str | None = None


class TokenManager:
    """Create and validate the application's strictly configured HS256 JWTs."""

    algorithm = "HS256"

    def __init__(self, settings: Settings) -> None:
        self._secret = settings.jwt_secret.get_secret_value()
        self._issuer = settings.jwt_issuer
        self._audience = settings.jwt_audience
        self._access_ttl = timedelta(seconds=settings.access_token_ttl_seconds)

    def new_token_id(self) -> str:
        return secrets.token_urlsafe(32)

    def hash_identifier(self, value: str) -> str:
        return hmac.new(
            self._secret.encode("utf-8"),
            value.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def create_access_token(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        issued_at: datetime,
        session_expires_at: datetime,
    ) -> str:
        expires_at = min(issued_at + self._access_ttl, session_expires_at)
        return self._encode(
            user_id=user_id,
            session_id=session_id,
            token_type="access",
            issued_at=issued_at,
            expires_at=expires_at,
        )

    def create_refresh_token(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        token_id: str,
        issued_at: datetime,
        expires_at: datetime,
    ) -> str:
        return self._encode(
            user_id=user_id,
            session_id=session_id,
            token_type="refresh",
            issued_at=issued_at,
            expires_at=expires_at,
            token_id=token_id,
        )

    def decode(self, token: str, *, expected_type: TokenType) -> TokenClaims:
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[self.algorithm],
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["sub", "sid", "typ", "iat", "exp", "iss", "aud"]},
            )
            token_type = payload.get("typ")
            if token_type != expected_type:
                raise InvalidTokenError("Unexpected token type")
            user_id = UUID(str(payload["sub"]))
            session_id = UUID(str(payload["sid"]))
            issued_at = datetime.fromtimestamp(float(payload["iat"]), tz=UTC)
            expires_at = datetime.fromtimestamp(float(payload["exp"]), tz=UTC)
            token_id_value = payload.get("jti")
            if expected_type == "refresh" and not isinstance(token_id_value, str):
                raise InvalidTokenError("Refresh token identifier is missing")
        except (jwt.InvalidTokenError, KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, InvalidTokenError):
                raise
            raise InvalidTokenError("Invalid authentication token") from exc

        return TokenClaims(
            user_id=user_id,
            session_id=session_id,
            token_type=cast(TokenType, token_type),
            issued_at=issued_at,
            expires_at=expires_at,
            token_id=cast(str | None, token_id_value),
        )

    def _encode(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        token_type: TokenType,
        issued_at: datetime,
        expires_at: datetime,
        token_id: str | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "sub": str(user_id),
            "sid": str(session_id),
            "typ": token_type,
            "iat": issued_at,
            "exp": expires_at,
            "iss": self._issuer,
            "aud": self._audience,
        }
        if token_id is not None:
            payload["jti"] = token_id
        return jwt.encode(payload, self._secret, algorithm=self.algorithm)
