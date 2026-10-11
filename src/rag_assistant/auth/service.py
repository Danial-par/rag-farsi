from __future__ import annotations

import hmac
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from rag_assistant.core.config import Settings
from rag_assistant.core.security import InvalidTokenError, PasswordManager, TokenManager
from rag_assistant.db.base import utc_now
from rag_assistant.db.models import AuthSession, User
from rag_assistant.db.repositories import AuthSessionRepository, UserRepository


class AuthenticationError(ValueError):
    """Raised for authentication failures safe to expose as a generic 401."""


class CsrfValidationError(ValueError):
    """Raised when an unsafe cookie-authenticated request lacks CSRF proof."""


class RefreshStatus(StrEnum):
    ROTATED = "rotated"
    CONCURRENT_REPLAY = "concurrent_replay"
    REUSE_DETECTED = "reuse_detected"


@dataclass(frozen=True, slots=True)
class AuthResult:
    user: User
    access_token: str
    refresh_token: str
    csrf_token: str
    session_expires_at: datetime


@dataclass(frozen=True, slots=True)
class RefreshResult:
    status: RefreshStatus
    access_token: str | None = None
    refresh_token: str | None = None
    csrf_token: str | None = None
    session_expires_at: datetime | None = None


class AuthService:
    """Coordinate password authentication, sessions, JWTs, rotation, and revocation."""

    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        password_manager: PasswordManager,
        token_manager: TokenManager,
    ) -> None:
        self._session = session
        self._settings = settings
        self._passwords = password_manager
        self._tokens = token_manager
        self._users = UserRepository(session)
        self._sessions = AuthSessionRepository(session)

    async def register(
        self, *, email: str, password: str, client_metadata: dict[str, Any]
    ) -> AuthResult:
        user = await self._users.create(
            email=email,
            password_hash=self._passwords.hash(password),
        )
        return await self._create_session(user=user, client_metadata=client_metadata)

    async def login(
        self, *, email: str, password: str, client_metadata: dict[str, Any]
    ) -> AuthResult:
        user = await self._users.get_by_normalized_email(email)
        if user is None:
            self._passwords.verify_dummy(password)
            raise AuthenticationError("Invalid email or password")
        if not self._passwords.verify(user.password_hash, password) or not user.is_active:
            raise AuthenticationError("Invalid email or password")
        if self._passwords.needs_rehash(user.password_hash):
            user.password_hash = self._passwords.hash(password)
        return await self._create_session(user=user, client_metadata=client_metadata)

    async def authenticate_access(
        self,
        raw_token: str,
        *,
        require_csrf: bool = False,
        csrf_cookie: str | None = None,
        csrf_header: str | None = None,
    ) -> User:
        try:
            claims = self._tokens.decode(raw_token, expected_type="access")
        except InvalidTokenError as exc:
            raise AuthenticationError("Not authenticated") from exc

        auth_session = await self._sessions.get_active(claims.session_id)
        if auth_session is None or auth_session.user_id != claims.user_id:
            raise AuthenticationError("Not authenticated")
        user = await self._users.get_by_id(claims.user_id)
        if user is None or not user.is_active:
            raise AuthenticationError("Not authenticated")
        if require_csrf:
            self._validate_csrf(auth_session, csrf_cookie, csrf_header)
        return user

    async def refresh(
        self,
        raw_token: str,
        *,
        csrf_cookie: str | None,
        csrf_header: str | None,
    ) -> RefreshResult:
        try:
            claims = self._tokens.decode(raw_token, expected_type="refresh")
        except InvalidTokenError as exc:
            raise AuthenticationError("Invalid authentication session") from exc
        if claims.token_id is None:
            raise AuthenticationError("Invalid authentication session")

        now = utc_now()
        auth_session = await self._sessions.get_for_update(claims.session_id)
        user = await self._users.get_by_id(claims.user_id)
        if (
            auth_session is None
            or user is None
            or not user.is_active
            or auth_session.user_id != claims.user_id
            or auth_session.revoked_at is not None
            or auth_session.expires_at <= now
        ):
            raise AuthenticationError("Invalid authentication session")
        self._validate_csrf(auth_session, csrf_cookie, csrf_header)

        presented_hash = self._tokens.hash_identifier(claims.token_id)
        if hmac.compare_digest(presented_hash, auth_session.refresh_token_hash):
            new_token_id = self._tokens.new_token_id()
            auth_session.previous_refresh_token_hash = auth_session.refresh_token_hash
            auth_session.previous_refresh_token_valid_until = now + timedelta(
                seconds=self._settings.refresh_reuse_grace_seconds
            )
            auth_session.refresh_token_hash = self._tokens.hash_identifier(new_token_id)
            await self._session.flush()
            return RefreshResult(
                status=RefreshStatus.ROTATED,
                access_token=self._tokens.create_access_token(
                    user_id=user.id,
                    session_id=auth_session.id,
                    issued_at=now,
                    session_expires_at=auth_session.expires_at,
                ),
                refresh_token=self._tokens.create_refresh_token(
                    user_id=user.id,
                    session_id=auth_session.id,
                    token_id=new_token_id,
                    issued_at=now,
                    expires_at=auth_session.expires_at,
                ),
                csrf_token=csrf_cookie,
                session_expires_at=auth_session.expires_at,
            )

        previous_hash = auth_session.previous_refresh_token_hash
        previous_valid_until = auth_session.previous_refresh_token_valid_until
        if (
            previous_hash is not None
            and previous_valid_until is not None
            and hmac.compare_digest(presented_hash, previous_hash)
            and previous_valid_until > now
        ):
            return RefreshResult(status=RefreshStatus.CONCURRENT_REPLAY)

        auth_session.revoked_at = now
        auth_session.revocation_reason = "refresh_token_reuse"
        await self._session.flush()
        return RefreshResult(status=RefreshStatus.REUSE_DETECTED)

    async def logout(
        self,
        raw_token: str,
        *,
        csrf_cookie: str | None,
        csrf_header: str | None,
    ) -> None:
        try:
            claims = self._tokens.decode(raw_token, expected_type="refresh")
        except InvalidTokenError as exc:
            raise AuthenticationError("Invalid authentication session") from exc
        auth_session = await self._sessions.get_for_update(claims.session_id)
        if (
            auth_session is None
            or auth_session.user_id != claims.user_id
            or auth_session.revoked_at is not None
        ):
            raise AuthenticationError("Invalid authentication session")
        self._validate_csrf(auth_session, csrf_cookie, csrf_header)
        auth_session.revoked_at = utc_now()
        auth_session.revocation_reason = "logout"
        await self._session.flush()

    async def _create_session(self, *, user: User, client_metadata: dict[str, Any]) -> AuthResult:
        now = utc_now()
        expires_at = now + timedelta(days=self._settings.refresh_token_ttl_days)
        refresh_token_id = self._tokens.new_token_id()
        csrf_token = self._tokens.new_csrf_token()
        auth_session = await self._sessions.create(
            user_id=user.id,
            expires_at=expires_at,
            refresh_token_hash=self._tokens.hash_identifier(refresh_token_id),
            csrf_token_hash=self._tokens.hash_identifier(csrf_token),
            client_metadata=client_metadata,
        )
        return AuthResult(
            user=user,
            access_token=self._tokens.create_access_token(
                user_id=user.id,
                session_id=auth_session.id,
                issued_at=now,
                session_expires_at=expires_at,
            ),
            refresh_token=self._tokens.create_refresh_token(
                user_id=user.id,
                session_id=auth_session.id,
                token_id=refresh_token_id,
                issued_at=now,
                expires_at=expires_at,
            ),
            csrf_token=csrf_token,
            session_expires_at=expires_at,
        )

    def _validate_csrf(
        self,
        auth_session: AuthSession,
        csrf_cookie: str | None,
        csrf_header: str | None,
    ) -> None:
        if csrf_cookie is None or csrf_header is None:
            raise CsrfValidationError("CSRF validation failed")
        if not hmac.compare_digest(csrf_cookie, csrf_header):
            raise CsrfValidationError("CSRF validation failed")
        if not hmac.compare_digest(
            self._tokens.hash_identifier(csrf_cookie), auth_session.csrf_token_hash
        ):
            raise CsrfValidationError("CSRF validation failed")
