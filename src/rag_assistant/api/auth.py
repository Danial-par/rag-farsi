from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from rag_assistant.auth.service import (
    AuthenticationError,
    AuthResult,
    AuthService,
    CsrfValidationError,
    RefreshResult,
    RefreshStatus,
)
from rag_assistant.core.config import Settings
from rag_assistant.db.models import User
from rag_assistant.db.repositories import PersistenceConflictError
from rag_assistant.db.session import get_db_session

router = APIRouter(prefix="/auth", tags=["authentication"])

ACCESS_COOKIE_NAME = "rag_access_token"
REFRESH_COOKIE_NAME = "rag_refresh_token"
CSRF_COOKIE_NAME = "rag_csrf_token"
CSRF_HEADER_NAME = "X-CSRF-Token"
REFRESH_COOKIE_PATH = "/api/v1/auth"
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class Credentials(BaseModel):
    email: EmailStr
    password: SecretStr = Field(min_length=12, max_length=128)


class UserResponse(BaseModel):
    id: UUID
    email: str

    @classmethod
    def from_user(cls, user: User) -> UserResponse:
        return cls(id=user.id, email=user.email)


def get_auth_service(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> AuthService:
    return AuthService(
        session,
        request.app.state.settings,
        request.app.state.password_manager,
        request.app.state.token_manager,
    )


async def get_current_user(
    request: Request,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> User:
    """Load the active user and session represented by the access cookie."""
    raw_token = request.cookies.get(ACCESS_COOKIE_NAME)
    if raw_token is None:
        raise _unauthorized("Not authenticated")
    try:
        return await service.authenticate_access(
            raw_token,
            require_csrf=request.method in UNSAFE_METHODS,
            csrf_cookie=request.cookies.get(CSRF_COOKIE_NAME),
            csrf_header=request.headers.get(CSRF_HEADER_NAME),
        )
    except CsrfValidationError as exc:
        raise _csrf_forbidden() from exc
    except AuthenticationError as exc:
        raise _unauthorized("Not authenticated") from exc


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    credentials: Credentials,
    request: Request,
    response: Response,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> UserResponse:
    try:
        async with session.begin():
            result = await service.register(
                email=str(credentials.email),
                password=credentials.password.get_secret_value(),
                client_metadata=_client_metadata(request),
            )
    except PersistenceConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Unable to register account",
            headers={"Cache-Control": "no-store"},
        ) from exc
    _set_auth_cookies(response, result, request.app.state.settings)
    _prevent_caching(response)
    return UserResponse.from_user(result.user)


@router.post("/login", response_model=UserResponse)
async def login(
    credentials: Credentials,
    request: Request,
    response: Response,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> UserResponse:
    try:
        async with session.begin():
            result = await service.login(
                email=str(credentials.email),
                password=credentials.password.get_secret_value(),
                client_metadata=_client_metadata(request),
            )
    except AuthenticationError as exc:
        raise _unauthorized("Invalid email or password") from exc
    _set_auth_cookies(response, result, request.app.state.settings)
    _prevent_caching(response)
    return UserResponse.from_user(result.user)


@router.post("/refresh", status_code=status.HTTP_204_NO_CONTENT)
async def refresh(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> Response:
    raw_token = request.cookies.get(REFRESH_COOKIE_NAME)
    if raw_token is None:
        return _auth_error_response(request.app.state.settings)
    try:
        async with session.begin():
            result = await service.refresh(
                raw_token,
                csrf_cookie=request.cookies.get(CSRF_COOKIE_NAME),
                csrf_header=request.headers.get(CSRF_HEADER_NAME),
            )
    except CsrfValidationError as exc:
        raise _csrf_forbidden() from exc
    except AuthenticationError:
        return _auth_error_response(request.app.state.settings)

    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    if result.status is RefreshStatus.ROTATED:
        _set_refreshed_cookies(response, result, request.app.state.settings)
    elif result.status is RefreshStatus.CONCURRENT_REPLAY:
        return _error_response(
            status_code=status.HTTP_409_CONFLICT,
            detail="Refresh already processed",
        )
    else:
        return _auth_error_response(request.app.state.settings)
    _prevent_caching(response)
    return response


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> Response:
    raw_token = request.cookies.get(REFRESH_COOKIE_NAME)
    if raw_token is None:
        raise _unauthorized("Invalid authentication session")
    try:
        async with session.begin():
            await service.logout(
                raw_token,
                csrf_cookie=request.cookies.get(CSRF_COOKIE_NAME),
                csrf_header=request.headers.get(CSRF_HEADER_NAME),
            )
    except CsrfValidationError as exc:
        raise _csrf_forbidden() from exc
    except AuthenticationError as exc:
        raise _unauthorized("Invalid authentication session") from exc
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    _clear_auth_cookies(response, request.app.state.settings)
    _prevent_caching(response)
    return response


@router.get("/me", response_model=UserResponse)
async def me(
    user: Annotated[User, Depends(get_current_user)],
    response: Response,
) -> UserResponse:
    _prevent_caching(response)
    return UserResponse.from_user(user)


def _set_auth_cookies(response: Response, result: AuthResult, settings: Settings) -> None:
    _set_session_cookies(
        response,
        access_token=result.access_token,
        refresh_token=result.refresh_token,
        csrf_token=result.csrf_token,
        session_expires_at=result.session_expires_at,
        settings=settings,
    )


def _set_session_cookies(
    response: Response,
    *,
    access_token: str,
    refresh_token: str,
    csrf_token: str,
    session_expires_at: datetime,
    settings: Settings,
) -> None:
    _set_cookie(
        response,
        key=ACCESS_COOKIE_NAME,
        value=access_token,
        max_age=settings.access_token_ttl_seconds,
        path="/",
        http_only=True,
        settings=settings,
    )
    refresh_max_age = _remaining_seconds(session_expires_at)
    _set_cookie(
        response,
        key=REFRESH_COOKIE_NAME,
        value=refresh_token,
        max_age=refresh_max_age,
        path=REFRESH_COOKIE_PATH,
        http_only=True,
        settings=settings,
    )
    _set_cookie(
        response,
        key=CSRF_COOKIE_NAME,
        value=csrf_token,
        max_age=refresh_max_age,
        path="/",
        http_only=False,
        settings=settings,
    )


def _set_refreshed_cookies(response: Response, result: RefreshResult, settings: Settings) -> None:
    if (
        result.access_token is None
        or result.refresh_token is None
        or result.csrf_token is None
        or not isinstance(result.session_expires_at, datetime)
    ):
        raise RuntimeError("Rotated refresh result is incomplete")
    _set_session_cookies(
        response,
        access_token=result.access_token,
        refresh_token=result.refresh_token,
        csrf_token=result.csrf_token,
        session_expires_at=result.session_expires_at,
        settings=settings,
    )


def _set_cookie(
    response: Response,
    *,
    key: str,
    value: str,
    max_age: int,
    path: str,
    http_only: bool,
    settings: Settings,
) -> None:
    response.set_cookie(
        key=key,
        value=value,
        max_age=max_age,
        path=path,
        secure=settings.secure_cookies,
        httponly=http_only,
        samesite="lax",
    )


def _clear_auth_cookies(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        ACCESS_COOKIE_NAME,
        path="/",
        secure=settings.secure_cookies,
        httponly=True,
        samesite="lax",
    )
    response.delete_cookie(
        REFRESH_COOKIE_NAME,
        path=REFRESH_COOKIE_PATH,
        secure=settings.secure_cookies,
        httponly=True,
        samesite="lax",
    )
    response.delete_cookie(
        CSRF_COOKIE_NAME,
        path="/",
        secure=settings.secure_cookies,
        httponly=False,
        samesite="lax",
    )


def _remaining_seconds(expires_at: datetime) -> int:
    return max(0, int((expires_at - datetime.now(expires_at.tzinfo)).total_seconds()))


def _client_metadata(request: Request) -> dict[str, str]:
    user_agent = request.headers.get("User-Agent", "")
    return {"user_agent": user_agent[:512]} if user_agent else {}


def _prevent_caching(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"Cache-Control": "no-store"},
    )


def _csrf_forbidden() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="CSRF validation failed",
        headers={"Cache-Control": "no-store"},
    )


def _auth_error_response(settings: Settings) -> JSONResponse:
    response = _error_response(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication session",
    )
    _clear_auth_cookies(response, settings)
    return response


def _error_response(*, status_code: int, detail: str) -> JSONResponse:
    response = JSONResponse(status_code=status_code, content={"detail": detail})
    _prevent_caching(response)
    return response
