from fastapi import APIRouter

from rag_assistant.api.auth import router as auth_router
from rag_assistant.api.conversations import router as conversations_router
from rag_assistant.api.documents import router as documents_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth_router)
api_router.include_router(documents_router)
api_router.include_router(conversations_router)
