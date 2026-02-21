from fastapi import APIRouter
from backend.api import chat

api_router = APIRouter()

api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
