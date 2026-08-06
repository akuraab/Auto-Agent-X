import asyncio
import json
import logging
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.models.schemas.chat import ChatRequest, ChatResponse
from backend.services.rag_service import RAGService

router = APIRouter()
logger = logging.getLogger(__name__)

CHAT_ERROR_MESSAGE = "Chat service is temporarily unavailable."
STREAM_ERROR_MESSAGE = "The response stream ended unexpectedly. Please try again."


def get_rag_service() -> RAGService:
    return RAGService()


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _safe_chat_stream(
    service: RAGService,
    request: ChatRequest,
    session_id: str,
) -> AsyncIterator[str]:
    try:
        async for chunk in service.chat_stream(
            query=request.query,
            session_id=session_id,
            context=request.context,
        ):
            yield chunk
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Streaming chat request failed", extra={"session_id": session_id})
        yield _format_sse("error", {"message": STREAM_ERROR_MESSAGE})


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())

    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Use /api/v1/chat/stream for streaming requests.",
        )

    try:
        result = await service.chat(
            query=request.query,
            session_id=session_id,
            context=request.context,
        )
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Chat request failed", extra={"session_id": session_id})
        raise HTTPException(status_code=500, detail=CHAT_ERROR_MESSAGE) from None


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    session_id = request.session_id or str(uuid.uuid4())

    return StreamingResponse(
        _safe_chat_stream(service, request, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
