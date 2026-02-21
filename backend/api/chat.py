from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from backend.models.schemas.chat import ChatRequest, ChatResponse
from backend.services.rag_service import RAGService
import uuid

router = APIRouter()

def get_rag_service():
    return RAGService()

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: RAGService = Depends(get_rag_service)
):
    session_id = request.session_id or str(uuid.uuid4())
    
    if request.stream:
        # TODO: Handle streaming response properly
        # For now, if stream is requested, we should return StreamingResponse
        # But the response model is ChatResponse.
        # This endpoint definition might need to be split or return type adjusted.
        pass

    try:
        result = await service.chat(
            query=request.query,
            session_id=session_id,
            context=request.context
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: RAGService = Depends(get_rag_service)
):
    session_id = request.session_id or str(uuid.uuid4())
    
    return StreamingResponse(
        service.chat_stream(
            query=request.query,
            session_id=session_id,
            context=request.context
        ),
        media_type="text/event-stream"
    )
