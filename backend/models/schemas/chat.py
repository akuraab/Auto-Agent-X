from typing import Dict, List, Optional, Any
from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    stream: bool = False

class IntentInfo(BaseModel):
    intent: str
    confidence: float
    entities: Dict[str, Any]
    requires_clarification: bool

class Citation(BaseModel):
    source: str
    content: str
    relevance: float

class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: Optional[IntentInfo] = None
    citations: List[Citation] = []
    metadata: Dict[str, Any] = {}
