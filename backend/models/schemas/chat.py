from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    stream: bool = False

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


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
    citations: List[Citation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
