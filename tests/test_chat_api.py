import json
from typing import List, Optional

import pytest
from fastapi.testclient import TestClient

from backend.api.chat import get_rag_service
from backend.main import app
from backend.models.schemas.chat import ChatRequest


class FakeRAGService:
    def __init__(
        self,
        *,
        chat_error: Optional[Exception] = None,
        stream_error: Optional[Exception] = None,
    ):
        self.chat_error = chat_error
        self.stream_error = stream_error
        self.queries: List[str] = []

    async def chat(self, query: str, session_id: str, **kwargs):
        self.queries.append(query)
        if self.chat_error:
            raise self.chat_error
        return {
            "session_id": session_id,
            "response": "ok",
            "citations": [],
            "metadata": {},
        }

    async def chat_stream(self, query: str, session_id: str, **kwargs):
        self.queries.append(query)
        yield 'event: status\ndata: {"stage": "generating"}\n\n'
        if self.stream_error:
            raise self.stream_error
        yield f"event: done\ndata: {json.dumps({'session_id': session_id, 'citations': []})}\n\n"


@pytest.fixture
def client_with_service():
    services: List[FakeRAGService] = []

    def make_client(service: FakeRAGService) -> TestClient:
        services.append(service)
        app.dependency_overrides[get_rag_service] = lambda: service
        return TestClient(app, raise_server_exceptions=False)

    yield make_client
    app.dependency_overrides.clear()
    for service in services:
        service.queries.clear()


def test_chat_request_strips_query_and_rejects_blank_input():
    assert ChatRequest(query="  explain this  ").query == "explain this"

    with pytest.raises(ValueError):
        ChatRequest(query="   ")

    with pytest.raises(ValueError):
        ChatRequest(query="x" * 10_001)


def test_non_stream_endpoint_rejects_stream_flag(client_with_service):
    service = FakeRAGService()
    client = client_with_service(service)

    response = client.post("/api/v1/chat/", json={"query": "hello", "stream": True})

    assert response.status_code == 400
    assert "/api/v1/chat/stream" in response.json()["detail"]
    assert service.queries == []


def test_non_stream_endpoint_hides_internal_errors(client_with_service):
    service = FakeRAGService(chat_error=RuntimeError("provider token abc123 was rejected"))
    client = client_with_service(service)

    response = client.post("/api/v1/chat/", json={"query": "hello"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Chat service is temporarily unavailable."}
    assert "abc123" not in response.text


def test_stream_endpoint_converts_generator_failure_to_safe_sse(client_with_service):
    service = FakeRAGService(stream_error=RuntimeError("provider token abc123 was rejected"))
    client = client_with_service(service)

    with client.stream("POST", "/api/v1/chat/stream", json={"query": "hello"}) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "event: status" in body
    assert "event: error" in body
    assert "The response stream ended unexpectedly." in body
    assert "abc123" not in body
