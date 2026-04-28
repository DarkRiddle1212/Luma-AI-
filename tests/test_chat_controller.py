"""
Unit tests for ChatController.

Tests verify:
- Valid ChatRequest → delegates to LumaService.process_chat() and returns HTTP 200 ChatResponse
- Missing user_id or message → HTTP 422
- Empty user_id or message → HTTP 422
- LumaService is never called directly on core modules
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from luma.api.controllers.chat_controller import router, get_luma_service


def make_mock_service():
    """Create a mock LumaService with async methods."""
    service = MagicMock()
    service.process_chat = AsyncMock(return_value={
        "response": "Hello, alice!",
        "insight_moments": [],
        "personalization": {"tone": "casual", "style": "concise", "focus": "high-level", "reasons": {}},
    })
    return service


def make_app(mock_service=None):
    """Create a minimal FastAPI app with the chat router and a mock service."""
    app = FastAPI()
    app.include_router(router)
    if mock_service is not None:
        app.dependency_overrides[get_luma_service] = lambda: mock_service
    return app


# ---------------------------------------------------------------------------
# Valid request → HTTP 200 ChatResponse
# ---------------------------------------------------------------------------

class TestChatValidRequest:
    def test_valid_request_returns_200(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/chat", json={"user_id": "alice", "message": "hello"})
        assert response.status_code == 200

    def test_valid_request_delegates_to_process_chat(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        client.post("/chat", json={"user_id": "alice", "message": "hello"})
        mock_service.process_chat.assert_called_once_with("alice", "hello")

    def test_valid_request_returns_chat_response_shape(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/chat", json={"user_id": "alice", "message": "hello"})
        body = response.json()
        assert "response" in body
        assert "insight_moments" in body
        assert "personalization" in body

    def test_valid_request_response_content(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/chat", json={"user_id": "alice", "message": "hello"})
        body = response.json()
        assert body["response"] == "Hello, alice!"
        assert body["insight_moments"] == []


# ---------------------------------------------------------------------------
# Missing fields → HTTP 422
# ---------------------------------------------------------------------------

class TestChatMissingFields:
    def test_missing_user_id_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/chat", json={"message": "hello"})
        assert response.status_code == 422

    def test_missing_message_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/chat", json={"user_id": "alice"})
        assert response.status_code == 422

    def test_missing_both_fields_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_missing_user_id_does_not_call_service(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        client.post("/chat", json={"message": "hello"})
        mock_service.process_chat.assert_not_called()


# ---------------------------------------------------------------------------
# Empty / whitespace fields → HTTP 422
# ---------------------------------------------------------------------------

class TestChatEmptyFields:
    def test_empty_user_id_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/chat", json={"user_id": "", "message": "hello"})
        assert response.status_code == 422

    def test_empty_message_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/chat", json={"user_id": "alice", "message": ""})
        assert response.status_code == 422

    def test_whitespace_user_id_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/chat", json={"user_id": "   ", "message": "hello"})
        assert response.status_code == 422

    def test_whitespace_message_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/chat", json={"user_id": "alice", "message": "\t\n"})
        assert response.status_code == 422

    def test_empty_fields_do_not_call_service(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        client.post("/chat", json={"user_id": "", "message": ""})
        mock_service.process_chat.assert_not_called()
