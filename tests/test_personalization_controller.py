"""
Unit tests for PersonalizationController.

Tests verify:
- GET /personalization?user_id=alice → calls service.get_personalization("alice"), returns HTTP 200
- Missing user_id query param → HTTP 422
- Empty user_id query param → HTTP 422
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from luma.api.controllers.personalization_controller import router
from luma.api.controllers.chat_controller import get_luma_service


def make_mock_ctx():
    """Create a mock AdaptationContext."""
    ctx = MagicMock()
    ctx.tone = "casual"
    ctx.style = "concise"
    ctx.focus = "high-level"
    ctx.reasons = {"tone": "user prefers casual"}
    return ctx


def make_mock_service():
    """Create a mock LumaService with async personalization method."""
    service = MagicMock()
    service.get_personalization = AsyncMock(return_value=make_mock_ctx())
    return service


def make_app(mock_service=None):
    """Create a minimal FastAPI app with the personalization router and a mock service."""
    app = FastAPI()
    app.include_router(router)
    if mock_service is not None:
        app.dependency_overrides[get_luma_service] = lambda: mock_service
    return app


# ---------------------------------------------------------------------------
# GET /personalization?user_id=alice
# ---------------------------------------------------------------------------

class TestGetPersonalization:
    def test_valid_user_id_returns_200(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.get("/personalization?user_id=alice")
        assert response.status_code == 200

    def test_valid_user_id_calls_get_personalization(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        client.get("/personalization?user_id=alice")
        mock_service.get_personalization.assert_called_once_with("alice")

    def test_valid_user_id_returns_personalization_response_shape(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.get("/personalization?user_id=alice")
        body = response.json()
        assert "tone" in body
        assert "style" in body
        assert "focus" in body
        assert "reasons" in body

    def test_valid_user_id_returns_correct_values(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.get("/personalization?user_id=alice")
        body = response.json()
        assert body["tone"] == "casual"
        assert body["style"] == "concise"
        assert body["focus"] == "high-level"
        assert body["reasons"] == {"tone": "user prefers casual"}

    def test_different_user_id_is_forwarded(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        client.get("/personalization?user_id=bob")
        mock_service.get_personalization.assert_called_once_with("bob")


# ---------------------------------------------------------------------------
# Missing user_id → HTTP 422
# ---------------------------------------------------------------------------

class TestMissingUserId:
    def test_missing_user_id_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.get("/personalization")
        assert response.status_code == 422

    def test_missing_user_id_does_not_call_service(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        client.get("/personalization")
        mock_service.get_personalization.assert_not_called()


# ---------------------------------------------------------------------------
# Empty user_id → HTTP 422
# ---------------------------------------------------------------------------

class TestEmptyUserId:
    def test_empty_user_id_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.get("/personalization?user_id=")
        assert response.status_code == 422

    def test_empty_user_id_does_not_call_service(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        client.get("/personalization?user_id=")
        mock_service.get_personalization.assert_not_called()
