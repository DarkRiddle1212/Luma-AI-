"""
Unit tests for TeacherController.

Tests verify:
- POST /teacher/start with valid body → calls service.start_teacher_mode(), returns HTTP 200 TeacherResponse
- POST /teacher/continue with valid body → calls service.continue_teacher_mode(), returns HTTP 200 TeacherResponse
- Missing or empty user_id or topic → HTTP 422
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from luma.api.controllers.teacher_controller import router
from luma.api.controllers.chat_controller import get_luma_service


def make_mock_session():
    """Create a mock TeachingSession."""
    session = MagicMock()
    session.session_id = "sess-abc"
    session.status = "active"
    session.lessons = []
    session.explanations = []
    session.exercises = []
    return session


def make_mock_service():
    """Create a mock LumaService with async teacher methods."""
    service = MagicMock()
    service.start_teacher_mode = AsyncMock(return_value=make_mock_session())
    service.continue_teacher_mode = AsyncMock(return_value=make_mock_session())
    return service


def make_app(mock_service=None):
    """Create a minimal FastAPI app with the teacher router and a mock service."""
    app = FastAPI()
    app.include_router(router)
    if mock_service is not None:
        app.dependency_overrides[get_luma_service] = lambda: mock_service
    return app


# ---------------------------------------------------------------------------
# POST /teacher/start
# ---------------------------------------------------------------------------

class TestTeacherStart:
    def test_valid_request_returns_200(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/teacher/start", json={"user_id": "alice", "topic": "python"})
        assert response.status_code == 200

    def test_valid_request_calls_start_teacher_mode(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        client.post("/teacher/start", json={"user_id": "alice", "topic": "python"})
        mock_service.start_teacher_mode.assert_called_once_with("alice", "python")

    def test_valid_request_returns_teacher_response_shape(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/teacher/start", json={"user_id": "alice", "topic": "python"})
        body = response.json()
        assert "session_id" in body
        assert "status" in body
        assert "lessons" in body
        assert "explanations" in body
        assert "exercises" in body

    def test_valid_request_response_content(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/teacher/start", json={"user_id": "alice", "topic": "python"})
        body = response.json()
        assert body["session_id"] == "sess-abc"
        assert body["status"] == "active"
        assert body["lessons"] == []

    def test_missing_user_id_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/teacher/start", json={"topic": "python"})
        assert response.status_code == 422

    def test_missing_topic_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/teacher/start", json={"user_id": "alice"})
        assert response.status_code == 422

    def test_empty_user_id_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/teacher/start", json={"user_id": "", "topic": "python"})
        assert response.status_code == 422

    def test_empty_topic_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/teacher/start", json={"user_id": "alice", "topic": ""})
        assert response.status_code == 422

    def test_whitespace_user_id_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/teacher/start", json={"user_id": "   ", "topic": "python"})
        assert response.status_code == 422

    def test_whitespace_topic_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/teacher/start", json={"user_id": "alice", "topic": "\t"})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /teacher/continue
# ---------------------------------------------------------------------------

class TestTeacherContinue:
    def test_valid_request_returns_200(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/teacher/continue", json={"user_id": "alice", "topic": "python"})
        assert response.status_code == 200

    def test_valid_request_calls_continue_teacher_mode(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        client.post("/teacher/continue", json={"user_id": "alice", "topic": "python"})
        mock_service.continue_teacher_mode.assert_called_once_with("alice", "python")

    def test_valid_request_returns_teacher_response_shape(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/teacher/continue", json={"user_id": "alice", "topic": "python"})
        body = response.json()
        assert "session_id" in body
        assert "status" in body
        assert "lessons" in body
        assert "explanations" in body
        assert "exercises" in body

    def test_missing_user_id_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/teacher/continue", json={"topic": "python"})
        assert response.status_code == 422

    def test_missing_topic_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/teacher/continue", json={"user_id": "alice"})
        assert response.status_code == 422

    def test_empty_user_id_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/teacher/continue", json={"user_id": "", "topic": "python"})
        assert response.status_code == 422

    def test_empty_topic_returns_422(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.post("/teacher/continue", json={"user_id": "alice", "topic": ""})
        assert response.status_code == 422

    def test_start_mode_not_called_on_continue(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        client.post("/teacher/continue", json={"user_id": "alice", "topic": "python"})
        mock_service.start_teacher_mode.assert_not_called()

    def test_continue_mode_not_called_on_start(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        client.post("/teacher/start", json={"user_id": "alice", "topic": "python"})
        mock_service.continue_teacher_mode.assert_not_called()
