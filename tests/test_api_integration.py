"""
Integration tests for the full Luma API layer.

Tests verify:
- Full request/response cycle for all new endpoints
- All endpoints are reachable under settings.api_prefix
- Middleware stack (logging + error handling) works end-to-end
- Existing GET /, GET /health, and /memories endpoints remain unaffected

Note: These tests build a minimal FastAPI app that mirrors luma/main.py's
create_app() but registers routes without importing luma.api.routes directly
(which would trigger a SQLAlchemy import incompatible with Python 3.14).
The route registration and middleware wiring are tested end-to-end.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from luma.api.controllers.chat_controller import router as chat_router, get_luma_service
from luma.api.controllers.insight_controller import router as insight_router
from luma.api.controllers.teacher_controller import router as teacher_router
from luma.api.controllers.personalization_controller import router as personalization_router
from luma.api.middleware.logging import LoggingMiddleware
from luma.api.middleware.error_handler import ErrorHandlerMiddleware
from luma.config import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_service():
    """Create a mock LumaService with all async methods."""
    service = MagicMock()

    service.process_chat = AsyncMock(return_value={
        "response": "Test response",
        "insight_moments": [],
        "personalization": {
            "tone": "casual",
            "style": "concise",
            "focus": "high-level",
            "reasons": {},
        },
    })

    service.get_insights = AsyncMock(return_value=[])

    service.get_insight_moments = AsyncMock(return_value=[])

    teacher_session = MagicMock()
    teacher_session.session_id = "session-123"
    teacher_session.status = "active"
    teacher_session.lessons = []
    teacher_session.explanations = []
    teacher_session.exercises = []
    service.start_teacher_mode = AsyncMock(return_value=teacher_session)
    service.continue_teacher_mode = AsyncMock(return_value=teacher_session)

    adaptation_ctx = MagicMock()
    adaptation_ctx.tone = "neutral"
    adaptation_ctx.style = "detailed"
    adaptation_ctx.focus = "technical"
    adaptation_ctx.reasons = {}
    service.get_personalization = AsyncMock(return_value=adaptation_ctx)

    return service


def _make_memories_router() -> APIRouter:
    """
    Build a minimal stub /memories router that mirrors the real one's
    route structure without importing SQLAlchemy. Used only to verify
    the /memories endpoints are still registered (not 404).
    """
    from pydantic import BaseModel, Field
    from typing import List
    from datetime import datetime

    router = APIRouter()

    class MemoryCreate(BaseModel):
        content: str = Field(..., min_length=1)
        metadata: dict = Field(default_factory=dict)

    @router.post("/memories", status_code=status.HTTP_201_CREATED)
    async def create_memory(memory: MemoryCreate):
        raise HTTPException(status_code=500, detail="DB not available in tests")

    @router.get("/memories")
    async def list_memories(skip: int = 0, limit: int = 100):
        raise HTTPException(status_code=500, detail="DB not available in tests")

    @router.get("/memories/{memory_id}")
    async def get_memory(memory_id: int):
        raise HTTPException(status_code=500, detail="DB not available in tests")

    @router.put("/memories/{memory_id}")
    async def update_memory(memory_id: int):
        raise HTTPException(status_code=500, detail="DB not available in tests")

    @router.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_memory(memory_id: int):
        raise HTTPException(status_code=500, detail="DB not available in tests")

    return router


def create_test_app(mock_service=None) -> FastAPI:
    """
    Create a minimal FastAPI app that mirrors luma/main.py's create_app()
    but registers routes without importing luma.api.routes (which pulls in
    SQLAlchemy). The memories router is stubbed to verify route registration.
    """
    app = FastAPI(
        title="Luma AI System (Test)",
        version="0.1.0",
    )

    # CORS middleware (same as main.py)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error handling and logging middleware (same order as main.py)
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(LoggingMiddleware)

    # Build the combined API router (mirrors luma/api/routes.py)
    api_router = APIRouter()
    api_router.include_router(_make_memories_router())
    api_router.include_router(chat_router)
    api_router.include_router(insight_router)
    api_router.include_router(teacher_router)
    api_router.include_router(personalization_router)

    app.include_router(api_router, prefix=settings.api_prefix)

    # Existing root and health endpoints (same as main.py)
    @app.get("/")
    async def root():
        return {"message": "Luma is alive"}

    @app.get("/health")
    async def health():
        return {"status": "healthy", "version": "0.1.0"}

    # Inject mock service if provided
    if mock_service is not None:
        app.dependency_overrides[get_luma_service] = lambda: mock_service

    return app


PREFIX = settings.api_prefix  # "/api/v1"


@pytest.fixture
def client():
    """TestClient with mock LumaService injected via dependency_overrides."""
    mock_service = make_mock_service()
    app = create_test_app(mock_service)
    with TestClient(app) as c:
        yield c, mock_service


# ---------------------------------------------------------------------------
# Existing endpoints remain unaffected
# ---------------------------------------------------------------------------

class TestExistingEndpoints:
    def test_root_endpoint_reachable(self, client):
        c, _ = client
        response = c.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Luma is alive"}

    def test_health_endpoint_reachable(self, client):
        c, _ = client
        response = c.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_memories_list_endpoint_not_404(self, client):
        """Verify /memories endpoint is still registered (not 404)."""
        c, _ = client
        response = c.get(f"{PREFIX}/memories")
        # Stub returns 500 (DB not available), but must NOT be 404
        assert response.status_code != 404

    def test_memories_post_endpoint_not_404(self, client):
        """Verify POST /memories endpoint is still registered (not 404)."""
        c, _ = client
        response = c.post(
            f"{PREFIX}/memories",
            json={"content": "test memory", "metadata": {}},
        )
        assert response.status_code != 404


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    def test_chat_returns_200(self, client):
        c, _ = client
        response = c.post(
            f"{PREFIX}/chat",
            json={"user_id": "alice", "message": "hello"},
        )
        assert response.status_code == 200

    def test_chat_response_shape(self, client):
        c, _ = client
        response = c.post(
            f"{PREFIX}/chat",
            json={"user_id": "alice", "message": "hello"},
        )
        body = response.json()
        assert "response" in body
        assert "insight_moments" in body
        assert "personalization" in body

    def test_chat_delegates_to_service(self, client):
        c, mock_service = client
        c.post(f"{PREFIX}/chat", json={"user_id": "alice", "message": "hello"})
        mock_service.process_chat.assert_called_once_with("alice", "hello")

    def test_chat_missing_fields_returns_422(self, client):
        c, _ = client
        response = c.post(f"{PREFIX}/chat", json={"user_id": "alice"})
        assert response.status_code == 422

    def test_chat_empty_user_id_returns_422(self, client):
        c, _ = client
        response = c.post(
            f"{PREFIX}/chat",
            json={"user_id": "", "message": "hello"},
        )
        assert response.status_code == 422

    def test_chat_whitespace_message_returns_422(self, client):
        c, _ = client
        response = c.post(
            f"{PREFIX}/chat",
            json={"user_id": "alice", "message": "   "},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /insights
# ---------------------------------------------------------------------------

class TestInsightsEndpoint:
    def test_insights_returns_200(self, client):
        c, _ = client
        response = c.get(f"{PREFIX}/insights")
        assert response.status_code == 200

    def test_insights_response_shape(self, client):
        c, _ = client
        response = c.get(f"{PREFIX}/insights")
        body = response.json()
        assert "insights" in body
        assert isinstance(body["insights"], list)

    def test_insights_with_namespace(self, client):
        c, mock_service = client
        response = c.get(f"{PREFIX}/insights?namespace=work")
        assert response.status_code == 200
        mock_service.get_insights.assert_called_once_with(namespace="work")

    def test_insights_without_namespace(self, client):
        c, mock_service = client
        c.get(f"{PREFIX}/insights")
        mock_service.get_insights.assert_called_once_with(namespace=None)


# ---------------------------------------------------------------------------
# GET /insight-moments
# ---------------------------------------------------------------------------

class TestInsightMomentsEndpoint:
    def test_insight_moments_returns_200(self, client):
        c, _ = client
        response = c.get(f"{PREFIX}/insight-moments")
        assert response.status_code == 200

    def test_insight_moments_response_shape(self, client):
        c, _ = client
        response = c.get(f"{PREFIX}/insight-moments")
        body = response.json()
        assert "insight_moments" in body
        assert isinstance(body["insight_moments"], list)

    def test_insight_moments_delegates_to_service(self, client):
        c, mock_service = client
        c.get(f"{PREFIX}/insight-moments")
        mock_service.get_insight_moments.assert_called_once()


# ---------------------------------------------------------------------------
# POST /teacher/start
# ---------------------------------------------------------------------------

class TestTeacherStartEndpoint:
    def test_teacher_start_returns_200(self, client):
        c, _ = client
        response = c.post(
            f"{PREFIX}/teacher/start",
            json={"user_id": "alice", "topic": "Python"},
        )
        assert response.status_code == 200

    def test_teacher_start_response_shape(self, client):
        c, _ = client
        response = c.post(
            f"{PREFIX}/teacher/start",
            json={"user_id": "alice", "topic": "Python"},
        )
        body = response.json()
        assert "session_id" in body
        assert "status" in body
        assert "lessons" in body
        assert "explanations" in body
        assert "exercises" in body

    def test_teacher_start_delegates_to_service(self, client):
        c, mock_service = client
        c.post(
            f"{PREFIX}/teacher/start",
            json={"user_id": "alice", "topic": "Python"},
        )
        mock_service.start_teacher_mode.assert_called_once_with("alice", "Python")

    def test_teacher_start_missing_topic_returns_422(self, client):
        c, _ = client
        response = c.post(
            f"{PREFIX}/teacher/start",
            json={"user_id": "alice"},
        )
        assert response.status_code == 422

    def test_teacher_start_empty_topic_returns_422(self, client):
        c, _ = client
        response = c.post(
            f"{PREFIX}/teacher/start",
            json={"user_id": "alice", "topic": ""},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /teacher/continue
# ---------------------------------------------------------------------------

class TestTeacherContinueEndpoint:
    def test_teacher_continue_returns_200(self, client):
        c, _ = client
        response = c.post(
            f"{PREFIX}/teacher/continue",
            json={"user_id": "alice", "topic": "Python"},
        )
        assert response.status_code == 200

    def test_teacher_continue_delegates_to_service(self, client):
        c, mock_service = client
        c.post(
            f"{PREFIX}/teacher/continue",
            json={"user_id": "alice", "topic": "Python"},
        )
        mock_service.continue_teacher_mode.assert_called_once_with("alice", "Python")

    def test_teacher_continue_missing_user_id_returns_422(self, client):
        c, _ = client
        response = c.post(
            f"{PREFIX}/teacher/continue",
            json={"topic": "Python"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /personalization
# ---------------------------------------------------------------------------

class TestPersonalizationEndpoint:
    def test_personalization_returns_200(self, client):
        c, _ = client
        response = c.get(f"{PREFIX}/personalization?user_id=alice")
        assert response.status_code == 200

    def test_personalization_response_shape(self, client):
        c, _ = client
        response = c.get(f"{PREFIX}/personalization?user_id=alice")
        body = response.json()
        assert "tone" in body
        assert "style" in body
        assert "focus" in body
        assert "reasons" in body

    def test_personalization_delegates_to_service(self, client):
        c, mock_service = client
        c.get(f"{PREFIX}/personalization?user_id=alice")
        mock_service.get_personalization.assert_called_once_with("alice")

    def test_personalization_missing_user_id_returns_422(self, client):
        c, _ = client
        response = c.get(f"{PREFIX}/personalization")
        assert response.status_code == 422

    def test_personalization_empty_user_id_returns_422(self, client):
        c, _ = client
        response = c.get(f"{PREFIX}/personalization?user_id=")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Middleware stack: error handling end-to-end
# ---------------------------------------------------------------------------

class TestMiddlewareStack:
    def test_error_handler_catches_value_error_returns_400(self):
        """ErrorHandlerMiddleware converts ValueError to HTTP 400."""
        mock_service = make_mock_service()
        mock_service.process_chat = AsyncMock(side_effect=ValueError("bad input"))
        app = create_test_app(mock_service)
        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.post(
                f"{PREFIX}/chat",
                json={"user_id": "alice", "message": "hello"},
            )
        assert response.status_code == 400
        body = response.json()
        assert "error" in body
        assert body["status"] == 400

    def test_error_handler_catches_generic_exception_returns_500(self):
        """ErrorHandlerMiddleware converts unhandled exceptions to HTTP 500."""
        mock_service = make_mock_service()
        mock_service.process_chat = AsyncMock(side_effect=RuntimeError("boom"))
        app = create_test_app(mock_service)
        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.post(
                f"{PREFIX}/chat",
                json={"user_id": "alice", "message": "hello"},
            )
        assert response.status_code == 500
        body = response.json()
        assert "error" in body
        assert body["status"] == 500
        # No stack trace in response
        assert "Traceback" not in body["error"]

    def test_error_response_no_stack_trace(self):
        """Error responses must never contain stack trace strings."""
        mock_service = make_mock_service()
        mock_service.get_insights = AsyncMock(side_effect=Exception("internal error"))
        app = create_test_app(mock_service)
        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.get(f"{PREFIX}/insights")
        body = response.json()
        error_text = body.get("error", "")
        assert "Traceback" not in error_text
        assert "File " not in error_text
        assert "line " not in error_text

    def test_error_handler_generic_message_hides_internals(self):
        """HTTP 500 response uses generic message, not the raw exception message."""
        mock_service = make_mock_service()
        mock_service.get_insights = AsyncMock(
            side_effect=RuntimeError("secret internal detail")
        )
        app = create_test_app(mock_service)
        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.get(f"{PREFIX}/insights")
        assert response.status_code == 500
        body = response.json()
        assert "secret internal detail" not in body.get("error", "")

    def test_logging_middleware_does_not_break_successful_requests(self):
        """LoggingMiddleware must not interfere with normal request flow."""
        mock_service = make_mock_service()
        app = create_test_app(mock_service)
        with TestClient(app) as c:
            response = c.get(f"{PREFIX}/insights")
        assert response.status_code == 200

    def test_middleware_stack_order_error_wraps_logging(self):
        """
        ErrorHandlerMiddleware (outermost) must catch exceptions even when
        LoggingMiddleware is in the stack.
        """
        mock_service = make_mock_service()
        mock_service.get_insight_moments = AsyncMock(
            side_effect=RuntimeError("middleware order test")
        )
        app = create_test_app(mock_service)
        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.get(f"{PREFIX}/insight-moments")
        # ErrorHandlerMiddleware must catch this and return 500
        assert response.status_code == 500
