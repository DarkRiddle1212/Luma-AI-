"""
Unit tests for InsightController.

Tests verify:
- GET /insights without namespace → calls service.get_insights() with no namespace, returns HTTP 200
- GET /insights?namespace=foo → calls service.get_insights(namespace="foo"), returns HTTP 200
- GET /insight-moments → calls service.get_insight_moments(), returns HTTP 200
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from luma.api.controllers.insight_controller import router
from luma.api.controllers.chat_controller import get_luma_service


def make_mock_service():
    """Create a mock LumaService with async methods."""
    service = MagicMock()
    service.get_insights = AsyncMock(return_value=["insight_a", "insight_b"])
    service.get_insight_moments = AsyncMock(return_value=["moment_1"])
    return service


def make_app(mock_service=None):
    """Create a minimal FastAPI app with the insight router and a mock service."""
    app = FastAPI()
    app.include_router(router)
    if mock_service is not None:
        app.dependency_overrides[get_luma_service] = lambda: mock_service
    return app


# ---------------------------------------------------------------------------
# GET /insights
# ---------------------------------------------------------------------------

class TestGetInsights:
    def test_without_namespace_returns_200(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.get("/insights")
        assert response.status_code == 200

    def test_without_namespace_calls_get_insights_no_namespace(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        client.get("/insights")
        mock_service.get_insights.assert_called_once_with(namespace=None)

    def test_without_namespace_returns_insights_list(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.get("/insights")
        body = response.json()
        assert "insights" in body
        assert body["insights"] == ["insight_a", "insight_b"]

    def test_with_namespace_returns_200(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.get("/insights?namespace=foo")
        assert response.status_code == 200

    def test_with_namespace_forwards_namespace(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        client.get("/insights?namespace=foo")
        mock_service.get_insights.assert_called_once_with(namespace="foo")

    def test_with_namespace_returns_insights_list(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.get("/insights?namespace=foo")
        body = response.json()
        assert "insights" in body
        assert isinstance(body["insights"], list)


# ---------------------------------------------------------------------------
# GET /insight-moments
# ---------------------------------------------------------------------------

class TestGetInsightMoments:
    def test_returns_200(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.get("/insight-moments")
        assert response.status_code == 200

    def test_calls_get_insight_moments(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        client.get("/insight-moments")
        mock_service.get_insight_moments.assert_called_once()

    def test_returns_insight_moments_list(self):
        mock_service = make_mock_service()
        app = make_app(mock_service)
        client = TestClient(app)
        response = client.get("/insight-moments")
        body = response.json()
        assert "insight_moments" in body
        assert body["insight_moments"] == ["moment_1"]
