"""
Unit tests for ErrorHandlerMiddleware.

Tests that the middleware correctly maps exceptions to HTTP responses:
- ValueError → HTTP 400 with exception message
- ValidationError → HTTP 422 with descriptive message
- Generic Exception → HTTP 500 with generic message (no internal details)
- All caught exceptions are logged before returning
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, field_validator, ValidationError

from luma.api.middleware.error_handler import ErrorHandlerMiddleware


def make_app(exc_factory):
    """Create a minimal FastAPI app that raises the given exception on GET /test."""
    app = FastAPI()
    app.add_middleware(ErrorHandlerMiddleware)

    @app.get("/test")
    async def test_route():
        raise exc_factory()

    return app


# ---------------------------------------------------------------------------
# ValueError → HTTP 400
# ---------------------------------------------------------------------------

class TestValueErrorMapping:
    def test_value_error_returns_400(self):
        app = make_app(lambda: ValueError("bad input value"))
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test")
        assert response.status_code == 400

    def test_value_error_body_contains_exception_message(self):
        app = make_app(lambda: ValueError("bad input value"))
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test")
        body = response.json()
        assert body["error"] == "bad input value"
        assert body["status"] == 400

    def test_value_error_is_logged(self):
        app = make_app(lambda: ValueError("logged value error"))
        client = TestClient(app, raise_server_exceptions=False)
        with patch("luma.api.middleware.error_handler.logger") as mock_logger:
            client.get("/test")
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "ValueError" in call_args


# ---------------------------------------------------------------------------
# ValidationError → HTTP 422
# ---------------------------------------------------------------------------

class SampleModel(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be empty")
        return v


def make_validation_error():
    try:
        SampleModel(name="")
    except ValidationError as exc:
        return exc
    raise AssertionError("Expected ValidationError was not raised")


class TestValidationErrorMapping:
    def test_validation_error_returns_422(self):
        app = make_app(make_validation_error)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test")
        assert response.status_code == 422

    def test_validation_error_body_has_error_and_status(self):
        app = make_app(make_validation_error)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test")
        body = response.json()
        assert "error" in body
        assert body["status"] == 422
        # The error message should be descriptive (from pydantic)
        assert len(body["error"]) > 0

    def test_validation_error_is_logged(self):
        app = make_app(make_validation_error)
        client = TestClient(app, raise_server_exceptions=False)
        with patch("luma.api.middleware.error_handler.logger") as mock_logger:
            client.get("/test")
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "ValidationError" in call_args


# ---------------------------------------------------------------------------
# Generic Exception → HTTP 500
# ---------------------------------------------------------------------------

class TestGenericExceptionMapping:
    def test_generic_exception_returns_500(self):
        app = make_app(lambda: RuntimeError("something went wrong internally"))
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test")
        assert response.status_code == 500

    def test_generic_exception_body_has_generic_message(self):
        app = make_app(lambda: RuntimeError("something went wrong internally"))
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test")
        body = response.json()
        assert body["status"] == 500
        # Must NOT expose internal details
        assert "something went wrong internally" not in body["error"]
        assert body["error"] == "An internal server error occurred."

    def test_generic_exception_body_has_no_traceback(self):
        app = make_app(lambda: RuntimeError("internal detail"))
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test")
        body = response.json()
        assert "Traceback" not in body["error"]
        assert "File " not in body["error"]
        assert "line " not in body["error"]

    def test_generic_exception_is_logged(self):
        app = make_app(lambda: RuntimeError("logged runtime error"))
        client = TestClient(app, raise_server_exceptions=False)
        with patch("luma.api.middleware.error_handler.logger") as mock_logger:
            client.get("/test")
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            assert "RuntimeError" in call_args

    def test_key_error_returns_500(self):
        app = make_app(lambda: KeyError("missing_key"))
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test")
        assert response.status_code == 500

    def test_type_error_returns_500(self):
        app = make_app(lambda: TypeError("type mismatch"))
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test")
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# Logging happens before response is returned
# ---------------------------------------------------------------------------

class TestLoggingBeforeResponse:
    def test_value_error_logged_before_response(self):
        """Verify logger.warning is called (i.e., before the response is sent)."""
        log_calls = []

        app = FastAPI()
        app.add_middleware(ErrorHandlerMiddleware)

        @app.get("/test")
        async def test_route():
            raise ValueError("test error")

        client = TestClient(app, raise_server_exceptions=False)

        with patch("luma.api.middleware.error_handler.logger") as mock_logger:
            mock_logger.warning.side_effect = lambda msg: log_calls.append(msg)
            response = client.get("/test")

        assert response.status_code == 400
        assert len(log_calls) == 1
        assert "ValueError" in log_calls[0]

    def test_generic_exception_logged_before_response(self):
        log_calls = []

        app = FastAPI()
        app.add_middleware(ErrorHandlerMiddleware)

        @app.get("/test")
        async def test_route():
            raise Exception("generic error")

        client = TestClient(app, raise_server_exceptions=False)

        with patch("luma.api.middleware.error_handler.logger") as mock_logger:
            mock_logger.error.side_effect = lambda msg: log_calls.append(msg)
            response = client.get("/test")

        assert response.status_code == 500
        assert len(log_calls) == 1
