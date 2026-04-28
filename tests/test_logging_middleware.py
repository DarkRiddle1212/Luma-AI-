"""
Unit tests for LoggingMiddleware.

Tests that the middleware correctly logs:
- HTTP method and path on every request
- Status code and response time on every response
- Exception type and message on errors (not the full traceback)
"""

import pytest
from unittest.mock import patch, call
from fastapi import FastAPI
from fastapi.testclient import TestClient

from luma.api.middleware.logging import LoggingMiddleware


def make_app(route_handler=None, raise_exc=None):
    """Create a minimal FastAPI app with LoggingMiddleware."""
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    if route_handler is not None:
        app.get("/test")(route_handler)
    else:
        @app.get("/test")
        async def default_route():
            if raise_exc is not None:
                raise raise_exc
            return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# Request logging: method and path
# ---------------------------------------------------------------------------

class TestRequestLogging:
    def test_logs_method_and_path_on_request(self):
        app = make_app()
        client = TestClient(app, raise_server_exceptions=False)

        with patch("luma.api.middleware.logging.logger") as mock_logger:
            client.get("/test")
            # First info call should be the request log
            first_call = mock_logger.info.call_args_list[0]
            log_msg = first_call[0][0]
            assert "GET" in log_msg
            assert "/test" in log_msg

    def test_logs_post_method_and_path(self):
        app = FastAPI()
        app.add_middleware(LoggingMiddleware)

        @app.post("/submit")
        async def submit():
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)

        with patch("luma.api.middleware.logging.logger") as mock_logger:
            client.post("/submit")
            first_call = mock_logger.info.call_args_list[0]
            log_msg = first_call[0][0]
            assert "POST" in log_msg
            assert "/submit" in log_msg

    def test_logs_different_paths(self):
        app = FastAPI()
        app.add_middleware(LoggingMiddleware)

        @app.get("/api/v1/resource")
        async def resource():
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)

        with patch("luma.api.middleware.logging.logger") as mock_logger:
            client.get("/api/v1/resource")
            first_call = mock_logger.info.call_args_list[0]
            log_msg = first_call[0][0]
            assert "/api/v1/resource" in log_msg


# ---------------------------------------------------------------------------
# Response logging: status code and response time
# ---------------------------------------------------------------------------

class TestResponseLogging:
    def test_logs_status_code_on_response(self):
        app = make_app()
        client = TestClient(app, raise_server_exceptions=False)

        with patch("luma.api.middleware.logging.logger") as mock_logger:
            client.get("/test")
            # Second info call should be the response log
            second_call = mock_logger.info.call_args_list[1]
            log_msg = second_call[0][0]
            assert "200" in log_msg

    def test_logs_response_time_in_ms(self):
        app = make_app()
        client = TestClient(app, raise_server_exceptions=False)

        with patch("luma.api.middleware.logging.logger") as mock_logger:
            client.get("/test")
            second_call = mock_logger.info.call_args_list[1]
            log_msg = second_call[0][0]
            # Should contain "ms" indicating milliseconds
            assert "ms" in log_msg

    def test_logs_method_and_path_in_response(self):
        app = make_app()
        client = TestClient(app, raise_server_exceptions=False)

        with patch("luma.api.middleware.logging.logger") as mock_logger:
            client.get("/test")
            second_call = mock_logger.info.call_args_list[1]
            log_msg = second_call[0][0]
            assert "GET" in log_msg
            assert "/test" in log_msg

    def test_logs_404_status_code(self):
        app = FastAPI()
        app.add_middleware(LoggingMiddleware)

        client = TestClient(app, raise_server_exceptions=False)

        with patch("luma.api.middleware.logging.logger") as mock_logger:
            client.get("/nonexistent")
            # Should have at least 2 info calls (request + response)
            assert mock_logger.info.call_count >= 2
            second_call = mock_logger.info.call_args_list[1]
            log_msg = second_call[0][0]
            assert "404" in log_msg

    def test_two_info_calls_per_successful_request(self):
        app = make_app()
        client = TestClient(app, raise_server_exceptions=False)

        with patch("luma.api.middleware.logging.logger") as mock_logger:
            client.get("/test")
            assert mock_logger.info.call_count == 2


# ---------------------------------------------------------------------------
# Error logging: exception type and message (not full traceback)
# ---------------------------------------------------------------------------

class TestErrorLogging:
    def test_logs_exception_type_on_error(self):
        app = FastAPI()
        app.add_middleware(LoggingMiddleware)

        @app.get("/error")
        async def error_route():
            raise RuntimeError("something failed")

        client = TestClient(app, raise_server_exceptions=False)

        with patch("luma.api.middleware.logging.logger") as mock_logger:
            client.get("/error")
            mock_logger.error.assert_called_once()
            log_msg = mock_logger.error.call_args[0][0]
            assert "RuntimeError" in log_msg

    def test_logs_exception_message_on_error(self):
        app = FastAPI()
        app.add_middleware(LoggingMiddleware)

        @app.get("/error")
        async def error_route():
            raise RuntimeError("something failed")

        client = TestClient(app, raise_server_exceptions=False)

        with patch("luma.api.middleware.logging.logger") as mock_logger:
            client.get("/error")
            log_msg = mock_logger.error.call_args[0][0]
            assert "something failed" in log_msg

    def test_does_not_log_traceback_on_error(self):
        """The error log message should not contain traceback markers."""
        app = FastAPI()
        app.add_middleware(LoggingMiddleware)

        @app.get("/error")
        async def error_route():
            raise RuntimeError("internal error")

        client = TestClient(app, raise_server_exceptions=False)

        with patch("luma.api.middleware.logging.logger") as mock_logger:
            client.get("/error")
            log_msg = mock_logger.error.call_args[0][0]
            assert "Traceback" not in log_msg
            assert "File " not in log_msg

    def test_does_not_log_request_body(self):
        """LoggingMiddleware must not log request body content."""
        app = FastAPI()
        app.add_middleware(LoggingMiddleware)

        @app.post("/data")
        async def data_route():
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)
        secret_body = "super_secret_password_12345"

        with patch("luma.api.middleware.logging.logger") as mock_logger:
            client.post("/data", json={"password": secret_body})
            all_log_messages = " ".join(
                str(c) for c in mock_logger.info.call_args_list
            )
            assert secret_body not in all_log_messages
