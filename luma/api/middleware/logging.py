"""
Logging middleware for the Luma API layer.

Logs every incoming request and outgoing response with method, path,
status code, and response time. Logs exception type and message on errors
without exposing stack traces or request/response bodies.
"""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from luma.utils.logger import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        logger.info(f"Request: {request.method} {request.url.path}")
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}")
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Response: {response.status_code} "
            f"({request.method} {request.url.path}) "
            f"in {duration_ms:.1f}ms"
        )
        return response
