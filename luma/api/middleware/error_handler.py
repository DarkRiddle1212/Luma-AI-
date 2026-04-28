"""
Error handler middleware for the Luma API layer.

Catches exceptions from route handlers and maps them to structured
ErrorResponse JSON with appropriate HTTP status codes. Never exposes
stack traces or internal implementation details in the response body.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from pydantic import ValidationError
from luma.api.schemas import ErrorResponse
from luma.utils.logger import get_logger

logger = get_logger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except ValidationError as exc:
            logger.warning(f"ValidationError: {exc}")
            return JSONResponse(
                status_code=422,
                content=ErrorResponse(
                    error=str(exc),
                    status=422,
                ).model_dump(),
            )
        except ValueError as exc:
            logger.warning(f"ValueError: {exc}")
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(error=str(exc), status=400).model_dump(),
            )
        except Exception as exc:
            logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="An internal server error occurred.",
                    status=500,
                ).model_dump(),
            )
