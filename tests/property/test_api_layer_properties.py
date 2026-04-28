"""
Property-based tests for the Luma API Layer.

Properties 1, 2, 5, and 6 cover schema validation behaviour.
Each test is tagged: Feature: luma-api-layer, Property N: description
Hypothesis configured with max_examples=100 per test.
"""

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from pydantic import ValidationError

from luma.api.schemas import (
    ChatRequest,
    TeacherRequest,
    InsightResponse,
    InsightMomentsResponse,
    TeacherResponse,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def whitespace_only_strategy():
    """Generate strings composed entirely of whitespace characters, including empty string."""
    return st.one_of(
        st.just(""),
        st.text(
            alphabet=st.sampled_from([" ", "\t", "\n", "\r", "\x0b", "\x0c"]),
            min_size=1,
            max_size=20,
        ),
    )


def non_empty_string_strategy():
    """Generate non-empty, non-whitespace strings suitable for valid fields."""
    return st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Po")),
        min_size=1,
        max_size=50,
    ).filter(lambda s: s.strip() != "")


def any_list_strategy():
    """Generate arbitrary lists including empty ones."""
    return st.lists(
        st.one_of(st.integers(), st.text(), st.booleans(), st.none()),
        min_size=0,
        max_size=20,
    )


# ---------------------------------------------------------------------------
# Property 1: ChatRequest rejects all whitespace-only user_id and message values
# ---------------------------------------------------------------------------

# Feature: luma-api-layer, Property 1: ChatRequest rejects all whitespace-only user_id and message values
@given(bad_value=whitespace_only_strategy(), good_value=non_empty_string_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_chat_request_rejects_whitespace_user_id(bad_value, good_value):
    """
    **Validates: Requirements 1.5, 1.6, 6.10**

    For any string composed entirely of whitespace characters (including ""),
    constructing a ChatRequest with that string as user_id SHALL raise a
    Pydantic ValidationError.
    """
    with pytest.raises(ValidationError):
        ChatRequest(user_id=bad_value, message=good_value)


# Feature: luma-api-layer, Property 1: ChatRequest rejects all whitespace-only user_id and message values
@given(bad_value=whitespace_only_strategy(), good_value=non_empty_string_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_chat_request_rejects_whitespace_message(bad_value, good_value):
    """
    **Validates: Requirements 1.5, 1.6, 6.10**

    For any string composed entirely of whitespace characters (including ""),
    constructing a ChatRequest with that string as message SHALL raise a
    Pydantic ValidationError.
    """
    with pytest.raises(ValidationError):
        ChatRequest(user_id=good_value, message=bad_value)


# ---------------------------------------------------------------------------
# Property 2: TeacherRequest rejects all whitespace-only user_id and topic values
# ---------------------------------------------------------------------------

# Feature: luma-api-layer, Property 2: TeacherRequest rejects all whitespace-only user_id and topic values
@given(bad_value=whitespace_only_strategy(), good_value=non_empty_string_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_teacher_request_rejects_whitespace_user_id(bad_value, good_value):
    """
    **Validates: Requirements 4.7, 6.11**

    For any string composed entirely of whitespace characters (including ""),
    constructing a TeacherRequest with that string as user_id SHALL raise a
    Pydantic ValidationError.
    """
    with pytest.raises(ValidationError):
        TeacherRequest(user_id=bad_value, topic=good_value)


# Feature: luma-api-layer, Property 2: TeacherRequest rejects all whitespace-only user_id and topic values
@given(bad_value=whitespace_only_strategy(), good_value=non_empty_string_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_teacher_request_rejects_whitespace_topic(bad_value, good_value):
    """
    **Validates: Requirements 4.7, 6.11**

    For any string composed entirely of whitespace characters (including ""),
    constructing a TeacherRequest with that string as topic SHALL raise a
    Pydantic ValidationError.
    """
    with pytest.raises(ValidationError):
        TeacherRequest(user_id=good_value, topic=bad_value)


# ---------------------------------------------------------------------------
# Property 5: InsightResponse and InsightMomentsResponse always contain a list field
# ---------------------------------------------------------------------------

# Feature: luma-api-layer, Property 5: InsightResponse and InsightMomentsResponse always contain a list field
@given(items=any_list_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_insight_response_insights_is_always_list(items):
    """
    **Validates: Requirements 2.5, 3.3**

    For any list of insights (including an empty list), constructing an
    InsightResponse SHALL produce an object with an insights field that is a list.
    """
    obj = InsightResponse(insights=items)
    assert isinstance(obj.insights, list), (
        f"Expected insights to be a list, got {type(obj.insights)}"
    )


# Feature: luma-api-layer, Property 5: InsightResponse and InsightMomentsResponse always contain a list field
@given(items=any_list_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_insight_moments_response_insight_moments_is_always_list(items):
    """
    **Validates: Requirements 2.5, 3.3**

    For any list of moments (including an empty list), constructing an
    InsightMomentsResponse SHALL produce an object with an insight_moments field
    that is a list.
    """
    obj = InsightMomentsResponse(insight_moments=items)
    assert isinstance(obj.insight_moments, list), (
        f"Expected insight_moments to be a list, got {type(obj.insight_moments)}"
    )


# ---------------------------------------------------------------------------
# Property 6: TeacherResponse always contains all required fields
# ---------------------------------------------------------------------------

def teacher_response_strategy():
    """Generate TeacherResponse-compatible field combinations."""
    return st.fixed_dictionaries({
        "session_id": st.text(min_size=1, max_size=50).filter(lambda s: s.strip() != ""),
        "status": st.sampled_from(["active", "completed", "paused", "started"]),
        "lessons": any_list_strategy(),
        "explanations": any_list_strategy(),
        "exercises": any_list_strategy(),
    })


# Feature: luma-api-layer, Property 6: TeacherResponse always contains all required fields
@given(data=teacher_response_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_teacher_response_always_has_all_required_fields(data):
    """
    **Validates: Requirements 4.5**

    For any TeachingSession-like object (with any combination of session_id,
    status, lessons, explanations, exercises), constructing a TeacherResponse
    SHALL produce an object with all five fields present and with the correct
    types: session_id (str), status (str), lessons (list), explanations (list),
    exercises (list).
    """
    obj = TeacherResponse(**data)

    assert isinstance(obj.session_id, str), (
        f"Expected session_id to be str, got {type(obj.session_id)}"
    )
    assert isinstance(obj.status, str), (
        f"Expected status to be str, got {type(obj.status)}"
    )
    assert isinstance(obj.lessons, list), (
        f"Expected lessons to be list, got {type(obj.lessons)}"
    )
    assert isinstance(obj.explanations, list), (
        f"Expected explanations to be list, got {type(obj.explanations)}"
    )
    assert isinstance(obj.exercises, list), (
        f"Expected exercises to be list, got {type(obj.exercises)}"
    )


# ---------------------------------------------------------------------------
# Property 4: Namespace forwarding is transparent
# ---------------------------------------------------------------------------

# Feature: luma-api-layer, Property 4: Namespace forwarding is transparent
@given(namespace=st.text(min_size=1, max_size=50).filter(lambda s: s.strip() != ""))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_namespace_forwarding_is_transparent(namespace):
    """
    **Validates: Requirements 2.3, 7.6**

    For any non-empty namespace string, calling LumaService.get_insights(namespace=ns)
    SHALL result in InsightEngine.generate_insights() being called with exactly
    namespace=ns — the value is forwarded unchanged.
    """
    import asyncio
    from unittest.mock import MagicMock
    from luma.api.services.luma_service import LumaService

    insight_engine = MagicMock()
    insight_report = MagicMock()
    insight_report.insights = []
    insight_engine.generate_insights.return_value = insight_report

    service = LumaService(
        memory_interface=MagicMock(),
        insight_engine=insight_engine,
        insight_moments_engine=MagicMock(),
        personalization_engine=MagicMock(),
        teacher_mode=MagicMock(),
    )

    asyncio.run(service.get_insights(namespace=namespace))

    insight_engine.generate_insights.assert_called_once_with(namespace=namespace)


# ---------------------------------------------------------------------------
# Property 3: ChatResponse always contains required structural fields
# ---------------------------------------------------------------------------

# Feature: luma-api-layer, Property 3: ChatResponse always contains required structural fields
@given(
    user_id=st.text(min_size=1, max_size=50).filter(lambda s: s.strip() != ""),
    message=st.text(min_size=1, max_size=200).filter(lambda s: s.strip() != ""),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_chat_response_always_has_required_structural_fields(user_id, message):
    """
    **Validates: Requirements 1.3, 5.3**

    For any valid (user_id, message) pair processed by LumaService.process_chat()
    (with mocked core dependencies), the returned dict SHALL always contain:
    - a non-empty ``response`` string
    - an ``insight_moments`` key whose value is a list
    - a ``personalization`` key whose value is a dict containing at minimum
      the keys ``tone``, ``style``, and ``focus``
    """
    import asyncio
    from unittest.mock import MagicMock
    from luma.api.services.luma_service import LumaService

    memory = MagicMock()
    memory.retrieve.return_value = {
        "memories": [],
        "total_count": 0,
        "query_metadata": {},
    }
    memory.store.return_value = "mem_id"

    adaptation_ctx = MagicMock()
    adaptation_ctx.tone = "casual"
    adaptation_ctx.style = "concise"
    adaptation_ctx.focus = "high-level"
    adaptation_ctx.reasons = {}
    adaptation_ctx.model_dump.return_value = {
        "tone": "casual",
        "style": "concise",
        "focus": "high-level",
        "reasons": {},
    }

    personalization_result = MagicMock()
    personalization_result.adaptation = adaptation_ctx

    personalization_engine = MagicMock()
    personalization_engine.personalize.return_value = personalization_result

    insight_moments_engine = MagicMock()
    insight_moments_engine.generate_moments.return_value = []

    service = LumaService(
        memory_interface=memory,
        insight_engine=MagicMock(),
        insight_moments_engine=insight_moments_engine,
        personalization_engine=personalization_engine,
        teacher_mode=MagicMock(),
    )

    result = asyncio.run(
        service.process_chat(user_id, message)
    )

    # response must be a non-empty string
    assert isinstance(result["response"], str), (
        f"Expected response to be str, got {type(result['response'])}"
    )
    assert len(result["response"]) > 0, "Expected response to be non-empty"

    # insight_moments must be a list
    assert isinstance(result["insight_moments"], list), (
        f"Expected insight_moments to be list, got {type(result['insight_moments'])}"
    )

    # personalization must be a dict with tone, style, focus
    p = result["personalization"]
    assert isinstance(p, dict), (
        f"Expected personalization to be dict, got {type(p)}"
    )
    assert "tone" in p, "personalization dict must contain 'tone'"
    assert "style" in p, "personalization dict must contain 'style'"
    assert "focus" in p, "personalization dict must contain 'focus'"


# ---------------------------------------------------------------------------
# Property 7: ErrorHandlerMiddleware never exposes stack traces
# ---------------------------------------------------------------------------

# Feature: luma-api-layer, Property 7: ErrorHandlerMiddleware never exposes stack traces
@given(
    exc_message=st.text(min_size=0, max_size=200),
    exc_type=st.sampled_from([
        RuntimeError, KeyError, AttributeError, IndexError,
        OSError, NotImplementedError, MemoryError, OverflowError,
    ]),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_error_handler_never_exposes_stack_traces(exc_message, exc_type):
    """
    **Validates: Requirements 9.5**

    For any exception type and message, when ErrorHandlerMiddleware catches the
    exception, the error field of the returned ErrorResponse SHALL NOT contain
    any of the strings "Traceback", "File ", or "line ".
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from luma.api.middleware.error_handler import ErrorHandlerMiddleware

    app = FastAPI()
    app.add_middleware(ErrorHandlerMiddleware)

    @app.get("/test")
    async def test_route():
        raise exc_type(exc_message)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/test")

    body = response.json()
    error_field = body.get("error", "")

    assert "Traceback" not in error_field, (
        f"error field contains 'Traceback': {error_field!r}"
    )
    assert "File " not in error_field, (
        f"error field contains 'File ': {error_field!r}"
    )
    assert "line " not in error_field, (
        f"error field contains 'line ': {error_field!r}"
    )


# ---------------------------------------------------------------------------
# Property 8: ErrorHandlerMiddleware maps all unhandled exceptions to HTTP 500
# ---------------------------------------------------------------------------

# Feature: luma-api-layer, Property 8: ErrorHandlerMiddleware maps all unhandled exceptions to HTTP 500
@given(
    exc_message=st.text(min_size=0, max_size=200),
    exc_type=st.sampled_from([
        RuntimeError, KeyError, AttributeError, IndexError,
        OSError, NotImplementedError, MemoryError, OverflowError,
        StopIteration, ArithmeticError, LookupError,
    ]),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_error_handler_maps_unhandled_exceptions_to_500(exc_message, exc_type):
    """
    **Validates: Requirements 9.1**

    For any exception that is not a ValueError or Pydantic ValidationError,
    when ErrorHandlerMiddleware catches it, the HTTP response status code SHALL
    be 500 and the body SHALL be a valid ErrorResponse JSON object.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from luma.api.middleware.error_handler import ErrorHandlerMiddleware

    app = FastAPI()
    app.add_middleware(ErrorHandlerMiddleware)

    @app.get("/test")
    async def test_route():
        raise exc_type(exc_message)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/test")

    assert response.status_code == 500, (
        f"Expected 500 for {exc_type.__name__}, got {response.status_code}"
    )

    body = response.json()
    assert "error" in body, f"Response body missing 'error' field: {body}"
    assert "status" in body, f"Response body missing 'status' field: {body}"
    assert isinstance(body["error"], str), (
        f"'error' field must be a string, got {type(body['error'])}"
    )
    assert body["status"] == 500, (
        f"'status' field must be 500, got {body['status']}"
    )


# ---------------------------------------------------------------------------
# Property 9: LoggingMiddleware logs method, path, and response time for every request
# ---------------------------------------------------------------------------

# Feature: luma-api-layer, Property 9: LoggingMiddleware logs method, path, and response time for every request
@given(
    path_segment=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu")),
        min_size=3,
        max_size=20,
    ).filter(lambda s: s.isalpha() and len(s) >= 3),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_logging_middleware_logs_method_path_and_response_time(path_segment):
    """
    **Validates: Requirements 8.1, 8.2, 8.5**

    For any HTTP method and URL path combination, after LoggingMiddleware
    processes the request, the log output SHALL contain the HTTP method string,
    the URL path string, and a numeric response time value in milliseconds —
    and SHALL NOT contain any substring of the request or response body.
    """
    import re
    from unittest.mock import patch
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from luma.api.middleware.logging import LoggingMiddleware

    # Use a body value that is clearly distinct from the path and URL
    # (a long sentinel string that cannot appear in method/path/status logs)
    body_sentinel = "SECRETBODYVALUE_XYZ_987654321"
    route_path = f"/{path_segment}"

    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get(route_path)
    async def test_route():
        # Return body_sentinel in response — must NOT appear in logs
        return {"data": body_sentinel}

    client = TestClient(app, raise_server_exceptions=False)

    log_messages = []

    with patch("luma.api.middleware.logging.logger") as mock_logger:
        mock_logger.info.side_effect = lambda msg: log_messages.append(msg)
        client.get(route_path)

    all_logs = " ".join(log_messages)

    # Must contain the HTTP method
    assert "GET" in all_logs, (
        f"Log output missing 'GET': {all_logs!r}"
    )

    # Must contain the URL path segment
    assert path_segment in all_logs, (
        f"Log output missing path segment '{path_segment}': {all_logs!r}"
    )

    # Must contain a numeric response time (digits followed by ms)
    assert re.search(r"\d+\.\d+ms", all_logs) or re.search(r"\d+ms", all_logs), (
        f"Log output missing numeric response time: {all_logs!r}"
    )

    # Must NOT contain the response body sentinel value
    assert body_sentinel not in all_logs, (
        f"Log output contains response body content '{body_sentinel}': {all_logs!r}"
    )
