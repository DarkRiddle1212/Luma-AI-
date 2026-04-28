"""
Unit tests for luma/api/schemas.py.

Covers ChatRequest, TeacherRequest, and all response schemas.
"""

import pytest
from pydantic import ValidationError

from luma.api.schemas import (
    ChatRequest,
    ChatResponse,
    InsightResponse,
    InsightMomentsResponse,
    TeacherRequest,
    TeacherResponse,
    PersonalizationResponse,
    ErrorResponse,
)


# ---------------------------------------------------------------------------
# ChatRequest
# ---------------------------------------------------------------------------

class TestChatRequest:
    def test_valid_chat_request(self):
        req = ChatRequest(user_id="alice", message="Hello!")
        assert req.user_id == "alice"
        assert req.message == "Hello!"

    def test_missing_user_id_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="Hello!")

    def test_missing_message_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(user_id="alice")

    def test_empty_user_id_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(user_id="", message="Hello!")

    def test_whitespace_user_id_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(user_id="   ", message="Hello!")

    def test_tab_whitespace_user_id_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(user_id="\t\n", message="Hello!")

    def test_empty_message_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(user_id="alice", message="")

    def test_whitespace_message_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(user_id="alice", message="   ")

    def test_both_empty_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(user_id="", message="")

    def test_message_with_leading_trailing_spaces_is_valid(self):
        # Non-whitespace-only strings with surrounding spaces are valid
        req = ChatRequest(user_id="alice", message="  hello  ")
        assert req.message == "  hello  "


# ---------------------------------------------------------------------------
# TeacherRequest
# ---------------------------------------------------------------------------

class TestTeacherRequest:
    def test_valid_teacher_request(self):
        req = TeacherRequest(user_id="bob", topic="Python basics")
        assert req.user_id == "bob"
        assert req.topic == "Python basics"

    def test_missing_user_id_raises(self):
        with pytest.raises(ValidationError):
            TeacherRequest(topic="Python basics")

    def test_missing_topic_raises(self):
        with pytest.raises(ValidationError):
            TeacherRequest(user_id="bob")

    def test_empty_user_id_raises(self):
        with pytest.raises(ValidationError):
            TeacherRequest(user_id="", topic="Python basics")

    def test_whitespace_user_id_raises(self):
        with pytest.raises(ValidationError):
            TeacherRequest(user_id="  ", topic="Python basics")

    def test_empty_topic_raises(self):
        with pytest.raises(ValidationError):
            TeacherRequest(user_id="bob", topic="")

    def test_whitespace_topic_raises(self):
        with pytest.raises(ValidationError):
            TeacherRequest(user_id="bob", topic="\n\t")

    def test_both_empty_raises(self):
        with pytest.raises(ValidationError):
            TeacherRequest(user_id="", topic="")


# ---------------------------------------------------------------------------
# Response schemas — valid inputs
# ---------------------------------------------------------------------------

class TestChatResponse:
    def test_valid_chat_response(self):
        resp = ChatResponse(
            response="Here is your answer.",
            insight_moments=[{"moment": "key insight"}],
            personalization={"tone": "casual", "style": "concise", "focus": "high-level"},
        )
        assert resp.response == "Here is your answer."
        assert isinstance(resp.insight_moments, list)
        assert isinstance(resp.personalization, dict)

    def test_empty_lists_and_dicts_are_valid(self):
        resp = ChatResponse(response="ok", insight_moments=[], personalization={})
        assert resp.insight_moments == []
        assert resp.personalization == {}


class TestInsightResponse:
    def test_valid_insight_response(self):
        resp = InsightResponse(insights=["insight1", "insight2"])
        assert resp.insights == ["insight1", "insight2"]

    def test_empty_insights_list(self):
        resp = InsightResponse(insights=[])
        assert resp.insights == []

    def test_insights_with_dicts(self):
        resp = InsightResponse(insights=[{"text": "foo", "confidence": 0.9}])
        assert len(resp.insights) == 1


class TestInsightMomentsResponse:
    def test_valid_insight_moments_response(self):
        resp = InsightMomentsResponse(insight_moments=["moment1"])
        assert resp.insight_moments == ["moment1"]

    def test_empty_insight_moments(self):
        resp = InsightMomentsResponse(insight_moments=[])
        assert resp.insight_moments == []


class TestTeacherResponse:
    def test_valid_teacher_response(self):
        resp = TeacherResponse(
            session_id="sess-123",
            status="active",
            lessons=["lesson1"],
            explanations=["explanation1"],
            exercises=["exercise1"],
        )
        assert resp.session_id == "sess-123"
        assert resp.status == "active"
        assert resp.lessons == ["lesson1"]
        assert resp.explanations == ["explanation1"]
        assert resp.exercises == ["exercise1"]

    def test_empty_lists_are_valid(self):
        resp = TeacherResponse(
            session_id="sess-456",
            status="completed",
            lessons=[],
            explanations=[],
            exercises=[],
        )
        assert resp.lessons == []
        assert resp.explanations == []
        assert resp.exercises == []

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            TeacherResponse(
                session_id="sess-789",
                status="active",
                lessons=[],
                explanations=[],
                # exercises missing
            )


class TestPersonalizationResponse:
    def test_valid_personalization_response(self):
        resp = PersonalizationResponse(
            tone="casual",
            style="concise",
            focus="high-level",
            reasons={"tone": "user prefers casual", "style": "short answers"},
        )
        assert resp.tone == "casual"
        assert resp.style == "concise"
        assert resp.focus == "high-level"
        assert isinstance(resp.reasons, dict)

    def test_empty_reasons_dict_is_valid(self):
        resp = PersonalizationResponse(
            tone="formal",
            style="detailed",
            focus="deep-technical",
            reasons={},
        )
        assert resp.reasons == {}


class TestErrorResponse:
    def test_valid_error_response(self):
        resp = ErrorResponse(error="Something went wrong", status=500)
        assert resp.error == "Something went wrong"
        assert resp.status == 500

    def test_status_422(self):
        resp = ErrorResponse(error="Validation failed", status=422)
        assert resp.status == 422

    def test_status_400(self):
        resp = ErrorResponse(error="Bad request", status=400)
        assert resp.status == 400

    def test_empty_error_string_is_valid(self):
        # ErrorResponse does not enforce non-empty error string
        resp = ErrorResponse(error="", status=500)
        assert resp.error == ""

    def test_missing_error_raises(self):
        with pytest.raises(ValidationError):
            ErrorResponse(status=500)

    def test_missing_status_raises(self):
        with pytest.raises(ValidationError):
            ErrorResponse(error="oops")

    def test_non_int_status_raises(self):
        with pytest.raises(ValidationError):
            ErrorResponse(error="oops", status="not-an-int")
