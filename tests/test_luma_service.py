"""
Unit tests for LumaService.

Tests verify:
- process_chat calls core modules in the correct order
- get_insights forwards namespace correctly
- start_teacher_mode and continue_teacher_mode both call TeacherMode.teach
- get_personalization calls PersonalizationEngine.personalize(user_id, "")
- No core module is instantiated inside LumaService
"""

import asyncio
import pytest
from unittest.mock import MagicMock, call, patch


def _make_service():
    """Build a LumaService with all dependencies mocked."""
    from luma.api.services.luma_service import LumaService

    memory = MagicMock()
    insight_engine = MagicMock()
    insight_moments_engine = MagicMock()
    personalization_engine = MagicMock()
    teacher_mode = MagicMock()

    # Configure default return values
    memory.retrieve.return_value = {
        "memories": [{"content": "past memory", "id": "1", "metadata": {},
                       "timestamp": "2024-01-01T00:00:00", "category": "chat", "tags": []}],
        "total_count": 1,
        "query_metadata": {},
    }
    memory.store.return_value = "mem_id_1"

    adaptation_ctx = MagicMock()
    adaptation_ctx.tone = "casual"
    adaptation_ctx.style = "concise"
    adaptation_ctx.focus = "high-level"
    adaptation_ctx.reasons = {}
    # Support model_dump
    adaptation_ctx.model_dump.return_value = {
        "tone": "casual", "style": "concise", "focus": "high-level", "reasons": {}
    }

    personalization_result = MagicMock()
    personalization_result.adaptation = adaptation_ctx
    personalization_engine.personalize.return_value = personalization_result

    insight_moments_engine.generate_moments.return_value = []

    insight_report = MagicMock()
    insight_report.insights = ["insight_a", "insight_b"]
    insight_engine.generate_insights.return_value = insight_report

    teaching_session = MagicMock()
    teaching_session.session_id = "sess-1"
    teaching_session.status = "active"
    teaching_session.lessons = []
    teaching_session.explanations = []
    teaching_session.exercises = []
    teacher_mode.teach.return_value = teaching_session

    service = LumaService(
        memory_interface=memory,
        insight_engine=insight_engine,
        insight_moments_engine=insight_moments_engine,
        personalization_engine=personalization_engine,
        teacher_mode=teacher_mode,
    )
    return service, memory, insight_engine, insight_moments_engine, personalization_engine, teacher_mode


def run(coro):
    """Helper to run a coroutine synchronously in tests."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# process_chat
# ---------------------------------------------------------------------------

class TestProcessChat:
    def test_returns_required_keys(self):
        service, *_ = _make_service()
        result = run(service.process_chat("alice", "hello"))
        assert "response" in result
        assert "insight_moments" in result
        assert "personalization" in result

    def test_response_is_non_empty_string(self):
        service, *_ = _make_service()
        result = run(service.process_chat("alice", "hello"))
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_insight_moments_is_list(self):
        service, *_ = _make_service()
        result = run(service.process_chat("alice", "hello"))
        assert isinstance(result["insight_moments"], list)

    def test_personalization_is_dict_with_required_keys(self):
        service, *_ = _make_service()
        result = run(service.process_chat("alice", "hello"))
        p = result["personalization"]
        assert isinstance(p, dict)
        assert "tone" in p
        assert "style" in p
        assert "focus" in p

    def test_memory_retrieve_called_with_correct_params(self):
        service, memory, *_ = _make_service()
        run(service.process_chat("alice", "hello"))
        memory.retrieve.assert_called_once_with(
            params={"query": "hello", "limit": 10}
        )

    def test_personalization_engine_called_with_user_id_and_message(self):
        service, _, _, _, personalization_engine, _ = _make_service()
        run(service.process_chat("alice", "hello"))
        personalization_engine.personalize.assert_called_once_with("alice", "hello")

    def test_insight_moments_engine_called(self):
        service, _, _, insight_moments_engine, _, _ = _make_service()
        run(service.process_chat("alice", "hello"))
        insight_moments_engine.generate_moments.assert_called_once()
        call_kwargs = insight_moments_engine.generate_moments.call_args
        assert call_kwargs.kwargs.get("insights") == [] or call_kwargs.args[0] == []

    def test_memory_store_called_with_message_and_metadata(self):
        service, memory, *_ = _make_service()
        run(service.process_chat("alice", "hello"))
        memory.store.assert_called_once_with(
            "hello",
            metadata={"user_id": "alice", "category": "chat"},
        )

    def test_orchestration_order(self):
        """retrieve → personalize → generate_moments → store (in that order)."""
        service, memory, _, insight_moments_engine, personalization_engine, _ = _make_service()
        manager = MagicMock()
        manager.attach_mock(memory.retrieve, "retrieve")
        manager.attach_mock(personalization_engine.personalize, "personalize")
        manager.attach_mock(insight_moments_engine.generate_moments, "generate_moments")
        manager.attach_mock(memory.store, "store")

        run(service.process_chat("alice", "hello"))

        call_names = [c[0] for c in manager.mock_calls]
        assert call_names.index("retrieve") < call_names.index("personalize")
        assert call_names.index("personalize") < call_names.index("generate_moments")
        assert call_names.index("generate_moments") < call_names.index("store")


# ---------------------------------------------------------------------------
# get_insights
# ---------------------------------------------------------------------------

class TestGetInsights:
    def test_with_namespace_forwards_namespace(self):
        service, _, insight_engine, *_ = _make_service()
        result = run(service.get_insights(namespace="foo"))
        insight_engine.generate_insights.assert_called_once_with(namespace="foo")
        assert result == ["insight_a", "insight_b"]

    def test_without_namespace_calls_without_argument(self):
        service, _, insight_engine, *_ = _make_service()
        run(service.get_insights())
        insight_engine.generate_insights.assert_called_once_with()

    def test_namespace_none_calls_without_argument(self):
        service, _, insight_engine, *_ = _make_service()
        run(service.get_insights(namespace=None))
        insight_engine.generate_insights.assert_called_once_with()

    def test_returns_insights_list(self):
        service, *_ = _make_service()
        result = run(service.get_insights(namespace="bar"))
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# get_insight_moments
# ---------------------------------------------------------------------------

class TestGetInsightMoments:
    def test_returns_list(self):
        service, *_ = _make_service()
        result = run(service.get_insight_moments())
        assert isinstance(result, list)

    def test_calls_generate_moments_with_empty_insights(self):
        service, _, _, insight_moments_engine, *_ = _make_service()
        run(service.get_insight_moments())
        insight_moments_engine.generate_moments.assert_called_once()
        call_kwargs = insight_moments_engine.generate_moments.call_args
        # insights must be []
        insights_arg = (
            call_kwargs.kwargs.get("insights")
            if call_kwargs.kwargs
            else call_kwargs.args[0]
        )
        assert insights_arg == []


# ---------------------------------------------------------------------------
# start_teacher_mode / continue_teacher_mode
# ---------------------------------------------------------------------------

class TestTeacherMode:
    def test_start_teacher_mode_calls_teach(self):
        service, _, _, _, _, teacher_mode = _make_service()
        result = run(service.start_teacher_mode("alice", "python"))
        teacher_mode.teach.assert_called_once_with("alice", "python")

    def test_continue_teacher_mode_calls_teach(self):
        service, _, _, _, _, teacher_mode = _make_service()
        result = run(service.continue_teacher_mode("alice", "python"))
        teacher_mode.teach.assert_called_once_with("alice", "python")

    def test_start_returns_teaching_session(self):
        service, _, _, _, _, teacher_mode = _make_service()
        result = run(service.start_teacher_mode("alice", "python"))
        assert result is teacher_mode.teach.return_value

    def test_continue_returns_teaching_session(self):
        service, _, _, _, _, teacher_mode = _make_service()
        result = run(service.continue_teacher_mode("alice", "python"))
        assert result is teacher_mode.teach.return_value


# ---------------------------------------------------------------------------
# get_personalization
# ---------------------------------------------------------------------------

class TestGetPersonalization:
    def test_calls_personalize_with_user_id_and_empty_string(self):
        service, _, _, _, personalization_engine, _ = _make_service()
        run(service.get_personalization("alice"))
        personalization_engine.personalize.assert_called_once_with("alice", "")

    def test_returns_adaptation_context(self):
        service, _, _, _, personalization_engine, _ = _make_service()
        result = run(service.get_personalization("alice"))
        assert result is personalization_engine.personalize.return_value.adaptation


# ---------------------------------------------------------------------------
# No core module instantiation inside LumaService
# ---------------------------------------------------------------------------

class TestNoCoreInstantiation:
    def test_luma_service_does_not_instantiate_memory_interface(self):
        """LumaService must not import or instantiate MemoryInterface internally."""
        from luma.api.services import luma_service as svc_module
        import inspect
        source = inspect.getsource(svc_module.LumaService.__init__)
        # Should not contain any direct class instantiation of core modules
        assert "MemoryInterface(" not in source
        assert "InsightEngine(" not in source
        assert "InsightMomentsEngine(" not in source
        assert "PersonalizationEngine(" not in source
        assert "TeacherMode(" not in source

    def test_all_dependencies_come_from_constructor(self):
        """All injected dependencies are stored as instance attributes."""
        from luma.api.services.luma_service import LumaService
        memory = MagicMock()
        insight_engine = MagicMock()
        insight_moments_engine = MagicMock()
        personalization_engine = MagicMock()
        teacher_mode = MagicMock()

        service = LumaService(
            memory_interface=memory,
            insight_engine=insight_engine,
            insight_moments_engine=insight_moments_engine,
            personalization_engine=personalization_engine,
            teacher_mode=teacher_mode,
        )

        assert service._memory_interface is memory
        assert service._insight_engine is insight_engine
        assert service._insight_moments_engine is insight_moments_engine
        assert service._personalization_engine is personalization_engine
        assert service._teacher_mode is teacher_mode
