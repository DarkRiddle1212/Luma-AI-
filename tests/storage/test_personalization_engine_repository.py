"""
Unit tests for PersonalizationEngine optional repository injection.

Covers:
- In-memory mode (no repository): engine works without error, repository never called
- Persistence mode (injected repository): get_by_user() called before adapting,
  upsert() called after updating profile
- upsert() receives correct user_id, interests, preferences, strengths
- Empty memory list: upsert() still called with empty profile data
- user_id parameter is forwarded to both get_by_user() and upsert()
- No import from luma.storage in the personalization_engine module

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5
"""

from __future__ import annotations

from typing import Any, List, Optional
from unittest.mock import MagicMock, call

import pytest

from luma.core.memory_interface import (
    MemoryEntry,
    MemoryInterface,
    MemoryRetrievalError,
    RetrievalResult,
)
from luma.core.personalization.adaptation_engine import AdaptationEngine
from luma.core.personalization.personalization_engine import PersonalizationEngine
from luma.core.personalization.preference_detector import PreferenceDetector
from luma.core.personalization.profile_builder import ProfileBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockMemoryInterface(MemoryInterface):
    def __init__(self, memories: List[MemoryEntry]) -> None:
        self._memories = memories

    def store(self, content: str, metadata: Optional[dict] = None) -> str:
        raise AssertionError("PersonalizationEngine must never call store()")

    def retrieve(self, query=None, params=None, limit=10, **kwargs) -> RetrievalResult:
        return {
            "memories": self._memories,
            "total_count": len(self._memories),
            "query_metadata": {},
        }


def _make_memories(n: int) -> List[MemoryEntry]:
    """Create n MemoryEntry dicts with varied content."""
    topics = ["python", "java", "rust", "python", "java", "python"]
    return [
        {
            "id": f"m{i}",
            "content": f"{topics[i % len(topics)]} programming language tutorial",
            "metadata": {},
            "timestamp": f"2024-0{(i % 9) + 1}-01T00:00:00",
            "category": "tech",
            "tags": ["programming"],
        }
        for i in range(n)
    ]


def _build_engine(
    memories: List[MemoryEntry],
    personalization_repository: Optional[Any] = None,
) -> PersonalizationEngine:
    """Build a PersonalizationEngine with low thresholds for easy detection."""
    return PersonalizationEngine(
        memory_interface=MockMemoryInterface(memories),
        profile_builder=ProfileBuilder(min_keyword_frequency=1),
        preference_detector=PreferenceDetector(min_confidence=0.0, min_frequency=1),
        adaptation_engine=AdaptationEngine(),
        personalization_repository=personalization_repository,
    )


# ---------------------------------------------------------------------------
# Tests: in-memory mode (no repository) — Requirement 12.4
# ---------------------------------------------------------------------------


class TestInMemoryMode:
    def test_engine_works_without_repository(self):
        """Engine operates normally when personalization_repository=None."""
        engine = _build_engine(_make_memories(6))
        result = engine.personalize(input_data="hello", context="session")
        assert result is not None
        assert result.profile is not None
        assert result.adaptation is not None

    def test_repository_attribute_is_none_by_default(self):
        """personalization_repository defaults to None."""
        engine = _build_engine(_make_memories(3))
        assert engine._personalization_repository is None

    def test_empty_memories_no_error_without_repository(self):
        """Empty memory list with no repository returns result without error."""
        engine = _build_engine([])
        result = engine.personalize(input_data="hi", context="ctx")
        assert result.profile.interests == []
        assert result.profile.strengths == []

    def test_none_input_raises_type_error(self):
        """TypeError raised when input_data is None, regardless of repository."""
        engine = _build_engine(_make_memories(3))
        with pytest.raises(TypeError):
            engine.personalize(input_data=None, context="ctx")

    def test_none_context_raises_type_error(self):
        """TypeError raised when context is None, regardless of repository."""
        engine = _build_engine(_make_memories(3))
        with pytest.raises(TypeError):
            engine.personalize(input_data="hi", context=None)


# ---------------------------------------------------------------------------
# Tests: persistence mode — Requirements 12.1, 12.2, 12.3
# ---------------------------------------------------------------------------


class TestPersistenceMode:
    def test_get_by_user_called_before_upsert(self):
        """get_by_user() is called before upsert() in the same personalize() call."""
        mock_repo = MagicMock()
        mock_repo.get_by_user.return_value = None
        engine = _build_engine(_make_memories(6), personalization_repository=mock_repo)

        engine.personalize(input_data="hi", context="ctx", user_id="alice")

        # Both must have been called
        mock_repo.get_by_user.assert_called_once()
        mock_repo.upsert.assert_called_once()

        # get_by_user must appear before upsert in the call order
        get_idx = next(
            i for i, c in enumerate(mock_repo.mock_calls) if c[0] == "get_by_user"
        )
        upsert_idx = next(
            i for i, c in enumerate(mock_repo.mock_calls) if c[0] == "upsert"
        )
        assert get_idx < upsert_idx, "get_by_user() must be called before upsert()"

    def test_get_by_user_called_with_correct_user_id(self):
        """get_by_user() receives the user_id passed to personalize()."""
        mock_repo = MagicMock()
        mock_repo.get_by_user.return_value = None
        engine = _build_engine(_make_memories(6), personalization_repository=mock_repo)

        engine.personalize(input_data="hi", context="ctx", user_id="bob")

        mock_repo.get_by_user.assert_called_once_with("bob")

    def test_upsert_called_with_correct_user_id(self):
        """upsert() receives the user_id passed to personalize()."""
        mock_repo = MagicMock()
        mock_repo.get_by_user.return_value = None
        engine = _build_engine(_make_memories(6), personalization_repository=mock_repo)

        engine.personalize(input_data="hi", context="ctx", user_id="carol")

        args, kwargs = mock_repo.upsert.call_args
        assert args[0] == "carol" or kwargs.get("user_id") == "carol"

    def test_upsert_called_with_interests(self):
        """upsert() receives interests as a list."""
        mock_repo = MagicMock()
        mock_repo.get_by_user.return_value = None
        engine = _build_engine(_make_memories(6), personalization_repository=mock_repo)

        engine.personalize(input_data="hi", context="ctx", user_id="u1")

        _, kwargs = mock_repo.upsert.call_args
        assert "interests" in kwargs
        assert isinstance(kwargs["interests"], list)

    def test_upsert_called_with_strengths(self):
        """upsert() receives strengths as a list."""
        mock_repo = MagicMock()
        mock_repo.get_by_user.return_value = None
        engine = _build_engine(_make_memories(6), personalization_repository=mock_repo)

        engine.personalize(input_data="hi", context="ctx", user_id="u1")

        _, kwargs = mock_repo.upsert.call_args
        assert "strengths" in kwargs
        assert isinstance(kwargs["strengths"], list)

    def test_upsert_called_with_preferences_dict(self):
        """upsert() receives preferences as a dict."""
        mock_repo = MagicMock()
        mock_repo.get_by_user.return_value = None
        engine = _build_engine(_make_memories(6), personalization_repository=mock_repo)

        engine.personalize(input_data="hi", context="ctx", user_id="u1")

        _, kwargs = mock_repo.upsert.call_args
        assert "preferences" in kwargs
        assert isinstance(kwargs["preferences"], dict)

    def test_upsert_called_once_per_personalize_call(self):
        """upsert() is called exactly once per personalize() invocation."""
        mock_repo = MagicMock()
        mock_repo.get_by_user.return_value = None
        engine = _build_engine(_make_memories(6), personalization_repository=mock_repo)

        engine.personalize(input_data="hi", context="ctx", user_id="u1")
        engine.personalize(input_data="hi", context="ctx", user_id="u1")

        assert mock_repo.upsert.call_count == 2
        assert mock_repo.get_by_user.call_count == 2

    def test_result_still_returned_with_repository(self):
        """PersonalizationResult is returned correctly even when repository is injected."""
        mock_repo = MagicMock()
        mock_repo.get_by_user.return_value = None
        engine = _build_engine(_make_memories(6), personalization_repository=mock_repo)

        result = engine.personalize(input_data="hi", context="ctx", user_id="u1")

        assert result is not None
        assert result.profile is not None
        assert result.preferences is not None
        assert result.adaptation is not None

    def test_default_user_id_forwarded_to_repository(self):
        """Default user_id='default' is forwarded to get_by_user() and upsert()."""
        mock_repo = MagicMock()
        mock_repo.get_by_user.return_value = None
        engine = _build_engine(_make_memories(6), personalization_repository=mock_repo)

        engine.personalize(input_data="hi", context="ctx")

        mock_repo.get_by_user.assert_called_once_with("default")
        args, kwargs = mock_repo.upsert.call_args
        assert args[0] == "default" or kwargs.get("user_id") == "default"

    def test_empty_memories_upsert_still_called(self):
        """upsert() is still called even when no memories are retrieved."""
        mock_repo = MagicMock()
        mock_repo.get_by_user.return_value = None
        engine = _build_engine([], personalization_repository=mock_repo)

        engine.personalize(input_data="hi", context="ctx", user_id="u1")

        mock_repo.upsert.assert_called_once()
        _, kwargs = mock_repo.upsert.call_args
        assert kwargs["interests"] == []
        assert kwargs["strengths"] == []


# ---------------------------------------------------------------------------
# Tests: no import from luma.storage — Requirement 12.5
# ---------------------------------------------------------------------------


class TestNoStorageImport:
    def test_personalization_engine_does_not_import_luma_storage(self):
        """PersonalizationEngine module must not import from luma.storage."""
        import luma.core.personalization.personalization_engine as module

        source_file = module.__file__
        with open(source_file) as f:
            source = f.read()

        assert "from luma.storage" not in source
        assert "import luma.storage" not in source
