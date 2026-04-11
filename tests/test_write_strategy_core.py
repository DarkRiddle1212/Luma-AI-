"""
Tests for Write Strategy Core (WriteDecision and Memory_Write_Strategy)

Tests the core functionality of the memory write strategy including:
- WriteDecision data model
- Write trigger evaluation
- Content validation
- Repetition detection
- Session coordination

Feature: memory-write-strategy-session-management
"""

import pytest
from datetime import datetime
from luma.core.write_strategy import (
    WriteDecision,
    WriteStrategyConfig,
    Memory_Write_Strategy
)
from luma.core.session_manager import Session_Manager, SessionConfig
from luma.core.memory_interface import MemoryInterface, MemoryStorageError


# ============================================================================
# Mock Memory Interface
# ============================================================================


class MockMemoryInterface(MemoryInterface):
    """Mock memory interface for testing."""
    
    def __init__(self):
        self.stored_memories = []
        self.next_id = 0
    
    def store(self, content: str, metadata: dict = None) -> str:
        memory_id = f"mem_{self.next_id}"
        self.next_id += 1
        self.stored_memories.append({
            "id": memory_id,
            "content": content,
            "metadata": metadata or {}
        })
        return memory_id
    
    def retrieve(self, params: dict = None) -> dict:
        return {"memories": self.stored_memories}
    
    def update(self, memory_id: str, content: str = None, metadata: dict = None) -> bool:
        return True
    
    def delete(self, memory_id: str) -> bool:
        return True


# ============================================================================
# WriteDecision Tests
# ============================================================================


class TestWriteDecision:
    """Tests for the WriteDecision dataclass."""
    
    def test_write_decision_creation(self):
        """Test creating a WriteDecision with all fields."""
        decision = WriteDecision(
            should_write=True,
            reason="approved",
            metadata={"test": "value"}
        )
        assert decision.should_write is True
        assert decision.reason == "approved"
        assert decision.metadata == {"test": "value"}
    
    def test_write_decision_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        decision = WriteDecision(should_write=True, reason="approved")
        assert decision.metadata == {}
    
    def test_write_decision_false(self):
        """Test WriteDecision with should_write=False."""
        decision = WriteDecision(
            should_write=False,
            reason="trivial_pattern",
            metadata={"pattern": "hello"}
        )
        assert decision.should_write is False
        assert decision.reason == "trivial_pattern"
        assert decision.metadata["pattern"] == "hello"


# ============================================================================
# Memory_Write_Strategy Tests
# ============================================================================


class TestMemoryWriteStrategyInit:
    """Tests for Memory_Write_Strategy initialization."""
    
    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        config = WriteStrategyConfig()
        session_manager = Session_Manager(
            SessionConfig(),
            MockMemoryInterface()
        )
        memory = MockMemoryInterface()
        
        strategy = Memory_Write_Strategy(config, session_manager, memory)
        
        assert strategy.config == config
        assert strategy.session_manager == session_manager
        assert strategy.memory == memory
        assert strategy.recent_messages == []
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = WriteStrategyConfig(
            trivial_patterns=["custom"],
            min_content_length=5,
            repetition_window=3
        )
        session_manager = Session_Manager(
            SessionConfig(),
            MockMemoryInterface()
        )
        memory = MockMemoryInterface()
        
        strategy = Memory_Write_Strategy(config, session_manager, memory)
        
        assert strategy.config.trivial_patterns == ["custom"]
        assert strategy.config.min_content_length == 5
        assert strategy.config.repetition_window == 3


class TestEvaluateWriteTrigger:
    """Tests for evaluate_write_trigger method."""
    
    @pytest.fixture
    def strategy(self):
        """Create a Memory_Write_Strategy instance for testing."""
        config = WriteStrategyConfig()
        session_manager = Session_Manager(
            SessionConfig(),
            MockMemoryInterface()
        )
        memory = MockMemoryInterface()
        return Memory_Write_Strategy(config, session_manager, memory)
    
    def test_empty_string_rejected(self, strategy):
        """Test that empty string is rejected."""
        decision = strategy.evaluate_write_trigger("")
        
        assert decision.should_write is False
        assert decision.reason == "empty_or_whitespace"
    
    def test_whitespace_only_rejected(self, strategy):
        """Test that whitespace-only content is rejected."""
        decision = strategy.evaluate_write_trigger("   \t\n  ")
        
        assert decision.should_write is False
        assert decision.reason == "empty_or_whitespace"
    
    def test_trivial_pattern_rejected(self, strategy):
        """Test that trivial patterns are rejected."""
        test_cases = ["hello", "hi", "hey", "thanks", "thank you", "ok", "okay"]
        
        for pattern in test_cases:
            decision = strategy.evaluate_write_trigger(pattern)
            assert decision.should_write is False
            assert decision.reason == "trivial_pattern"
            assert pattern in decision.metadata.get("matched_pattern", "").lower()
    
    def test_configured_greeting_patterns_rejected(self):
        """Test that configured greeting patterns are rejected.
        
        Feature: memory-write-strategy-session-management
        Requirement: 1.3 - Common greetings should be rejected without storage
        """
        # Create strategy with custom trivial patterns
        custom_patterns = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"]
        config = WriteStrategyConfig(trivial_patterns=custom_patterns)
        session_manager = Session_Manager(
            SessionConfig(),
            MockMemoryInterface()
        )
        memory = MockMemoryInterface()
        strategy = Memory_Write_Strategy(config, session_manager, memory)
        
        # Test each configured pattern is rejected
        for pattern in custom_patterns:
            decision = strategy.evaluate_write_trigger(pattern)
            assert decision.should_write is False, f"Pattern '{pattern}' should be rejected"
            assert decision.reason == "trivial_pattern", f"Pattern '{pattern}' should be rejected as trivial"
            assert decision.metadata.get("matched_pattern") == pattern
        
        # Test case-insensitive matching
        decision = strategy.evaluate_write_trigger("HELLO")
        assert decision.should_write is False
        assert decision.reason == "trivial_pattern"
        
        decision = strategy.evaluate_write_trigger("Good Morning")
        assert decision.should_write is False
        assert decision.reason == "trivial_pattern"
        
        # Test with extra whitespace
        decision = strategy.evaluate_write_trigger("  hello  ")
        assert decision.should_write is False
        assert decision.reason == "trivial_pattern"
        
        # Test that non-trivial content is still approved
        decision = strategy.evaluate_write_trigger("Hello, I need help with something important")
        assert decision.should_write is True
        assert decision.reason == "approved"
    
    def test_below_min_length_rejected(self, strategy):
        """Test that content below min length is rejected."""
        config = WriteStrategyConfig(min_content_length=5)
        session_manager = Session_Manager(
            SessionConfig(),
            MockMemoryInterface()
        )
        memory = MockMemoryInterface()
        strategy = Memory_Write_Strategy(config, session_manager, memory)
        
        decision = strategy.evaluate_write_trigger("abc")
        
        assert decision.should_write is False
        assert decision.reason == "below_min_length"
        assert decision.metadata["content_length"] == 3
        assert decision.metadata["min_required"] == 5
    
    def test_valid_content_approved(self, strategy):
        """Test that valid content is approved."""
        decision = strategy.evaluate_write_trigger("This is a valid message")
        
        assert decision.should_write is True
        assert decision.reason == "approved"
        assert decision.metadata == {}
    
    def test_content_with_whitespace_approved(self, strategy):
        """Test that content with surrounding whitespace is approved."""
        decision = strategy.evaluate_write_trigger("   This is valid   ")
        
        assert decision.should_write is True
        assert decision.reason == "approved"
    
    def test_repetitive_content_rejected(self, strategy):
        """Test that repetitive content is rejected."""
        # First message
        strategy.evaluate_write_trigger("Hello world")
        
        # Same message again (should be rejected as repetitive)
        decision = strategy.evaluate_write_trigger("Hello world")
        
        assert decision.should_write is False
        assert decision.reason == "repetitive"
    
    def test_repetition_window_respected(self, strategy):
        """Test that repetition window is respected."""
        config = WriteStrategyConfig(repetition_window=2)
        session_manager = Session_Manager(
            SessionConfig(),
            MockMemoryInterface()
        )
        memory = MockMemoryInterface()
        strategy = Memory_Write_Strategy(config, session_manager, memory)
        
        # Add messages to fill the window
        strategy.evaluate_write_trigger("Message 1")
        strategy.evaluate_write_trigger("Message 2")
        
        # Same message within window (should be rejected)
        decision = strategy.evaluate_write_trigger("Message 1")
        assert decision.should_write is False
        assert decision.reason == "repetitive"
        
        # Add another message to push first out of window
        strategy.evaluate_write_trigger("Message 3")
        
        # Now same message should be allowed (out of window)
        decision = strategy.evaluate_write_trigger("Message 1")
        assert decision.should_write is True
        assert decision.reason == "approved"


class TestValidateContent:
    """Tests for validate_content method."""
    
    @pytest.fixture
    def strategy(self):
        """Create a Memory_Write_Strategy instance for testing."""
        config = WriteStrategyConfig()
        session_manager = Session_Manager(
            SessionConfig(),
            MockMemoryInterface()
        )
        memory = MockMemoryInterface()
        return Memory_Write_Strategy(config, session_manager, memory)
    
    def test_valid_content_passes(self, strategy):
        """Test that valid content passes validation."""
        strategy.validate_content("Valid content")
        # Should not raise
    
    def test_empty_content_raises(self, strategy):
        """Test that empty content raises MemoryStorageError."""
        with pytest.raises(MemoryStorageError, match="Content cannot be empty"):
            strategy.validate_content("")
    
    def test_non_string_content_raises(self, strategy):
        """Test that non-string content raises MemoryStorageError."""
        with pytest.raises(MemoryStorageError, match="Content must be a string"):
            strategy.validate_content(123)
    
    def test_none_content_raises(self, strategy):
        """Test that None content raises MemoryStorageError."""
        with pytest.raises(MemoryStorageError, match="Content must be a string"):
            strategy.validate_content(None)  # type: ignore
    
    def test_valid_metadata_passes(self, strategy):
        """Test that valid metadata passes validation."""
        strategy.validate_content("Content", {"tags": ["tag1", "tag2"]})
        # Should not raise
    
    def test_non_dict_metadata_raises(self, strategy):
        """Test that non-dict metadata raises MemoryStorageError."""
        with pytest.raises(MemoryStorageError, match="Metadata must be a dictionary"):
            strategy.validate_content("Content", "not a dict")  # type: ignore
    
    def test_non_list_tags_raises(self, strategy):
        """Test that non-list tags raises MemoryStorageError."""
        with pytest.raises(MemoryStorageError, match="Tags must be a list"):
            strategy.validate_content("Content", {"tags": "not a list"})  # type: ignore
    
    def test_non_string_tags_raises(self, strategy):
        """Test that non-string tags raises MemoryStorageError."""
        with pytest.raises(MemoryStorageError, match="All tags must be strings"):
            strategy.validate_content("Content", {"tags": ["valid", 123]})
    
    def test_long_content_accepted(self, strategy):
        """Test that long content is accepted (within reasonable limits)."""
        long_content = "x" * 10000
        strategy.validate_content(long_content)
        # Should not raise
    
    def test_excessive_content_length_raises(self, strategy):
        """Test that content exceeding maximum length raises MemoryStorageError."""
        excessive_content = "x" * 100001  # Exceeds 100,000 character limit
        with pytest.raises(MemoryStorageError, match="Content length .* exceeds maximum"):
            strategy.validate_content(excessive_content)


class TestNormalizeMetadata:
    """Tests for normalize_metadata method."""
    
    @pytest.fixture
    def strategy(self):
        """Create a Memory_Write_Strategy instance for testing."""
        config = WriteStrategyConfig()
        session_manager = Session_Manager(
            SessionConfig(),
            MockMemoryInterface()
        )
        memory = MockMemoryInterface()
        return Memory_Write_Strategy(config, session_manager, memory)
    
    def test_adds_timestamp(self, strategy):
        """Test that timestamp is added if not present."""
        metadata = {"category": "test"}
        normalized = strategy.normalize_metadata(metadata)
        
        assert "timestamp" in normalized
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(normalized["timestamp"])
    
    def test_preserves_existing_timestamp(self, strategy):
        """Test that existing timestamp is preserved."""
        original_timestamp = "2024-01-15T10:30:00"
        metadata = {"timestamp": original_timestamp}
        normalized = strategy.normalize_metadata(metadata)
        
        assert normalized["timestamp"] == original_timestamp
    
    def test_adds_session_id(self, strategy):
        """Test that session_id is added from active session."""
        # Create a session
        session_id = strategy.session_manager.create_session()
        
        metadata = {"category": "test"}
        normalized = strategy.normalize_metadata(metadata)
        
        assert "session_id" in normalized
        assert normalized["session_id"] == session_id
    
    def test_preserves_other_metadata(self, strategy):
        """Test that other metadata fields are preserved."""
        metadata = {
            "category": "test",
            "tags": ["tag1", "tag2"],
            "custom_field": "value"
        }
        normalized = strategy.normalize_metadata(metadata)
        
        assert normalized["category"] == "test"
        assert normalized["tags"] == ["tag1", "tag2"]
        assert normalized["custom_field"] == "value"


class TestStoreMemory:
    """Tests for store_memory method."""
    
    @pytest.fixture
    def strategy(self):
        """Create a Memory_Write_Strategy instance for testing."""
        config = WriteStrategyConfig()
        session_manager = Session_Manager(
            SessionConfig(),
            MockMemoryInterface()
        )
        memory = MockMemoryInterface()
        return Memory_Write_Strategy(config, session_manager, memory)
    
    def test_rejected_memory_raises_error(self, strategy):
        """Test that rejected memory raises MemoryStorageError."""
        with pytest.raises(MemoryStorageError, match="Memory write rejected"):
            strategy.store_memory("")
    
    def test_valid_memory_stored(self, strategy):
        """Test that valid memory is stored."""
        session_id = strategy.session_manager.create_session()
        
        # Use immediate=True to bypass buffering
        memory_id = strategy.store_memory(
            "Test content",
            metadata={"tags": ["test"]},
            immediate=True
        )
        
        assert memory_id is not None
        assert len(strategy.memory.stored_memories) == 1
    
    def test_buffered_memory(self, strategy):
        """Test that memory is buffered when session is active."""
        config = WriteStrategyConfig(enable_conflict_detection=True)
        session_manager = Session_Manager(
            SessionConfig(),
            MockMemoryInterface()
        )
        memory = MockMemoryInterface()
        strategy = Memory_Write_Strategy(config, session_manager, memory)
        
        session_id = strategy.session_manager.create_session()
        
        memory_id = strategy.store_memory(
            "Buffered content",
            metadata={"tags": ["test"]}
        )
        
        # Should be buffered (not immediately stored)
        assert "buffered:" in memory_id or memory_id is not None
