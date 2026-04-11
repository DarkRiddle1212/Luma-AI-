"""
Unit tests for Reasoning Engine data models.

Tests Pydantic validation, field types, optional fields, and serialization
for Reasoning_Request, Reasoning_Context, and Reasoning_Result models.
"""

import pytest
from pydantic import ValidationError
from luma.core.reasoning.schemas import (
    Reasoning_Request,
    Reasoning_Context,
    Reasoning_Result
)


class TestReasoningContext:
    """Test suite for Reasoning_Context data model."""
    
    def test_empty_context_creation(self):
        """Test creating an empty Reasoning_Context with default values."""
        context = Reasoning_Context()
        assert context.memories == []
        assert context.metadata == {}
    
    def test_context_with_memories(self):
        """Test creating Reasoning_Context with memories."""
        memories = [
            {"id": "mem_1", "content": "User likes Python", "metadata": {}},
            {"id": "mem_2", "content": "User prefers VS Code", "metadata": {}}
        ]
        context = Reasoning_Context(memories=memories)
        assert len(context.memories) == 2
        assert context.memories[0]["id"] == "mem_1"
        assert context.memories[1]["content"] == "User prefers VS Code"
    
    def test_context_with_metadata(self):
        """Test creating Reasoning_Context with metadata."""
        metadata = {"user_id": "123", "session_id": "abc", "timestamp": "2024-01-01"}
        context = Reasoning_Context(metadata=metadata)
        assert context.metadata["user_id"] == "123"
        assert context.metadata["session_id"] == "abc"
    
    def test_context_serialization(self):
        """Test Reasoning_Context serialization to dict."""
        context = Reasoning_Context(
            memories=[{"id": "mem_1", "content": "test", "metadata": {}}],
            metadata={"user_id": "123"}
        )
        data = context.model_dump()
        assert "memories" in data
        assert "metadata" in data
        assert data["memories"][0]["id"] == "mem_1"
        assert data["metadata"]["user_id"] == "123"
    
    def test_context_deserialization(self):
        """Test Reasoning_Context deserialization from dict."""
        data = {
            "memories": [{"id": "mem_1", "content": "test", "metadata": {}}],
            "metadata": {"user_id": "123"}
        }
        context = Reasoning_Context(**data)
        assert len(context.memories) == 1
        assert context.metadata["user_id"] == "123"


class TestReasoningRequest:
    """Test suite for Reasoning_Request data model."""
    
    def test_request_with_query_only(self):
        """Test creating Reasoning_Request with query only."""
        request = Reasoning_Request(query="What is Python?")
        assert request.query == "What is Python?"
        assert request.context.memories == []
        assert request.context.metadata == {}
    
    def test_request_with_query_and_context(self):
        """Test creating Reasoning_Request with query and context."""
        context = Reasoning_Context(
            memories=[{"id": "mem_1", "content": "User likes Python", "metadata": {}}]
        )
        request = Reasoning_Request(query="What do I like?", context=context)
        assert request.query == "What do I like?"
        assert len(request.context.memories) == 1
    
    def test_request_empty_query_validation(self):
        """Test that empty query string is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Reasoning_Request(query="")
        assert "query" in str(exc_info.value)
    
    def test_request_missing_query_validation(self):
        """Test that missing query field is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Reasoning_Request()
        assert "query" in str(exc_info.value)
    
    def test_request_query_type_validation(self):
        """Test that non-string query is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Reasoning_Request(query=123)
        assert "query" in str(exc_info.value)
    
    def test_request_serialization(self):
        """Test Reasoning_Request serialization to dict."""
        request = Reasoning_Request(
            query="Test query",
            context=Reasoning_Context(metadata={"user_id": "123"})
        )
        data = request.model_dump()
        assert data["query"] == "Test query"
        assert "context" in data
        assert data["context"]["metadata"]["user_id"] == "123"
    
    def test_request_deserialization(self):
        """Test Reasoning_Request deserialization from dict."""
        data = {
            "query": "Test query",
            "context": {
                "memories": [{"id": "mem_1", "content": "test", "metadata": {}}],
                "metadata": {"user_id": "123"}
            }
        }
        request = Reasoning_Request(**data)
        assert request.query == "Test query"
        assert len(request.context.memories) == 1


class TestReasoningResult:
    """Test suite for Reasoning_Result data model."""
    
    def test_result_with_answer_only(self):
        """Test creating Reasoning_Result with answer only."""
        result = Reasoning_Result(answer="Python is a programming language.")
        assert result.answer == "Python is a programming language."
        assert result.used_memories == []
        assert result.confidence is None
    
    def test_result_with_used_memories(self):
        """Test creating Reasoning_Result with used memories."""
        result = Reasoning_Result(
            answer="You like Python and JavaScript.",
            used_memories=["mem_1", "mem_2"]
        )
        assert result.answer == "You like Python and JavaScript."
        assert len(result.used_memories) == 2
        assert "mem_1" in result.used_memories
    
    def test_result_with_confidence(self):
        """Test creating Reasoning_Result with confidence score."""
        result = Reasoning_Result(
            answer="Test answer",
            confidence=0.95
        )
        assert result.confidence == 0.95
    
    def test_result_confidence_validation_min(self):
        """Test that confidence below 0.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Reasoning_Result(answer="Test", confidence=-0.1)
        assert "confidence" in str(exc_info.value)
    
    def test_result_confidence_validation_max(self):
        """Test that confidence above 1.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Reasoning_Result(answer="Test", confidence=1.5)
        assert "confidence" in str(exc_info.value)
    
    def test_result_confidence_boundary_values(self):
        """Test that confidence boundary values (0.0 and 1.0) are accepted."""
        result_min = Reasoning_Result(answer="Test", confidence=0.0)
        result_max = Reasoning_Result(answer="Test", confidence=1.0)
        assert result_min.confidence == 0.0
        assert result_max.confidence == 1.0
    
    def test_result_missing_answer_validation(self):
        """Test that missing answer field is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Reasoning_Result()
        assert "answer" in str(exc_info.value)
    
    def test_result_answer_type_validation(self):
        """Test that non-string answer is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Reasoning_Result(answer=123)
        assert "answer" in str(exc_info.value)
    
    def test_result_serialization(self):
        """Test Reasoning_Result serialization to dict."""
        result = Reasoning_Result(
            answer="Test answer",
            used_memories=["mem_1", "mem_2"],
            confidence=0.85
        )
        data = result.model_dump()
        assert data["answer"] == "Test answer"
        assert len(data["used_memories"]) == 2
        assert data["confidence"] == 0.85
    
    def test_result_deserialization(self):
        """Test Reasoning_Result deserialization from dict."""
        data = {
            "answer": "Test answer",
            "used_memories": ["mem_1"],
            "confidence": 0.9
        }
        result = Reasoning_Result(**data)
        assert result.answer == "Test answer"
        assert result.used_memories == ["mem_1"]
        assert result.confidence == 0.9
    
    def test_result_serialization_with_none_confidence(self):
        """Test that None confidence is properly serialized."""
        result = Reasoning_Result(answer="Test")
        data = result.model_dump()
        assert data["confidence"] is None


class TestDataModelIntegration:
    """Integration tests for data models working together."""
    
    def test_full_reasoning_flow_models(self):
        """Test complete flow from request to result."""
        # Create request with context
        context = Reasoning_Context(
            memories=[
                {"id": "mem_1", "content": "User likes Python", "metadata": {}},
                {"id": "mem_2", "content": "User uses VS Code", "metadata": {}}
            ],
            metadata={"user_id": "123", "session_id": "abc"}
        )
        request = Reasoning_Request(
            query="What programming language do I like?",
            context=context
        )
        
        # Simulate processing and create result
        result = Reasoning_Result(
            answer="Based on your memories, you like Python.",
            used_memories=["mem_1"],
            confidence=0.92
        )
        
        # Verify the flow
        assert request.query == "What programming language do I like?"
        assert len(request.context.memories) == 2
        assert result.answer == "Based on your memories, you like Python."
        assert result.used_memories == ["mem_1"]
        assert result.confidence == 0.92
    
    def test_serialization_round_trip(self):
        """Test that models can be serialized and deserialized without data loss."""
        # Create original request
        original_request = Reasoning_Request(
            query="Test query",
            context=Reasoning_Context(
                memories=[{"id": "mem_1", "content": "test", "metadata": {}}],
                metadata={"user_id": "123"}
            )
        )
        
        # Serialize and deserialize
        data = original_request.model_dump()
        restored_request = Reasoning_Request(**data)
        
        # Verify no data loss
        assert restored_request.query == original_request.query
        assert len(restored_request.context.memories) == len(original_request.context.memories)
        assert restored_request.context.metadata == original_request.context.metadata
