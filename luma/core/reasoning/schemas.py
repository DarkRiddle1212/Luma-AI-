"""
Data models for the Reasoning Engine.

This module defines Pydantic data models for reasoning requests, context, and results.
All models use strong typing and validation to ensure type safety and clear contracts.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class Reasoning_Context(BaseModel):
    """
    Data model representing injected context structure.
    
    This model defines the structure of context provided to the reasoning engine,
    typically containing memories and metadata from the Context Injection Engine.
    
    Attributes:
        memories (List[Dict[str, Any]]): List of memory objects with id, content, and metadata.
        metadata (Dict[str, Any]): Additional context metadata (timestamps, user info, etc.).
    
    Example:
        >>> context = Reasoning_Context(
        ...     memories=[
        ...         {"id": "mem_1", "content": "User likes Python", "metadata": {}}
        ...     ],
        ...     metadata={"user_id": "123", "session_id": "abc"}
        ... )
    """
    memories: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of memory objects with id, content, and metadata"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context metadata"
    )


class Reasoning_Request(BaseModel):
    """
    Data model representing an incoming reasoning request.
    
    This model defines the structure of a reasoning request with a user query
    and optional context to be used during reasoning.
    
    Attributes:
        query (str): The user's query or question to be processed.
        context (Reasoning_Context): Optional injected context with memories and metadata.
    
    Example:
        >>> request = Reasoning_Request(
        ...     query="What programming languages do I like?",
        ...     context=Reasoning_Context(
        ...         memories=[{"id": "mem_1", "content": "User likes Python", "metadata": {}}]
        ...     )
        ... )
    """
    query: str = Field(
        ...,
        description="The user's query or question to be processed",
        min_length=1
    )
    context: Reasoning_Context = Field(
        default_factory=Reasoning_Context,
        description="Injected context with memories and metadata"
    )


class Reasoning_Result(BaseModel):
    """
    Data model representing the structured output from the reasoning engine.
    
    This model defines the structure of the reasoning result containing the answer,
    used memories, and optional confidence information.
    
    Attributes:
        answer (str): The generated answer or response text.
        used_memories (List[str]): List of memory IDs that were referenced in the answer.
        confidence (Optional[float]): Optional confidence score between 0.0 and 1.0.
    
    Example:
        >>> result = Reasoning_Result(
        ...     answer="You like Python and JavaScript.",
        ...     used_memories=["mem_1", "mem_2"],
        ...     confidence=0.95
        ... )
    """
    answer: str = Field(
        ...,
        description="The generated answer or response text"
    )
    used_memories: List[str] = Field(
        default_factory=list,
        description="List of memory IDs referenced in the answer"
    )
    confidence: Optional[float] = Field(
        None,
        description="Optional confidence score between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )


__all__ = ['Reasoning_Request', 'Reasoning_Context', 'Reasoning_Result']
