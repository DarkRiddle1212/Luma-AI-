"""
Pydantic request and response schemas for the Luma API layer.
"""

from pydantic import BaseModel, field_validator
from typing import Any


class ChatRequest(BaseModel):
    user_id: str
    message: str

    @field_validator("user_id", "message")
    @classmethod
    def must_be_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field must be a non-empty, non-whitespace string")
        return v


class ChatResponse(BaseModel):
    response: str
    insight_moments: list
    personalization: dict


class InsightResponse(BaseModel):
    insights: list


class InsightMomentsResponse(BaseModel):
    insight_moments: list


class TeacherRequest(BaseModel):
    user_id: str
    topic: str

    @field_validator("user_id", "topic")
    @classmethod
    def must_be_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field must be a non-empty, non-whitespace string")
        return v


class TeacherResponse(BaseModel):
    session_id: str
    status: str
    lessons: list
    explanations: list
    exercises: list


class PersonalizationResponse(BaseModel):
    tone: str
    style: str
    focus: str
    reasons: dict


class ErrorResponse(BaseModel):
    error: str
    status: int
