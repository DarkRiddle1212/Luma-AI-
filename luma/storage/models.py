"""
luma/storage/models.py

SQLAlchemy ORM models for the Luma Persistence & Storage Layer.

All four domain entities are defined here and inherit from ``Base`` (defined in
``luma.storage.database``).  ORM models are never exposed outside
``luma/storage/``; repositories convert them to typed domain dataclasses before
returning values to callers.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from luma.storage.database import Base


def _utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# MemoryModel
# ---------------------------------------------------------------------------


class MemoryModel(Base):
    """ORM model for the ``memories`` table."""

    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    namespace: Mapped[str | None] = mapped_column(String, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    importance_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    final_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        index=True,
    )

    __table_args__ = (
        Index("ix_memories_user_id_namespace", "user_id", "namespace"),
    )


# ---------------------------------------------------------------------------
# InsightModel
# ---------------------------------------------------------------------------


class InsightModel(Base):
    """ORM model for the ``insights`` table."""

    __tablename__ = "insights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        index=True,
    )


# ---------------------------------------------------------------------------
# UserProfileModel
# ---------------------------------------------------------------------------


class UserProfileModel(Base):
    """ORM model for the ``user_profiles`` table."""

    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    interests: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    preferences: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    strengths: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )


# ---------------------------------------------------------------------------
# LearningProgressModel
# ---------------------------------------------------------------------------


class LearningProgressModel(Base):
    """ORM model for the ``learning_progress`` table."""

    __tablename__ = "learning_progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    topic: Mapped[str] = mapped_column(String, nullable=False)
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    weak_areas: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )

    __table_args__ = (
        UniqueConstraint("user_id", "topic", name="uq_learning_progress_user_topic"),
        Index("ix_learning_progress_user_id_topic", "user_id", "topic"),
    )
