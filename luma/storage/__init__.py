"""
luma/storage/__init__.py

Public API for the Luma Persistence & Storage Layer.

Exports:
- Exception hierarchy: StorageError, RepositoryError, StorageConfigurationError, MigrationError
- Domain dataclasses: MemoryRecord, InsightRecord, UserProfileRecord, LearningProgressRecord
- Infrastructure: DatabaseManager, StorageConfig, MigrationRunner
- Repositories: MemoryRepository, InsightRepository, PersonalizationRepository, TeacherRepository
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class StorageError(Exception):
    """Base class for all storage layer errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


class RepositoryError(StorageError):
    """Raised by any repository method when a database operation fails."""


class StorageConfigurationError(StorageError):
    """Raised during DatabaseManager initialisation for invalid configuration."""


class MigrationError(StorageError):
    """Raised when a migration fails to apply."""

    def __init__(
        self,
        version: int,
        message: str,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(f"Migration v{version:03d} failed: {message}", cause)
        self.version = version


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MemoryRecord:
    """Domain object returned by MemoryRepository. Never an ORM model instance."""

    id: int
    user_id: str
    namespace: Optional[str]
    content: str
    importance_score: float
    final_score: float
    created_at: datetime


@dataclass
class InsightRecord:
    """Domain object returned by InsightRepository. Never an ORM model instance."""

    id: int
    user_id: str
    message: str
    confidence: float
    evidence: Optional[dict]
    created_at: datetime


@dataclass
class UserProfileRecord:
    """Domain object returned by PersonalizationRepository. Never an ORM model instance."""

    user_id: str
    interests: list
    preferences: dict
    strengths: list
    updated_at: datetime


@dataclass
class LearningProgressRecord:
    """Domain object returned by TeacherRepository. Never an ORM model instance."""

    id: int
    user_id: str
    topic: str
    progress: float
    weak_areas: list
    last_updated: datetime


# ---------------------------------------------------------------------------
# Infrastructure re-exports (imported after domain types to avoid circularity)
# ---------------------------------------------------------------------------

from luma.storage.config import StorageConfig
from luma.storage.database import DatabaseManager
from luma.storage.migrations import MigrationRunner
from luma.storage.repositories.insight_repository import InsightRepository
from luma.storage.repositories.memory_repository import MemoryRepository
from luma.storage.repositories.personalization_repository import PersonalizationRepository
from luma.storage.repositories.teacher_repository import TeacherRepository

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Exceptions
    "StorageError",
    "RepositoryError",
    "StorageConfigurationError",
    "MigrationError",
    # Domain dataclasses
    "MemoryRecord",
    "InsightRecord",
    "UserProfileRecord",
    "LearningProgressRecord",
    # Infrastructure
    "StorageConfig",
    "DatabaseManager",
    "MigrationRunner",
    # Repositories
    "MemoryRepository",
    "InsightRepository",
    "PersonalizationRepository",
    "TeacherRepository",
]
