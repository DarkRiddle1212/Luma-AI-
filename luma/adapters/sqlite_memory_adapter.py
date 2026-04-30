"""
SQLite Memory Adapter Module.

Delegates to MemoryRepository (injected via constructor) instead of the
legacy MemoryManager. Implements the MemoryInterface abstraction.
"""

import logging
import time
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from luma.core.memory_interface import (
    MemoryInterface,
    MemoryStorageError,
    MemoryRetrievalError,
    QueryParameters,
    RetrievalResult,
    MemoryEntry,
)

if TYPE_CHECKING:
    from luma.core.metrics_collector import MetricsCollector
    from luma.core.structured_logger import StructuredLogger

logger = logging.getLogger(__name__)


class SQLiteMemoryAdapter(MemoryInterface):
    """
    Adapter that delegates to MemoryRepository to implement MemoryInterface.

    Both ``memory_repository`` and ``database_manager`` are injected via the
    constructor; this class never instantiates them internally.

    Parameters
    ----------
    memory_repository:
        A ``MemoryRepository`` instance (used only as a type hint / fallback;
        the adapter creates a fresh repository per session via
        ``database_manager.get_session()``).
    database_manager:
        A ``DatabaseManager`` instance whose ``get_session()`` context manager
        is used to obtain a transactional session for each operation.
    """

    def __init__(self, memory_repository, database_manager) -> None:
        # memory_repository is kept for reference / future use but the adapter
        # always creates a fresh MemoryRepository inside get_session() so that
        # each operation runs in its own transaction.
        self._memory_repository = memory_repository
        self._db = database_manager

    # ------------------------------------------------------------------
    # MemoryInterface implementation
    # ------------------------------------------------------------------

    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Persist *content* to the memories table.

        Metadata field mapping
        ----------------------
        - ``user_id``    → ``MemoryRepository.create(user_id=...)``
        - ``namespace``  → ``MemoryRepository.create(namespace=...)``
        - ``category``   → fallback for ``namespace`` when ``namespace`` is absent
        - ``importance`` → ``MemoryRepository.create(importance_score=...)``
        - ``final_score``→ ``MemoryRepository.create(final_score=...)``

        Returns
        -------
        str
            String representation of the auto-generated record id.

        Raises
        ------
        MemoryStorageError
            If the underlying repository raises ``RepositoryError`` or any
            other exception.
        """
        from luma.storage.repositories.memory_repository import MemoryRepository
        from luma.storage import RepositoryError

        meta = metadata or {}
        user_id: str = meta.get("user_id", "default")
        namespace: Optional[str] = meta.get("namespace") or meta.get("category")
        importance_score: float = float(meta.get("importance", 0.0))
        final_score: float = float(meta.get("final_score", 0.0))

        try:
            with self._db.get_session() as session:
                repo = MemoryRepository(session)
                record = repo.create(
                    user_id=user_id,
                    namespace=namespace,
                    content=content,
                    importance_score=importance_score,
                    final_score=final_score,
                )
                return str(record.id)
        except RepositoryError as exc:
            logger.error("Failed to store memory: %s", exc)
            raise MemoryStorageError(f"Storage failed: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected error storing memory: %s", exc)
            raise MemoryStorageError(f"Storage failed: {exc}") from exc

    def retrieve(
        self,
        query: Optional[str] = None,
        params: Optional[QueryParameters] = None,
        limit: int = 10,
        metrics_collector: Optional["MetricsCollector"] = None,
        logger: Optional["StructuredLogger"] = None,
    ) -> RetrievalResult:
        """
        Retrieve memories from the memories table.

        Parameter mapping
        -----------------
        - ``params["category"]`` → ``namespace`` argument of
          ``MemoryRepository.get_by_user()``
        - ``params["limit"]`` or *limit* → ``limit`` argument
        - ``params["query"]`` is accepted but not forwarded to the repository
          (full-text search is not yet implemented at the repository layer).

        Returns
        -------
        RetrievalResult
            Structured result with ``memories``, ``total_count``, and
            ``query_metadata``.

        Raises
        ------
        MemoryRetrievalError
            If the underlying repository raises ``RepositoryError`` or any
            other exception.
        """
        from luma.storage.repositories.memory_repository import MemoryRepository
        from luma.storage import RepositoryError

        start_perf = time.perf_counter()

        # Resolve effective parameters
        effective_limit: int = limit
        user_id: str = "default"
        namespace: Optional[str] = None

        if params:
            effective_limit = params.get("limit", limit)  # type: ignore[assignment]
            user_id = params.get("user_id", "default")  # type: ignore[misc]
            namespace = params.get("category")  # type: ignore[assignment]

        try:
            with self._db.get_session() as session:
                repo = MemoryRepository(session)
                records = repo.get_by_user(
                    user_id=user_id,
                    namespace=namespace,
                    limit=effective_limit,
                )

            memories: List[MemoryEntry] = [
                {
                    "id": str(r.id),
                    "content": r.content,
                    "metadata": {
                        "user_id": r.user_id,
                        "namespace": r.namespace,
                        "importance_score": r.importance_score,
                        "final_score": r.final_score,
                    },
                    "timestamp": r.created_at.isoformat(),
                    "category": r.namespace or "",
                    "tags": [],
                }
                for r in records
            ]

            elapsed_ms = (time.perf_counter() - start_perf) * 1000

            filters_applied: Dict[str, Any] = {}
            if namespace is not None:
                filters_applied["category"] = namespace

            result: RetrievalResult = {
                "memories": memories,
                "total_count": len(memories),
                "query_metadata": {
                    "execution_time_ms": elapsed_ms,
                    "filters_applied": filters_applied,
                    "limit": effective_limit,
                    "has_more": False,
                },
            }

            if metrics_collector is not None:
                metrics_collector.record_duration("retrieval_latency_ms", elapsed_ms)
                metrics_collector.increment("retrieval_count")

            if logger is not None:
                logger.log("memory_retrieval", {
                    "total_count": len(memories),
                    "duration_ms": elapsed_ms,
                    "filters": filters_applied,
                })

            return result

        except RepositoryError as exc:
            module_logger = logging.getLogger(__name__)
            module_logger.error("Failed to retrieve memories: %s", exc)
            raise MemoryRetrievalError(f"Retrieval failed: {exc}") from exc
        except Exception as exc:
            module_logger = logging.getLogger(__name__)
            module_logger.error("Unexpected error retrieving memories: %s", exc)
            raise MemoryRetrievalError(f"Retrieval failed: {exc}") from exc
