"""
Session Manager for Luma Memory System.

This module provides session management capabilities for tracking conversation
sessions and managing memory buffering during active sessions.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
import threading
import uuid
import logging
import time

from luma.core.memory_interface import MemoryInterface, MemoryStorageError
from luma.core.write_strategy import SessionConfig

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """
    Represents an active conversation session.
    
    A session tracks a logical grouping of related messages within a time-bounded
    conversation. It maintains metadata about the session lifecycle and buffers
    memories that haven't yet been persisted to long-term storage.
    
    Attributes:
        session_id: Unique identifier for the session (UUID)
        start_time: When the session was created
        last_activity_time: Timestamp of the last message in this session
        message_count: Number of messages processed in this session
        buffered_memories: In-memory buffer of memories not yet persisted
        metadata: Custom session metadata (e.g., user_id, device_id)
    """
    session_id: str
    start_time: datetime
    last_activity_time: datetime
    message_count: int = 0
    buffered_memories: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)



class Session_Manager:
    """
    Manages conversation sessions and memory buffering.
    
    The Session_Manager is responsible for:
    - Creating and tracking conversation sessions with unique session_ids
    - Maintaining in-memory buffers of session memories
    - Managing session lifecycle (creation, activity tracking, expiration)
    - Persisting buffered memories when sessions end
    - Supporting concurrent sessions with thread-safe state management
    - Running periodic cleanup for expired sessions
    
    Thread Safety:
        All public methods are thread-safe using an RLock for synchronization.
        Multiple threads can safely create sessions, buffer memories, and
        end sessions concurrently.
    """
    
    def __init__(
        self,
        config: SessionConfig,
        memory_interface: MemoryInterface
    ):
        """
        Initialize the Session_Manager.
        
        Args:
            config: Configuration for session management behavior
            memory_interface: Interface for persisting memories to long-term storage
        """
        self.config = config
        self.memory = memory_interface
        self.sessions: Dict[str, Session] = {}  # session_id -> Session
        self.lock = threading.RLock()  # Thread-safe session access
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_task()
    
    def create_session(self, metadata: Optional[Dict] = None) -> str:
        """
        Create a new session with a unique session_id.
        
        Generates a UUID for the session and initializes session state with
        current timestamp. The session is immediately active and ready to
        buffer memories.
        
        Args:
            metadata: Optional custom metadata for the session (e.g., user_id, device_id)
        
        Returns:
            The unique session_id (UUID string)
        
        Thread-safe: Yes
        """
        with self.lock:
            session_id = str(uuid.uuid4())
            now = datetime.now(UTC)
            session = Session(
                session_id=session_id,
                start_time=now,
                last_activity_time=now,
                metadata=metadata or {}
            )
            self.sessions[session_id] = session
            logger.info(
                f"Session created: session_id={session_id}, "
                f"metadata={metadata or {}}"
            )
            return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by its ID.
        
        Checks if the session exists and whether it has expired. If the session
        has expired, it is automatically cleaned up and None is returned.
        
        Args:
            session_id: The unique identifier of the session
        
        Returns:
            The Session object if found and not expired, None otherwise
        
        Thread-safe: Yes
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if session and self._is_expired(session):
                logger.info(f"Session {session_id} has expired, cleaning up")
                self._expire_session(session_id)
                return None
            return session
    
    def update_activity(self, session_id: str) -> None:
        """
        Update the last activity time for a session.
        
        Should be called whenever a message is processed in the session.
        Updates the last_activity_time to prevent premature expiration and
        increments the message_count.
        
        Args:
            session_id: The unique identifier of the session
        
        Thread-safe: Yes
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session.last_activity_time = datetime.now(UTC)
                session.message_count += 1
                logger.debug(f"Updated activity for session {session_id}, message count: {session.message_count}")

    def buffer_memory(
        self,
        session_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add a memory to the session buffer.

        Buffers the memory in the session's in-memory buffer rather than
        immediately persisting it. If the buffer exceeds max_buffer_size,
        the oldest memories are flushed to long-term storage.

        Args:
            session_id: The unique identifier of the session
            content: The memory content to buffer
            metadata: Metadata associated with the memory

        Raises:
            ValueError: If the session does not exist

        Thread-safe: Yes
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

            memory_entry = {
                "content": content,
                "metadata": metadata,
                "buffered_at": datetime.now(UTC).isoformat()
            }
            session.buffered_memories.append(memory_entry)
            
            buffer_size = len(session.buffered_memories)
            logger.debug(
                f"Memory buffered: session_id={session_id}, "
                f"buffer_size={buffer_size}, "
                f"content_length={len(content)}"
            )

            # Handle buffer overflow
            if buffer_size > self.config.max_buffer_size:
                logger.info(
                    f"Buffer overflow: session_id={session_id}, "
                    f"buffer_size={buffer_size}, "
                    f"max_buffer_size={self.config.max_buffer_size}, "
                    f"flushing oldest memories"
                )
                self._flush_oldest_memories(session_id)

    def get_session_memories(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all buffered memories for a session.

        Returns a copy of the buffered memories list to prevent external
        modification of the internal buffer.

        Args:
            session_id: The unique identifier of the session

        Returns:
            A list of buffered memory entries (copies), or empty list if
            session not found

        Thread-safe: Yes
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                return []
            return session.buffered_memories.copy()

    def _flush_oldest_memories(self, session_id: str) -> int:
        """
        Flush the oldest memories from the session buffer to long-term storage.

        When the buffer exceeds max_buffer_size, this method persists the
        oldest half of the buffered memories to make room for new ones.

        Args:
            session_id: The unique identifier of the session

        Returns:
            The count of successfully flushed memories

        Note:
            This method assumes the lock is already held by the caller.
        """
        session = self.sessions.get(session_id)
        if not session:
            return 0

        # Calculate how many to flush (half of buffer)
        flush_count = len(session.buffered_memories) // 2
        if flush_count == 0:
            return 0

        # Get the oldest memories to flush
        memories_to_flush = session.buffered_memories[:flush_count]
        flushed_count = 0

        for memory_entry in memories_to_flush:
            try:
                self.memory.store(
                    content=memory_entry["content"],
                    metadata=memory_entry["metadata"]
                )
                flushed_count += 1
            except MemoryStorageError as e:
                # Log error but continue flushing other memories
                logger.error(
                    f"Failed to flush memory from session {session_id}: {e}",
                    exc_info=True
                )

        # Remove flushed memories from buffer
        session.buffered_memories = session.buffered_memories[flush_count:]

        logger.info(
            f"Buffer flush completed: session_id={session_id}, "
            f"flushed_count={flushed_count}/{flush_count}, "
            f"remaining_buffer_size={len(session.buffered_memories)}"
        )

        return flushed_count

    
    def end_session(self, session_id: str, persist: bool = True) -> int:
        """
        End a session and optionally persist buffered memories.
        
        When a session ends normally, buffered memories are persisted to
        long-term storage (unless persist=False). The session is then removed
        from active tracking.
        
        Args:
            session_id: The unique identifier of the session to end
            persist: Whether to persist buffered memories (default: True)
        
        Returns:
            The count of successfully persisted memories
        
        Thread-safe: Yes
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                logger.warning(f"Attempted to end non-existent session {session_id}")
                return 0
            
            # Calculate session duration
            duration = datetime.now(UTC) - session.start_time
            duration_seconds = duration.total_seconds()
            
            persisted_count = 0
            if persist:
                persisted_count = self._persist_buffered_memories(session_id)
            
            del self.sessions[session_id]
            logger.info(
                f"Session ended: session_id={session_id}, "
                f"persisted_count={persisted_count}, "
                f"message_count={session.message_count}, "
                f"duration={duration_seconds:.2f}s"
            )
            return persisted_count
    
    def _persist_buffered_memories(self, session_id: str) -> int:
        """
        Persist all buffered memories for a session.
        
        Iterates through the session's buffered memories and stores each one
        via the MemoryInterface. Errors during persistence are logged but don't
        stop the process - we attempt to persist all memories.
        
        Args:
            session_id: The unique identifier of the session
        
        Returns:
            The count of successfully persisted memories
        
        Note:
            This method assumes the lock is already held by the caller.
        """
        session = self.sessions.get(session_id)
        if not session:
            return 0
        
        persisted_count = 0
        for memory_entry in session.buffered_memories:
            try:
                self.memory.store(
                    content=memory_entry["content"],
                    metadata=memory_entry["metadata"]
                )
                persisted_count += 1
            except MemoryStorageError as e:
                # Log error but continue persisting other memories
                logger.error(
                    f"Failed to persist memory from session {session_id}: {e}",
                    exc_info=True
                )
        
        session.buffered_memories.clear()
        return persisted_count
    
    def _is_expired(self, session: Session) -> bool:
        """
        Check if a session has expired based on the configured timeout.
        
        A session expires when the time since last_activity_time exceeds
        the configured timeout_seconds.
        
        Args:
            session: The session to check
        
        Returns:
            True if the session has expired, False otherwise
        
        Note:
            This method assumes the lock is already held by the caller.
        """
        timeout = timedelta(seconds=self.config.timeout_seconds)
        return datetime.now(UTC) - session.last_activity_time > timeout
    
    def _expire_session(self, session_id: str) -> None:
        """
        Expire a session and persist its buffered memories.
        
        Called when a session is detected as expired. Persists buffered
        memories before cleanup to ensure no data loss.
        
        Args:
            session_id: The unique identifier of the session to expire
        
        Note:
            This method assumes the lock is already held by the caller.
        """
        session = self.sessions.get(session_id)
        if not session:
            return
        
        # Calculate session duration
        duration = datetime.now(UTC) - session.start_time
        duration_seconds = duration.total_seconds()
        
        # Persist buffered memories
        persisted_count = self._persist_buffered_memories(session_id)
        
        # Remove session
        del self.sessions[session_id]
        
        logger.info(
            f"Session expired: session_id={session_id}, "
            f"reason=timeout, "
            f"persisted_count={persisted_count}, "
            f"message_count={session.message_count}, "
            f"duration={duration_seconds:.2f}s"
        )
    
    def _cleanup_expired_sessions(self) -> None:
        """
        Periodic task to clean up expired sessions.
        
        Iterates through all active sessions, identifies expired ones,
        and expires them (persisting their buffered memories).
        
        Thread-safe: Yes
        """
        with self.lock:
            expired_ids = [
                sid for sid, session in self.sessions.items()
                if self._is_expired(session)
            ]
            
            if expired_ids:
                logger.info(
                    f"Cleanup: found {len(expired_ids)} expired sessions to clean up"
                )
                
            for session_id in expired_ids:
                try:
                    self._expire_session(session_id)
                except Exception as e:
                    logger.error(
                        f"Cleanup: failed to expire session {session_id}: {e}",
                        exc_info=True
                    )
            
            if expired_ids:
                logger.info(
                    f"Cleanup: completed, cleaned up {len(expired_ids)} sessions"
                )
    
    def _start_cleanup_task(self) -> None:
        """
        Start a background thread for periodic session cleanup.
        
        The cleanup thread runs as a daemon and periodically calls
        _cleanup_expired_sessions() at the configured interval.
        """
        def cleanup_loop():
            logger.info(
                f"Cleanup task started: "
                f"interval={self.config.cleanup_interval_seconds}s, "
                f"timeout={self.config.timeout_seconds}s"
            )
            while not self._stop_cleanup.is_set():
                # Use wait() instead of sleep() so we can be interrupted immediately
                if self._stop_cleanup.wait(timeout=self.config.cleanup_interval_seconds):
                    # Event was set, exit loop
                    break
                # Timeout occurred, run cleanup
                try:
                    self._cleanup_expired_sessions()
                except Exception as e:
                    logger.error(
                        f"Cleanup task error: {e}",
                        exc_info=True
                    )
            
            logger.info("Cleanup task stopped")
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def shutdown(self) -> None:
        """
        Shutdown the Session_Manager and cleanup resources.
        
        Stops the cleanup thread and persists all active sessions.
        Should be called when the application is shutting down.
        """
        logger.info("Shutting down Session_Manager")
        self._stop_cleanup.set()
        
        # Persist all active sessions
        with self.lock:
            session_ids = list(self.sessions.keys())
            for session_id in session_ids:
                self.end_session(session_id, persist=True)
        
        # Wait for cleanup thread to finish
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
        
        logger.info("Session_Manager shutdown complete")
    def cancel_session(self, session_id: str) -> None:
        """
        Cancel a session and discard all buffered memories without persistence.

        This is a convenience method that calls end_session with persist=False.
        Use this when you want to abort a session and discard all buffered
        memories without saving them to long-term storage.

        Args:
            session_id: The unique identifier of the session to cancel

        Thread-safe: Yes
        """
        self.end_session(session_id, persist=False)
        logger.info(f"Cancelled session {session_id}, discarded buffered memories")


    def cancel_session(self, session_id: str) -> int:
        """
        Cancel a session and discard all buffered memories without persistence.

        This method is a convenience wrapper around end_session(persist=False)
        that explicitly indicates the session is being cancelled rather than
        ended normally.

        Args:
            session_id: The unique identifier of the session to cancel

        Returns:
            The count of memories that would have been persisted (always 0
            since cancellation discards without persistence)

        Thread-safe: Yes
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                logger.warning(f"Attempted to cancel non-existent session {session_id}")
                return 0
            
            discarded_count = len(session.buffered_memories)
            
            # End session without persisting
            self.end_session(session_id, persist=False)
            
            logger.info(
                f"Session cancelled: session_id={session_id}, "
                f"discarded_count={discarded_count}"
            )
            return 0
