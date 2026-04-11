"""
Pytest Configuration and Shared Fixtures

This module provides shared test fixtures for the Luma test suite.
Includes database, client, and sample data fixtures.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from luma.main import app
from luma.database import get_db, Base
from luma.memory.models import Memory  # Import to register with Base
from luma.memory.repository import MemoryRepository
from luma.memory.service import MemoryService


@pytest.fixture(scope="function")
def test_db() -> Generator[Session, None, None]:
    """
    Provide a test SQLite database for testing.
    
    Creates a fresh database for each test function to ensure isolation.
    Tables are created before the test and dropped after.
    
    Yields:
        Session: SQLAlchemy session connected to test database
    """
    import tempfile
    import os
    
    # Create a temporary database file
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    
    try:
        # Create engine with the temporary database
        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False}
        )
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        # Create session factory
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create session
        session = TestingSessionLocal()
        
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
            engine.dispose()
    finally:
        # Clean up the temporary database file
        os.close(db_fd)
        os.unlink(db_path)


@pytest.fixture(scope="function")
def test_client(test_db: Session) -> Generator[TestClient, None, None]:
    """
    Provide a FastAPI test client with test database.
    
    Overrides the get_db dependency to use the test database
    instead of the production database.
    
    Args:
        test_db: Test database session fixture
    
    Returns:
        TestClient: FastAPI test client configured for testing
    """
    def override_get_db() -> Generator[Session, None, None]:
        yield test_db
    
    app.dependency_overrides[get_db] = override_get_db
    
    try:
        client = TestClient(app)
        yield client
    finally:
        # Clean up dependency override
        app.dependency_overrides.clear()


@pytest.fixture
def sample_memory() -> dict:
    """
    Provide sample memory data for testing.
    
    Returns:
        dict: Sample memory data with content and metadata
    """
    return {
        "content": "Test memory content",
        "metadata": {"source": "test", "priority": "high"}
    }


@pytest.fixture
def memory_repository(test_db: Session) -> MemoryRepository:
    """
    Provide a MemoryRepository instance with test database.
    
    Args:
        test_db: Test database session fixture
    
    Returns:
        MemoryRepository: Repository instance for testing
    """
    return MemoryRepository(test_db)


@pytest.fixture
def memory_service(memory_repository: MemoryRepository) -> MemoryService:
    """
    Provide a MemoryService instance with test repository.
    
    Args:
        memory_repository: Repository fixture
    
    Returns:
        MemoryService: Service instance for testing
    """
    return MemoryService(memory_repository)
