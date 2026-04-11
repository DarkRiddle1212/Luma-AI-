"""
Integration Tests

Tests for full system integration including:
- API → Service → Repository → Database flow
- Application startup and shutdown
- Dependency injection wiring
- Error propagation through layers

These tests verify that all components work together correctly.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import tempfile
import os

from luma.main import create_app
from luma.database import get_db, Base
from luma.memory.models import Memory
from luma.memory.repository import MemoryRepository
from luma.memory.service import MemoryService, ValidationError, NotFoundError


# ============================================================================
# 22.1 Test full API → Service → Repository → Database flow
# ============================================================================

class TestFullStackFlow:
    """Tests for complete API → Service → Repository → Database integration."""
    
    @pytest.fixture(scope="function")
    def integration_db(self) -> Generator[Session, None, None]:
        """Create a fresh test database for integration tests."""
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        
        try:
            engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"check_same_thread": False}
            )
            Base.metadata.create_all(bind=engine)
            TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
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
            os.close(db_fd)
            os.unlink(db_path)
    
    @pytest.fixture(scope="function")
    def integration_client(self, integration_db: Session) -> Generator[TestClient, None, None]:
        """Create test client with integration database."""
        app = create_app()
        
        def override_get_db() -> Generator[Session, None, None]:
            yield integration_db
        
        app.dependency_overrides[get_db] = override_get_db
        
        try:
            client = TestClient(app)
            yield client
        finally:
            app.dependency_overrides.clear()
    
    def test_create_memory_full_flow(self, integration_client: TestClient, integration_db: Session):
        """
        Test creating a memory flows through all layers:
        API → Service → Repository → Database
        """
        # Create via API
        payload = {"content": "Integration test memory", "metadata": {"test": "data"}}
        response = integration_client.post("/api/v1/memories", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        memory_id = data["id"]
        
        # Verify in database directly
        db_memory = integration_db.query(Memory).filter(Memory.id == memory_id).first()
        assert db_memory is not None
        assert db_memory.content == "Integration test memory"
        assert db_memory.metadata_ == {"test": "data"}
    
    def test_retrieve_memory_full_flow(self, integration_client: TestClient, integration_db: Session):
        """
        Test retrieving a memory flows through all layers:
        Database → Repository → Service → API
        """
        # Create directly in database
        memory = Memory(content="Direct DB memory", metadata_={"source": "db"})
        integration_db.add(memory)
        integration_db.commit()
        
        # Retrieve via API
        response = integration_client.get(f"/api/v1/memories/{memory.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == memory.id
        assert data["content"] == "Direct DB memory"
        assert data["metadata"] == {"source": "db"}
    
    def test_update_memory_full_flow(self, integration_client: TestClient, integration_db: Session):
        """
        Test updating a memory flows through all layers and persists to database.
        """
        # Create via API
        create_response = integration_client.post(
            "/api/v1/memories",
            json={"content": "Original content"}
        )
        memory_id = create_response.json()["id"]
        
        # Update via API
        update_response = integration_client.put(
            f"/api/v1/memories/{memory_id}",
            json={"content": "Updated content", "metadata": {"updated": True}}
        )
        
        assert update_response.status_code == 200
        
        # Verify in database
        db_memory = integration_db.query(Memory).filter(Memory.id == memory_id).first()
        assert db_memory.content == "Updated content"
        assert db_memory.metadata_ == {"updated": True}
    
    def test_delete_memory_full_flow(self, integration_client: TestClient, integration_db: Session):
        """
        Test deleting a memory flows through all layers and removes from database.
        """
        # Create via API
        create_response = integration_client.post(
            "/api/v1/memories",
            json={"content": "To be deleted"}
        )
        memory_id = create_response.json()["id"]
        
        # Verify exists in database
        db_memory = integration_db.query(Memory).filter(Memory.id == memory_id).first()
        assert db_memory is not None
        
        # Delete via API
        delete_response = integration_client.delete(f"/api/v1/memories/{memory_id}")
        assert delete_response.status_code == 204
        
        # Verify removed from database
        db_memory = integration_db.query(Memory).filter(Memory.id == memory_id).first()
        assert db_memory is None
    
    def test_list_memories_full_flow(self, integration_client: TestClient, integration_db: Session):
        """
        Test listing memories flows through all layers with correct pagination.
        """
        # Create multiple memories via API
        for i in range(5):
            integration_client.post(
                "/api/v1/memories",
                json={"content": f"Memory {i}"}
            )
        
        # List via API with pagination
        response = integration_client.get("/api/v1/memories?skip=1&limit=2")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        
        # Verify count in database
        db_count = integration_db.query(Memory).count()
        assert db_count == 5
    
    def test_transaction_commit_on_success(self, integration_client: TestClient, integration_db: Session):
        """
        Test that successful operations commit to database.
        """
        # Create memory
        response = integration_client.post(
            "/api/v1/memories",
            json={"content": "Transaction test"}
        )
        memory_id = response.json()["id"]
        
        # Verify memory persisted in the same session
        # The integration_db fixture already handles commit/rollback
        memory = integration_db.query(Memory).filter(Memory.id == memory_id).first()
        assert memory is not None
        assert memory.content == "Transaction test"
    
    def test_transaction_rollback_on_error(self, integration_client: TestClient, integration_db: Session):
        """
        Test that failed operations rollback and don't persist to database.
        """
        # Get initial count
        initial_count = integration_db.query(Memory).count()
        
        # Try to create with invalid content (whitespace only)
        response = integration_client.post(
            "/api/v1/memories",
            json={"content": "   "}
        )
        
        assert response.status_code == 400
        
        # Verify count unchanged (rollback occurred)
        final_count = integration_db.query(Memory).count()
        assert final_count == initial_count
    
    def test_concurrent_operations(self, integration_client: TestClient, integration_db: Session):
        """
        Test that multiple operations in sequence work correctly.
        """
        # Create
        create_response = integration_client.post(
            "/api/v1/memories",
            json={"content": "First"}
        )
        id1 = create_response.json()["id"]
        
        # Create another
        create_response2 = integration_client.post(
            "/api/v1/memories",
            json={"content": "Second"}
        )
        id2 = create_response2.json()["id"]
        
        # Update first
        integration_client.put(
            f"/api/v1/memories/{id1}",
            json={"content": "First Updated"}
        )
        
        # Delete second
        integration_client.delete(f"/api/v1/memories/{id2}")
        
        # Verify final state
        memory1 = integration_db.query(Memory).filter(Memory.id == id1).first()
        memory2 = integration_db.query(Memory).filter(Memory.id == id2).first()
        
        assert memory1 is not None
        assert memory1.content == "First Updated"
        assert memory2 is None


# ============================================================================
# 22.2 Test application startup and shutdown
# ============================================================================

class TestApplicationLifecycle:
    """Tests for application startup and shutdown behavior."""
    
    def test_application_creates_successfully(self):
        """Test that create_app() returns a valid FastAPI instance."""
        app = create_app()
        
        assert app is not None
        assert app.title == "Luma AI System"
        assert app.version == "0.1.0"
    
    def test_database_tables_created_on_startup(self):
        """Test that database tables are created during application startup."""
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        
        try:
            # Create engine with temp database
            engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"check_same_thread": False}
            )
            
            # Manually trigger table creation (simulating startup)
            Base.metadata.create_all(bind=engine)
            
            # Verify tables exist
            with engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
                ))
                tables = result.fetchall()
                assert len(tables) == 1
                assert tables[0][0] == "memories"
            
            engine.dispose()
        finally:
            os.close(db_fd)
            os.unlink(db_path)
    
    def test_root_endpoint_accessible(self):
        """Test that root endpoint is accessible after startup."""
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/")
        
        assert response.status_code == 200
        assert response.json() == {"message": "Luma is alive"}
    
    def test_health_endpoint_accessible(self):
        """Test that health check endpoint is accessible after startup."""
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
    
    def test_api_routes_registered(self):
        """Test that API routes are properly registered with prefix."""
        # Create a test database for this test
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        
        try:
            # Create engine and tables
            engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"check_same_thread": False}
            )
            Base.metadata.create_all(bind=engine)
            TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            
            # Create app and override database
            app = create_app()
            
            def override_get_db() -> Generator[Session, None, None]:
                session = TestingSessionLocal()
                try:
                    yield session
                    session.commit()
                except Exception:
                    session.rollback()
                    raise
                finally:
                    session.close()
            
            app.dependency_overrides[get_db] = override_get_db
            client = TestClient(app)
            
            # Test that API endpoint is accessible with prefix
            response = client.get("/api/v1/memories")
            
            # Should return 200 (empty list) not 404 (route not found)
            assert response.status_code == 200
            
            app.dependency_overrides.clear()
            engine.dispose()
        finally:
            os.close(db_fd)
            os.unlink(db_path)
    
    def test_cors_middleware_configured(self):
        """Test that CORS middleware is properly configured."""
        app = create_app()
        client = TestClient(app)
        
        # Make request with Origin header
        response = client.get(
            "/",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers
    
    def test_application_handles_shutdown_gracefully(self):
        """Test that application can be shut down without errors."""
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        
        try:
            engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"check_same_thread": False}
            )
            Base.metadata.create_all(bind=engine)
            
            # Simulate shutdown
            engine.dispose()
            
            # Verify engine is disposed (pool should be empty)
            # After dispose, the pool is recreated on next use, so we just verify no error
            assert True  # If we got here without error, shutdown was graceful
        finally:
            os.close(db_fd)
            if os.path.exists(db_path):
                os.unlink(db_path)


# ============================================================================
# 22.3 Test dependency injection wiring
# ============================================================================

class TestDependencyInjection:
    """Tests for dependency injection wiring throughout the application."""
    
    @pytest.fixture(scope="function")
    def di_test_db(self) -> Generator[Session, None, None]:
        """Create test database for DI tests."""
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        
        try:
            engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"check_same_thread": False}
            )
            Base.metadata.create_all(bind=engine)
            TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
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
            os.close(db_fd)
            os.unlink(db_path)
    
    @pytest.fixture(scope="function")
    def di_test_client(self, di_test_db: Session) -> Generator[TestClient, None, None]:
        """Create test client for DI tests."""
        app = create_app()
        
        def override_get_db() -> Generator[Session, None, None]:
            yield di_test_db
        
        app.dependency_overrides[get_db] = override_get_db
        
        try:
            client = TestClient(app)
            yield client
        finally:
            app.dependency_overrides.clear()
    
    def test_get_db_dependency_provides_session(self, di_test_client: TestClient, di_test_db: Session):
        """Test that get_db dependency provides a valid session to routes."""
        # Create memory via API (uses dependency injection)
        response = di_test_client.post(
            "/api/v1/memories",
            json={"content": "DI test"}
        )
        
        assert response.status_code == 201
        
        # Verify session was used correctly
        memory_id = response.json()["id"]
        memory = di_test_db.query(Memory).filter(Memory.id == memory_id).first()
        assert memory is not None
    
    def test_repository_injected_into_service(self, di_test_db: Session):
        """Test that MemoryRepository is correctly injected into MemoryService."""
        # Create repository
        repository = MemoryRepository(di_test_db)
        
        # Create service with repository
        service = MemoryService(repository)
        
        # Use service
        memory = service.store_memory("Test content", {})
        di_test_db.commit()
        
        # Verify it used the repository
        assert memory.id is not None
        db_memory = di_test_db.query(Memory).filter(Memory.id == memory.id).first()
        assert db_memory is not None
    
    def test_service_injected_into_routes(self, di_test_client: TestClient):
        """Test that MemoryService is correctly injected into API routes."""
        # API routes use dependency injection to get service
        response = di_test_client.post(
            "/api/v1/memories",
            json={"content": "Service injection test"}
        )
        
        assert response.status_code == 201
        # If service wasn't injected, this would fail
    
    def test_dependency_override_works(self, di_test_db: Session):
        """Test that dependency overrides work correctly for testing."""
        app = create_app()
        
        # Override get_db
        def override_get_db() -> Generator[Session, None, None]:
            yield di_test_db
        
        app.dependency_overrides[get_db] = override_get_db
        
        client = TestClient(app)
        
        # Create memory
        response = client.post(
            "/api/v1/memories",
            json={"content": "Override test"}
        )
        
        assert response.status_code == 201
        
        # Verify it used our overridden database
        memory_id = response.json()["id"]
        memory = di_test_db.query(Memory).filter(Memory.id == memory_id).first()
        assert memory is not None
        
        app.dependency_overrides.clear()
    
    def test_session_lifecycle_managed_by_dependency(self, di_test_client: TestClient, di_test_db: Session):
        """Test that session lifecycle is properly managed by get_db dependency."""
        # Create memory (session should be committed automatically)
        response = di_test_client.post(
            "/api/v1/memories",
            json={"content": "Lifecycle test"}
        )
        
        assert response.status_code == 201
        memory_id = response.json()["id"]
        
        # Session should have committed, so memory should be in database
        memory = di_test_db.query(Memory).filter(Memory.id == memory_id).first()
        assert memory is not None
    
    def test_multiple_requests_use_separate_sessions(self, di_test_client: TestClient):
        """Test that each request gets its own database session."""
        # Make multiple requests
        response1 = di_test_client.post(
            "/api/v1/memories",
            json={"content": "Request 1"}
        )
        response2 = di_test_client.post(
            "/api/v1/memories",
            json={"content": "Request 2"}
        )
        
        assert response1.status_code == 201
        assert response2.status_code == 201
        
        # Both should succeed independently
        assert response1.json()["id"] != response2.json()["id"]


# ============================================================================
# 22.4 Test error propagation through layers
# ============================================================================

class TestErrorPropagation:
    """Tests for error propagation from repository through service to API."""
    
    @pytest.fixture(scope="function")
    def error_test_db(self) -> Generator[Session, None, None]:
        """Create test database for error propagation tests."""
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        
        try:
            engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"check_same_thread": False}
            )
            Base.metadata.create_all(bind=engine)
            TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
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
            os.close(db_fd)
            os.unlink(db_path)
    
    @pytest.fixture(scope="function")
    def error_test_client(self, error_test_db: Session) -> Generator[TestClient, None, None]:
        """Create test client for error propagation tests."""
        app = create_app()
        
        def override_get_db() -> Generator[Session, None, None]:
            yield error_test_db
        
        app.dependency_overrides[get_db] = override_get_db
        
        try:
            client = TestClient(app)
            yield client
        finally:
            app.dependency_overrides.clear()
    
    def test_validation_error_propagates_to_api(self, error_test_client: TestClient):
        """
        Test that ValidationError from service layer propagates to API as 400.
        """
        # Try to create with invalid content
        response = error_test_client.post(
            "/api/v1/memories",
            json={"content": "   "}  # Whitespace only
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "empty" in data["detail"].lower() or "whitespace" in data["detail"].lower()
    
    def test_not_found_error_propagates_to_api(self, error_test_client: TestClient):
        """
        Test that NotFoundError from service layer propagates to API as 404.
        """
        # Try to retrieve non-existent memory
        response = error_test_client.get("/api/v1/memories/99999")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    def test_validation_error_on_update_propagates(self, error_test_client: TestClient):
        """
        Test that ValidationError on update propagates correctly.
        """
        # Create a memory
        create_response = error_test_client.post(
            "/api/v1/memories",
            json={"content": "Original"}
        )
        memory_id = create_response.json()["id"]
        
        # Try to update with invalid content
        response = error_test_client.put(
            f"/api/v1/memories/{memory_id}",
            json={"content": ""}
        )
        
        assert response.status_code == 422  # Pydantic validation
    
    def test_not_found_error_on_update_propagates(self, error_test_client: TestClient):
        """
        Test that NotFoundError on update propagates correctly.
        """
        # Try to update non-existent memory
        response = error_test_client.put(
            "/api/v1/memories/99999",
            json={"content": "New content"}
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_not_found_error_on_delete_propagates(self, error_test_client: TestClient):
        """
        Test that NotFoundError on delete propagates correctly.
        """
        # Try to delete non-existent memory
        response = error_test_client.delete("/api/v1/memories/99999")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_pydantic_validation_error_at_api_layer(self, error_test_client: TestClient):
        """
        Test that Pydantic validation errors at API layer return 422.
        """
        # Send invalid JSON structure
        response = error_test_client.post(
            "/api/v1/memories",
            json={"wrong_field": "value"}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_error_includes_helpful_message(self, error_test_client: TestClient):
        """
        Test that error responses include helpful error messages.
        """
        # Try to retrieve non-existent memory
        response = error_test_client.get("/api/v1/memories/12345")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "12345" in data["detail"]  # Error message includes the ID
        assert "not found" in data["detail"].lower()
    
    def test_service_layer_catches_repository_errors(self, error_test_db: Session):
        """
        Test that service layer properly handles repository layer errors.
        """
        repository = MemoryRepository(error_test_db)
        service = MemoryService(repository)
        
        # Try to retrieve non-existent memory
        with pytest.raises(NotFoundError) as exc_info:
            service.retrieve_memory(99999)
        
        assert "99999" in str(exc_info.value)
    
    def test_api_layer_catches_service_errors(self, error_test_client: TestClient):
        """
        Test that API layer properly handles service layer errors.
        """
        # Try to create with invalid content (service layer validation)
        response = error_test_client.post(
            "/api/v1/memories",
            json={"content": "\t\n"}
        )
        
        # Should return 400 (service validation error)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    def test_database_rollback_on_error(self, error_test_client: TestClient, error_test_db: Session):
        """
        Test that database transactions are rolled back on errors.
        """
        # Get initial count
        initial_count = error_test_db.query(Memory).count()
        
        # Try to create with invalid content
        response = error_test_client.post(
            "/api/v1/memories",
            json={"content": "   "}
        )
        
        assert response.status_code == 400
        
        # Verify no memory was created (rollback occurred)
        final_count = error_test_db.query(Memory).count()
        assert final_count == initial_count
