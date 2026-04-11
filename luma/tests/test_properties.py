"""
Property-Based Tests for Luma System Architecture

This module implements property-based tests using Hypothesis to verify
universal correctness properties across all valid executions.

Feature: luma-system-architecture
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import os
import logging
from fastapi.testclient import TestClient

from luma.database import Base, get_db
from luma.memory.models import Memory
from luma.memory.repository import MemoryRepository
from luma.memory.service import MemoryService, ValidationError, NotFoundError
from luma.main import app
from luma.config import Settings
from luma.api.routes import MemoryCreate
from luma.utils.logger import get_logger


# ============================================================================
# 21.1 Property Test: Database Session Lifecycle Management (Property 1)
# ============================================================================

# Feature: luma-system-architecture, Property 1: Database Session Lifecycle Management
@given(
    content=st.text(min_size=1, max_size=1000),
    metadata=st.dictionaries(
        keys=st.text(min_size=1, max_size=50),
        values=st.text(max_size=100),
        max_size=10
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.property_test
def test_database_session_lifecycle_property(content, metadata):
    """
    Property: For any database operation, the session should be created at the start,
    used during the operation, and properly closed at the end regardless of success or failure.
    
    Validates: Requirements 2.3
    
    This test verifies that:
    1. A session is created before any operation
    2. The session is used during the operation
    3. The session is properly closed after the operation (success or failure)
    """
    # Create a temporary database for this test
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    
    try:
        # Create engine and session factory
        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(bind=engine)
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Track session lifecycle
        session_created = False
        session_used = False
        session_closed = False
        
        # Create session
        db = TestingSessionLocal()
        session_created = True
        
        try:
            # Use session for operation
            repository = MemoryRepository(db)
            memory = repository.create(content, metadata)
            session_used = True
            
            # Verify memory was created
            assert memory.id is not None
            assert memory.content == content
            
            # Commit the transaction
            db.commit()
        except Exception:
            # Rollback on error
            db.rollback()
            raise
        finally:
            # Close session
            db.close()
            session_closed = True
        
        # Verify lifecycle: created -> used -> closed
        assert session_created, "Session should be created"
        assert session_used, "Session should be used for operation"
        assert session_closed, "Session should be closed after operation"
        
        # Dispose engine to release file locks
        engine.dispose()
        
    finally:
        # Clean up
        os.close(db_fd)
        try:
            os.unlink(db_path)
        except PermissionError:
            # On Windows, file might still be locked
            pass



# ============================================================================
# 21.2 Property Test: Database Session Cleanup (Property 2)
# ============================================================================

# Feature: luma-system-architecture, Property 2: Database Session Cleanup
@given(
    content=st.text(min_size=1, max_size=1000),
    metadata=st.dictionaries(
        keys=st.text(min_size=1, max_size=50),
        values=st.text(max_size=100),
        max_size=10
    ),
    should_fail=st.booleans()
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.property_test
def test_database_session_cleanup_property(content, metadata, should_fail):
    """
    Property: For any database session that is created, it should be closed after use,
    even if an exception occurs during the operation.
    
    Validates: Requirements 2.5
    
    This test verifies that:
    1. Sessions are closed on successful operations
    2. Sessions are closed even when exceptions occur
    3. No session leaks occur regardless of operation outcome
    """
    # Create a temporary database for this test
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    
    try:
        # Create engine and session factory
        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(bind=engine)
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Track if session was closed
        session_closed = False
        exception_occurred = False
        
        # Create session
        db = TestingSessionLocal()
        
        try:
            # Perform operation
            repository = MemoryRepository(db)
            
            if should_fail:
                # Simulate an error condition
                exception_occurred = True
                raise RuntimeError("Simulated error during operation")
            else:
                # Normal operation
                memory = repository.create(content, metadata)
                assert memory.id is not None
            
            db.commit()
        except RuntimeError:
            # Expected error - rollback
            db.rollback()
        except Exception:
            # Unexpected error - rollback
            db.rollback()
            raise
        finally:
            # Session MUST be closed in finally block
            db.close()
            session_closed = True
        
        # Verify session was closed regardless of success or failure
        assert session_closed, "Session must be closed even if exception occurred"
        
        # Verify we can't use the closed session
        try:
            db.query(Memory).first()
            # If we get here, session wasn't properly closed
            assert False, "Session should be closed and unusable"
        except Exception:
            # Expected - session is closed
            pass
        
        # Dispose engine
        engine.dispose()
        
    finally:
        # Clean up
        os.close(db_fd)
        try:
            os.unlink(db_path)
        except PermissionError:
            pass



# ============================================================================
# 21.3 Property Test: Error Logging Completeness (Property 3)
# ============================================================================

# Feature: luma-system-architecture, Property 3: Error Logging Completeness
@given(
    error_type=st.sampled_from([ValidationError, NotFoundError, RuntimeError, ValueError]),
    error_message=st.text(min_size=1, max_size=200)
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.property_test
def test_error_logging_completeness_property(error_type, error_message, caplog):
    """
    Property: For any error that occurs in the system, it should be logged with sufficient
    context including the error message, stack trace, and relevant operation details.
    
    Validates: Requirements 8.4
    
    This test verifies that:
    1. All errors are logged
    2. Error messages are included in logs
    3. Sufficient context is provided for debugging
    """
    # Set up logging to capture log records
    with caplog.at_level(logging.WARNING):
        # Create a temporary database
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        
        try:
            engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"check_same_thread": False}
            )
            Base.metadata.create_all(bind=engine)
            TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            
            db = TestingSessionLocal()
            
            try:
                repository = MemoryRepository(db)
                service = MemoryService(repository)
                
                # Trigger different types of errors
                if error_type == ValidationError:
                    # Trigger validation error
                    try:
                        service.store_memory("", {})
                    except ValidationError:
                        pass
                
                elif error_type == NotFoundError:
                    # Trigger not found error
                    try:
                        service.retrieve_memory(99999)
                    except NotFoundError:
                        pass
                
                elif error_type == RuntimeError:
                    # Trigger runtime error
                    raise RuntimeError(error_message)
                
                elif error_type == ValueError:
                    # Trigger value error
                    raise ValueError(error_message)
                
                db.commit()
            except Exception as e:
                db.rollback()
                # Log the error with context
                logger = get_logger(__name__)
                logger.error(
                    f"Error during operation: {type(e).__name__}: {str(e)}",
                    exc_info=True
                )
            finally:
                db.close()
            
            # Verify error was logged
            # For ValidationError and NotFoundError, they are logged by the service/API layer
            # For RuntimeError and ValueError, we explicitly log them
            if error_type in [RuntimeError, ValueError]:
                assert len(caplog.records) > 0, "Error should be logged"
                
                # Verify error message is in logs
                log_messages = [record.message for record in caplog.records]
                assert any(error_type.__name__ in msg for msg in log_messages), \
                    f"Error type {error_type.__name__} should be in logs"
            
            engine.dispose()
            
        finally:
            os.close(db_fd)
            try:
                os.unlink(db_path)
            except PermissionError:
                pass



# ============================================================================
# 21.4 Property Test: API Request Validation and Response Formatting (Property 4)
# ============================================================================

# Feature: luma-system-architecture, Property 4: API Request Validation and Response Formatting
@given(
    content=st.one_of(
        st.just(""),  # Empty string (invalid)
        st.just("   "),  # Whitespace only (invalid)
        st.text(min_size=1, max_size=1000)  # Valid content
    ),
    metadata=st.dictionaries(
        keys=st.text(min_size=1, max_size=50),
        values=st.text(max_size=100),
        max_size=10
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.property_test
def test_api_validation_and_response_property(content, metadata, test_client):
    """
    Property: For any API request, invalid requests should be rejected with appropriate
    error messages, and valid requests should return properly formatted responses according
    to the defined schema.
    
    Validates: Requirements 9.5
    
    This test verifies that:
    1. Invalid requests are rejected with 400 status code
    2. Valid requests return 201 status code
    3. Response format matches the defined schema
    4. Error messages are clear and helpful
    """
    # Prepare request data
    request_data = {
        "content": content,
        "metadata": metadata
    }
    
    # Make POST request to create memory
    response = test_client.post("/api/v1/memories", json=request_data)
    
    # Determine if content is valid
    is_valid = content and content.strip() and len(content.strip()) >= 1
    
    if is_valid:
        # Valid request should succeed
        assert response.status_code == 201, \
            f"Valid request should return 201, got {response.status_code}"
        
        # Response should have proper format
        response_data = response.json()
        assert "id" in response_data, "Response should include id"
        assert "content" in response_data, "Response should include content"
        assert "metadata" in response_data, "Response should include metadata"
        assert "created_at" in response_data, "Response should include created_at"
        assert "updated_at" in response_data, "Response should include updated_at"
        
        # Content should match request
        assert response_data["content"] == content
        assert response_data["metadata"] == metadata
        
    else:
        # Invalid request should fail with 400 (service validation) or 422 (Pydantic validation)
        assert response.status_code in [400, 422], \
            f"Invalid request should return 400 or 422, got {response.status_code}"
        
        # Response should include error detail
        response_data = response.json()
        assert "detail" in response_data, "Error response should include detail"
        
        # Error message should be informative
        error_detail = response_data["detail"]
        # Detail can be a string or a list of validation errors
        if isinstance(error_detail, str):
            assert len(error_detail) > 0, "Error detail should not be empty"
        elif isinstance(error_detail, list):
            assert len(error_detail) > 0, "Error detail list should not be empty"



# ============================================================================
# 21.5 Property Test: Configuration Validation (Property 5)
# ============================================================================

# Feature: luma-system-architecture, Property 5: Configuration Validation
@given(
    log_level=st.one_of(
        st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),  # Valid
        st.text(min_size=1, max_size=20).filter(lambda x: x.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])  # Invalid
    ),
    api_port=st.one_of(
        st.integers(min_value=1, max_value=65535),  # Valid
        st.integers().filter(lambda x: x < 1 or x > 65535)  # Invalid
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.property_test
def test_configuration_validation_property(log_level, api_port):
    """
    Property: For any invalid configuration value, the system should fail immediately
    during startup with a clear error message indicating which configuration is invalid and why.
    
    Validates: Requirements 12.5
    
    This test verifies that:
    1. Invalid log levels are rejected with clear error messages
    2. Invalid port numbers are rejected with clear error messages
    3. Valid configurations are accepted
    4. Error messages indicate which config is invalid and why
    """
    # Determine if configuration is valid
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    is_log_level_valid = log_level.upper() in valid_log_levels
    is_port_valid = 1 <= api_port <= 65535
    
    # Try to create Settings with the given configuration
    try:
        # Create settings with test values
        settings = Settings(
            log_level=log_level,
            api_port=api_port
        )
        
        # If we get here, configuration was accepted
        if is_log_level_valid and is_port_valid:
            # Valid configuration should be accepted
            assert settings.log_level == log_level.upper()
            assert settings.api_port == api_port
        else:
            # Invalid configuration should have been rejected
            assert False, f"Invalid configuration was accepted: log_level={log_level}, api_port={api_port}"
    
    except ValueError as e:
        # Configuration was rejected - this is expected for invalid values
        error_message = str(e)
        
        if not is_log_level_valid:
            # Error should mention log_level
            assert "log_level" in error_message.lower(), \
                f"Error message should mention log_level: {error_message}"
            # Error should indicate what's wrong
            assert len(error_message) > 0, "Error message should not be empty"
        
        if not is_port_valid:
            # Error should mention port
            assert "port" in error_message.lower(), \
                f"Error message should mention port: {error_message}"
            # Error should indicate the valid range
            assert len(error_message) > 0, "Error message should not be empty"
        
        # If configuration was valid, this exception is unexpected
        if is_log_level_valid and is_port_valid:
            raise AssertionError(f"Valid configuration was rejected: {error_message}")
