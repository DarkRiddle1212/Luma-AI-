"""
Unit tests for dependency wiring functions.

Tests the initialize_application() and verify_dependencies() functions
to ensure proper component initialization and dependency verification.
"""

import pytest
import tempfile
import os
from pathlib import Path

from luma.container import initialize_application, verify_dependencies, cleanup_application
from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import LLMInterface, StubLLM
from luma.core.ranking_engine import RankingEngine
from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter


class TestInitializeApplication:
    """Test suite for initialize_application() function."""
    
    def test_initialize_with_defaults(self):
        """Test initialization with default parameters."""
        # Create temporary database path
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Initialize application
            engine, storage = initialize_application(db_path=db_path, return_storage=True)
            
            try:
                # Verify engine is created
                assert engine is not None
                assert isinstance(engine, ReasoningEngine)
                
                # Verify LLM is StubLLM (default)
                assert engine.llm is not None
                assert isinstance(engine.llm, StubLLM)
                
                # Verify memory adapter is configured
                assert engine.memory is not None
                assert isinstance(engine.memory, SQLiteMemoryAdapter)
            finally:
                # Cleanup connections
                cleanup_application(storage)
    
    def test_initialize_with_custom_llm(self):
        """Test initialization with custom LLM implementation."""
        # Create custom LLM
        class CustomLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                return "Custom response"
        
        custom_llm = CustomLLM()
        
        # Create temporary database path
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Initialize application with custom LLM
            engine, storage = initialize_application(db_path=db_path, llm=custom_llm, return_storage=True)
            
            try:
                # Verify custom LLM is used
                assert engine.llm is custom_llm
                assert isinstance(engine.llm, CustomLLM)
            finally:
                # Cleanup connections
                cleanup_application(storage)
    
    def test_initialize_creates_database_directory(self):
        """Test that initialization creates database directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use nested directory that doesn't exist
            db_path = os.path.join(tmpdir, "nested", "dir", "test_memory.db")
            
            # Initialize application
            engine, storage = initialize_application(db_path=db_path, return_storage=True)
            
            try:
                # Verify directory was created
                assert os.path.exists(os.path.dirname(db_path))
                assert engine is not None
            finally:
                # Cleanup connections
                cleanup_application(storage)
    
    def test_initialize_creates_all_components(self):
        """Test that all components are created during initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Initialize application
            engine, storage = initialize_application(db_path=db_path, return_storage=True)
            
            try:
                # Verify all components exist
                assert engine is not None
                assert engine.llm is not None
                assert engine.memory is not None
                
                # Verify memory adapter has memory_manager
                assert hasattr(engine.memory, 'memory_manager')
                assert engine.memory.memory_manager is not None
            finally:
                # Cleanup connections
                cleanup_application(storage)


class TestVerifyDependencies:
    """Test suite for verify_dependencies() function."""
    
    def test_verify_with_all_dependencies(self):
        """Test verification passes when all dependencies are present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Initialize application with all dependencies
            engine, storage = initialize_application(db_path=db_path, return_storage=True)
            
            try:
                # Verify dependencies - should not raise
                verify_dependencies(engine)
            finally:
                # Cleanup connections
                cleanup_application(storage)
    
    def test_verify_raises_error_when_llm_missing(self):
        """Test verification raises RuntimeError when LLM is missing."""
        # Create engine with missing LLM by manually setting it to None after construction
        # (ReasoningEngine constructor now defaults to StubLLM, so we need to override)
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=None)
        engine.llm = None  # Manually set to None to test the verification
        
        # Verify dependencies should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            verify_dependencies(engine)
        
        assert "LLM dependency not configured" in str(exc_info.value)
    
    def test_verify_warns_when_memory_missing(self, caplog):
        """Test verification logs warning when memory is missing (optional)."""
        # Create engine with LLM but no memory
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=None)
        
        # Verify dependencies - should not raise, but should log warning
        verify_dependencies(engine)
        
        # Check that warning was logged
        assert any("Memory dependency not configured" in record.message 
                  for record in caplog.records)
    
    def test_verify_logs_success_with_all_dependencies(self, caplog):
        """Test verification logs success when all dependencies present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Initialize application with all dependencies
            engine, storage = initialize_application(db_path=db_path, return_storage=True)
            
            try:
                # Verify dependencies
                verify_dependencies(engine)
                
                # Check that success was logged
                assert any("Dependency verification passed" in record.message 
                          for record in caplog.records)
            finally:
                # Cleanup connections
                cleanup_application(storage)


class TestIntegration:
    """Integration tests for dependency wiring."""
    
    def test_end_to_end_initialization_and_verification(self):
        """Test complete initialization and verification flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Initialize application
            engine, storage = initialize_application(db_path=db_path, return_storage=True)
            
            try:
                # Verify dependencies
                verify_dependencies(engine)
                
                # Test that engine can process messages
                result = engine.process_message("Hello, Luma!")
                
                assert result is not None
                assert "response" in result
                assert "intent" in result
                assert result["intent"] == "general"
            finally:
                # Cleanup connections
                cleanup_application(storage)
    
    def test_memory_operations_work_after_initialization(self):
        """Test that memory operations work after initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Initialize application
            engine, storage = initialize_application(db_path=db_path, return_storage=True)
            
            try:
                # Test store memory operation
                result = engine.process_message("Remember to buy milk")
                
                assert result is not None
                assert result["intent"] == "store_memory"
                assert "memory_id" in result["metadata"]
            finally:
                # Cleanup connections
                cleanup_application(storage)


class TestRankingEngineIntegration:
    """Test suite for RankingEngine integration with dependency injection."""
    
    def test_initialize_with_default_ranking_config(self):
        """Test initialization creates RankingEngine with default configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Initialize application with default ranking config
            engine, storage = initialize_application(db_path=db_path, return_storage=True)
            
            try:
                # Verify engine is created
                assert engine is not None
                assert isinstance(engine, ReasoningEngine)
            finally:
                cleanup_application(storage)
    
    def test_initialize_with_custom_ranking_config(self):
        """Test initialization with custom ranking configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Custom ranking configuration
            ranking_config = {
                "alpha": 0.7,
                "beta": 0.3,
                "gamma": 0.0,
                "decay_constant": 0.001,
                "similarity_threshold": 0.5,
                "score_threshold": 0.3
            }
            
            # Initialize application with custom ranking config
            engine, storage = initialize_application(
                db_path=db_path,
                ranking_config=ranking_config,
                return_storage=True
            )
            
            try:
                # Verify engine is created
                assert engine is not None
                assert isinstance(engine, ReasoningEngine)
            finally:
                cleanup_application(storage)
    
    def test_initialize_with_invalid_ranking_config_raises_error(self):
        """Test that invalid ranking configuration raises ValueError."""
        # Invalid ranking configuration (weights don't sum to 1)
        invalid_config = {
            "alpha": 0.5,
            "beta": 0.3,
            "gamma": 0.3,  # Sum = 1.1, invalid
            "decay_constant": 0.0001,
            "similarity_threshold": 0.3,
            "score_threshold": 0.2
        }
        
        # Should raise ValueError during initialization
        with pytest.raises(ValueError) as exc_info:
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = os.path.join(tmpdir, "test_memory.db")
                engine, storage = initialize_application(
                    db_path=db_path,
                    ranking_config=invalid_config,
                    return_storage=True
                )
        
        assert "weight sum" in str(exc_info.value).lower()
        assert "1.1" in str(exc_info.value)
    
    def test_ranking_config_from_settings(self):
        """Test loading ranking configuration from settings."""
        from luma.container import load_ranking_config_from_settings
        
        # Load config from settings
        config = load_ranking_config_from_settings()
        
        # Verify config has all required keys
        assert "alpha" in config
        assert "beta" in config
        assert "gamma" in config
        assert "decay_constant" in config
        assert "similarity_threshold" in config
        assert "score_threshold" in config
        
        # Verify default values
        assert config["alpha"] == 0.5
        assert config["beta"] == 0.3
        assert config["gamma"] == 0.2
        assert config["decay_constant"] == 0.0001
        assert config["similarity_threshold"] == 0.3
        assert config["score_threshold"] == 0.2
    
    def test_create_ranking_engine_directly(self):
        """Test creating RankingEngine directly using container function."""
        from luma.container import create_ranking_engine
        
        # Create with default config
        engine = create_ranking_engine()
        assert engine is not None
        assert isinstance(engine, RankingEngine)
        
        # Create with custom config
        engine2 = create_ranking_engine(
            alpha=0.7,
            beta=0.3,
            gamma=0.0,
            decay_constant=0.001
        )
        assert engine2 is not None
        assert isinstance(engine2, RankingEngine)
