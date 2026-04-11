"""
Integration tests for memory system integration with ranking engine.

Tests end-to-end retrieval with ranking, configuration loading from config files,
and dependency injection wiring.

**Validates: Requirements 1.6, 2.3, 4.1, 4.2**
"""

import pytest
import tempfile
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

from luma.container import (
    initialize_application,
    create_ranking_engine,
    load_ranking_config_from_settings,
    verify_dependencies,
    cleanup_application
)
from luma.core.ranking_engine import (
    RankingEngine,
    RankingConfig,
    RankedMemory,
    memory_entry_to_ranked_memory
)
from luma.core.llm_interface import StubLLM
from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus


class TestEndToEndRetrievalWithRanking:
    """Test end-to-end retrieval with ranking integration."""
    
    def test_memory_entry_to_ranked_memory_conversion(self):
        """Test conversion from MemoryEntry to RankedMemory."""
        # Create a MemoryEntry
        timestamp = datetime.now(timezone.utc)
        entry = MemoryEntry(
            id="mem_123",
            timestamp=timestamp,
            action="User searched for Python tutorials",
            context={"importance": 0.8, "category": "search"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device_1",
            sync_status=SyncStatus.SYNCED,
            tags=["search", "python"]
        )
        
        # Convert to RankedMemory
        ranked = memory_entry_to_ranked_memory(
            entry,
            similarity_score=0.85,
            namespace="conversation"
        )
        
        # Verify conversion
        assert ranked.memory_id == "mem_123"
        assert ranked.timestamp == timestamp
        assert ranked.content == "User searched for Python tutorials"
        assert ranked.namespace == "conversation"
        assert ranked.similarity_score == 0.85
        assert ranked.importance_score == 0.8  # From context
        assert ranked.recency_score == 0.0  # Not computed yet
        assert ranked.final_score == 0.0  # Not computed yet
        assert ranked.memory_entry == entry
    
    def test_memory_entry_without_importance(self):
        """Test conversion when importance is not in context."""
        entry = MemoryEntry(
            id="mem_456",
            timestamp=datetime.now(timezone.utc),
            action="User logged in",
            context={},  # No importance
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device_1",
            sync_status=SyncStatus.SYNCED,
            tags=["auth"]
        )
        
        ranked = memory_entry_to_ranked_memory(entry, similarity_score=0.75)
        
        # Should default to 0.0
        assert ranked.importance_score == 0.0
    
    def test_memory_entry_with_invalid_importance(self):
        """Test conversion with invalid importance values."""
        # Test with importance > 1.0
        entry1 = MemoryEntry(
            id="mem_789",
            timestamp=datetime.now(timezone.utc),
            action="Test action",
            context={"importance": 1.5},  # Invalid: > 1.0
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device_1",
            sync_status=SyncStatus.SYNCED,
            tags=[]
        )
        
        ranked1 = memory_entry_to_ranked_memory(entry1, similarity_score=0.5)
        assert ranked1.importance_score == 1.0  # Clamped to 1.0
        
        # Test with importance < 0.0
        entry2 = MemoryEntry(
            id="mem_790",
            timestamp=datetime.now(timezone.utc),
            action="Test action",
            context={"importance": -0.5},  # Invalid: < 0.0
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device_1",
            sync_status=SyncStatus.SYNCED,
            tags=[]
        )
        
        ranked2 = memory_entry_to_ranked_memory(entry2, similarity_score=0.5)
        assert ranked2.importance_score == 0.0  # Clamped to 0.0
    
    def test_end_to_end_retrieval_with_ranking(self):
        """Test complete end-to-end flow: storage -> retrieval -> ranking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Initialize application
            engine, storage = initialize_application(db_path=db_path, return_storage=True)
            
            try:
                # Store some memories
                base_time = datetime.now(timezone.utc)
                
                # Memory 1: High similarity, recent
                result1 = engine.process_message("Remember to buy milk")
                assert result1["intent"] == "store_memory"
                mem_id_1 = result1["metadata"]["memory_id"]
                
                # Memory 2: Medium similarity, older
                result2 = engine.process_message("Remember to call mom")
                assert result2["intent"] == "store_memory"
                mem_id_2 = result2["metadata"]["memory_id"]
                
                # Memory 3: Low similarity, recent
                result3 = engine.process_message("Remember to exercise")
                assert result3["intent"] == "store_memory"
                mem_id_3 = result3["metadata"]["memory_id"]
                
                # Retrieve memories from storage
                memory_manager = engine.memory.memory_manager
                all_memories = memory_manager.query_memories(limit=1000)
                
                # Verify memories were stored
                assert len(all_memories) >= 3
                
                # Convert to RankedMemory objects with simulated similarity scores
                ranked_memories = []
                for mem in all_memories:
                    if mem.id == mem_id_1:
                        similarity = 0.9  # High similarity
                    elif mem.id == mem_id_2:
                        similarity = 0.6  # Medium similarity
                    elif mem.id == mem_id_3:
                        similarity = 0.4  # Low similarity
                    else:
                        continue
                    
                    ranked = memory_entry_to_ranked_memory(
                        mem,
                        similarity_score=similarity,
                        namespace="conversation"
                    )
                    ranked_memories.append(ranked)
                
                # Create ranking engine and rank memories
                ranking_engine = create_ranking_engine(
                    alpha=0.7,  # Favor similarity
                    beta=0.3,   # Some recency
                    gamma=0.0,  # No importance
                    similarity_threshold=0.3,
                    score_threshold=0.2
                )
                
                ranked_results = ranking_engine.rank(ranked_memories, current_time=base_time)
                
                # Verify ranking worked
                assert len(ranked_results) == 3
                
                # Verify scores were computed
                for mem in ranked_results:
                    assert mem.recency_score > 0
                    assert mem.final_score > 0
                
                # Verify ordering (highest similarity should be first)
                assert ranked_results[0].memory_id == mem_id_1
                
            finally:
                cleanup_application(storage)
    
    def test_end_to_end_with_threshold_filtering(self):
        """Test end-to-end flow with threshold filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Initialize application
            engine, storage = initialize_application(db_path=db_path, return_storage=True)
            
            try:
                # Store memories
                result1 = engine.process_message("Remember important meeting")
                result2 = engine.process_message("Remember to water plants")
                result3 = engine.process_message("Remember to check email")
                
                # Retrieve memories
                memory_manager = engine.memory.memory_manager
                all_memories = memory_manager.query_memories(limit=1000)
                
                # Convert to RankedMemory with varying similarity scores
                ranked_memories = []
                for i, mem in enumerate(all_memories[:3]):
                    similarity = 0.9 - (i * 0.3)  # 0.9, 0.6, 0.3
                    ranked = memory_entry_to_ranked_memory(
                        mem,
                        similarity_score=similarity
                    )
                    ranked_memories.append(ranked)
                
                # Create ranking engine with high threshold
                ranking_engine = create_ranking_engine(
                    alpha=0.5,
                    beta=0.5,
                    gamma=0.0,
                    similarity_threshold=0.5,  # High threshold
                    score_threshold=0.3
                )
                
                ranked_results = ranking_engine.rank(ranked_memories)
                
                # Only memories with similarity >= 0.5 should pass
                assert len(ranked_results) == 2
                assert all(m.similarity_score >= 0.5 for m in ranked_results)
                
            finally:
                cleanup_application(storage)


class TestConfigurationLoading:
    """Test configuration loading from config files."""
    
    def test_load_ranking_config_from_settings(self):
        """Test loading ranking configuration from settings."""
        config = load_ranking_config_from_settings()
        
        # Verify all required keys are present
        assert "alpha" in config
        assert "beta" in config
        assert "gamma" in config
        assert "decay_constant" in config
        assert "similarity_threshold" in config
        assert "score_threshold" in config
        
        # Verify types
        assert isinstance(config["alpha"], float)
        assert isinstance(config["beta"], float)
        assert isinstance(config["gamma"], float)
        assert isinstance(config["decay_constant"], float)
        assert isinstance(config["similarity_threshold"], float)
        assert isinstance(config["score_threshold"], float)
        
        # Verify default values from config.py
        assert config["alpha"] == 0.5
        assert config["beta"] == 0.3
        assert config["gamma"] == 0.2
        assert config["decay_constant"] == 0.0001
        assert config["similarity_threshold"] == 0.3
        assert config["score_threshold"] == 0.2
    
    def test_create_ranking_engine_with_loaded_config(self):
        """Test creating ranking engine with loaded configuration."""
        config = load_ranking_config_from_settings()
        ranking_engine = create_ranking_engine(**config)
        
        # Verify engine was created successfully
        assert ranking_engine is not None
        assert isinstance(ranking_engine, RankingEngine)
        
        # Verify config was applied
        assert ranking_engine.config.alpha == config["alpha"]
        assert ranking_engine.config.beta == config["beta"]
        assert ranking_engine.config.gamma == config["gamma"]
        assert ranking_engine.config.decay_constant == config["decay_constant"]
        assert ranking_engine.config.similarity_threshold == config["similarity_threshold"]
        assert ranking_engine.config.score_threshold == config["score_threshold"]
    
    def test_initialize_application_with_config_from_settings(self):
        """Test initializing application with config loaded from settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Load config from settings
            ranking_config = load_ranking_config_from_settings()
            
            # Initialize application with loaded config
            engine, storage = initialize_application(
                db_path=db_path,
                ranking_config=ranking_config,
                return_storage=True
            )
            
            try:
                # Verify engine was created
                assert engine is not None
                
                # Verify dependencies
                verify_dependencies(engine)
                
            finally:
                cleanup_application(storage)
    
    def test_custom_config_overrides_defaults(self):
        """Test that custom configuration overrides default values."""
        custom_config = {
            "alpha": 0.7,
            "beta": 0.2,
            "gamma": 0.1,
            "decay_constant": 0.001,
            "similarity_threshold": 0.5,
            "score_threshold": 0.4
        }
        
        ranking_engine = create_ranking_engine(**custom_config)
        
        # Verify custom values were applied
        assert ranking_engine.config.alpha == 0.7
        assert ranking_engine.config.beta == 0.2
        assert ranking_engine.config.gamma == 0.1
        assert ranking_engine.config.decay_constant == 0.001
        assert ranking_engine.config.similarity_threshold == 0.5
        assert ranking_engine.config.score_threshold == 0.4
    
    def test_invalid_config_raises_error_on_load(self):
        """Test that invalid configuration raises error during initialization."""
        invalid_config = {
            "alpha": 0.5,
            "beta": 0.3,
            "gamma": 0.3,  # Sum = 1.1, invalid
            "decay_constant": 0.0001,
            "similarity_threshold": 0.3,
            "score_threshold": 0.2
        }
        
        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            create_ranking_engine(**invalid_config)
        
        assert "weights must sum" in str(exc_info.value).lower()


class TestDependencyInjectionWiring:
    """Test dependency injection wiring for ranking engine."""
    
    def test_ranking_engine_created_during_initialization(self):
        """Test that ranking engine is created during application initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Initialize application
            engine, storage = initialize_application(db_path=db_path, return_storage=True)
            
            try:
                # Verify engine has all dependencies
                assert engine is not None
                assert engine.llm is not None
                assert engine.memory is not None
                
                # Verify memory adapter is properly wired
                assert hasattr(engine.memory, 'memory_manager')
                assert engine.memory.memory_manager is not None
                
            finally:
                cleanup_application(storage)
    
    def test_ranking_engine_with_custom_config_in_initialization(self):
        """Test ranking engine with custom config during initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            custom_config = {
                "alpha": 0.6,
                "beta": 0.4,
                "gamma": 0.0,
                "decay_constant": 0.002,
                "similarity_threshold": 0.4,
                "score_threshold": 0.3
            }
            
            # Initialize with custom ranking config
            engine, storage = initialize_application(
                db_path=db_path,
                ranking_config=custom_config,
                return_storage=True
            )
            
            try:
                # Verify engine was created
                assert engine is not None
                
                # Verify all dependencies are wired
                verify_dependencies(engine)
                
            finally:
                cleanup_application(storage)
    
    def test_ranking_engine_can_be_created_independently(self):
        """Test that ranking engine can be created independently of full app."""
        # Create ranking engine directly
        ranking_engine = create_ranking_engine(
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            decay_constant=0.0001,
            similarity_threshold=0.3,
            score_threshold=0.2
        )
        
        # Verify it works independently
        assert ranking_engine is not None
        assert isinstance(ranking_engine, RankingEngine)
        
        # Test ranking with sample data
        base_time = datetime.now(timezone.utc)
        memories = [
            RankedMemory(
                memory_id="1",
                timestamp=base_time,
                content="test",
                namespace=None,
                similarity_score=0.8,
                importance_score=0.5,
                recency_score=0.0,
                final_score=0.0,
                memory_entry=None
            )
        ]
        
        result = ranking_engine.rank(memories, current_time=base_time)
        assert len(result) == 1
        assert result[0].final_score > 0
    
    def test_multiple_ranking_engines_can_coexist(self):
        """Test that multiple ranking engines with different configs can coexist."""
        # Create similarity-focused engine
        similarity_engine = create_ranking_engine(
            alpha=0.8,
            beta=0.2,
            gamma=0.0,
            decay_constant=0.0001,
            similarity_threshold=0.5,
            score_threshold=0.3
        )
        
        # Create recency-focused engine
        recency_engine = create_ranking_engine(
            alpha=0.2,
            beta=0.8,
            gamma=0.0,
            decay_constant=0.01,  # Faster decay
            similarity_threshold=0.3,
            score_threshold=0.2
        )
        
        # Verify both exist with different configs
        assert similarity_engine.config.alpha == 0.8
        assert recency_engine.config.alpha == 0.2
        assert similarity_engine.config.decay_constant == 0.0001
        assert recency_engine.config.decay_constant == 0.01
        
        # Test that they produce different rankings
        base_time = datetime.now(timezone.utc)
        memories = [
            RankedMemory(
                memory_id="old_high_sim",
                timestamp=base_time - timedelta(hours=10),
                content="test",
                namespace=None,
                similarity_score=0.9,
                importance_score=0.0,
                recency_score=0.0,
                final_score=0.0,
                memory_entry=None
            ),
            RankedMemory(
                memory_id="new_low_sim",
                timestamp=base_time,
                content="test",
                namespace=None,
                similarity_score=0.5,
                importance_score=0.0,
                recency_score=0.0,
                final_score=0.0,
                memory_entry=None
            )
        ]
        
        sim_result = similarity_engine.rank(memories.copy(), current_time=base_time)
        rec_result = recency_engine.rank(memories.copy(), current_time=base_time)
        
        # Similarity engine should rank old_high_sim first
        assert sim_result[0].memory_id == "old_high_sim"
        
        # Recency engine should rank new_low_sim first
        assert rec_result[0].memory_id == "new_low_sim"
    
    def test_dependency_verification_passes_with_ranking_engine(self):
        """Test that dependency verification passes when ranking engine is configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Initialize application
            engine, storage = initialize_application(db_path=db_path, return_storage=True)
            
            try:
                # Verify dependencies - should not raise
                verify_dependencies(engine)
                
            finally:
                cleanup_application(storage)


class TestIntegrationEdgeCases:
    """Test edge cases in memory system integration."""
    
    def test_empty_memory_store_with_ranking(self):
        """Test ranking with empty memory store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Initialize application
            engine, storage = initialize_application(db_path=db_path, return_storage=True)
            
            try:
                # Create ranking engine
                ranking_engine = create_ranking_engine()
                
                # Try to rank empty list
                result = ranking_engine.rank([])
                
                # Should return empty list
                assert len(result) == 0
                
            finally:
                cleanup_application(storage)
    
    def test_all_memories_filtered_by_thresholds(self):
        """Test when all memories are filtered out by thresholds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Initialize application
            engine, storage = initialize_application(db_path=db_path, return_storage=True)
            
            try:
                # Store memories
                engine.process_message("Remember task 1")
                engine.process_message("Remember task 2")
                
                # Retrieve memories
                memory_manager = engine.memory.memory_manager
                all_memories = memory_manager.query_memories(limit=1000)
                
                # Convert to RankedMemory with low similarity scores
                ranked_memories = []
                for mem in all_memories[:2]:
                    ranked = memory_entry_to_ranked_memory(
                        mem,
                        similarity_score=0.2  # Low similarity
                    )
                    ranked_memories.append(ranked)
                
                # Create ranking engine with high threshold
                ranking_engine = create_ranking_engine(
                    alpha=0.5,
                    beta=0.5,
                    gamma=0.0,
                    similarity_threshold=0.8,  # Very high threshold
                    score_threshold=0.5
                )
                
                result = ranking_engine.rank(ranked_memories)
                
                # All should be filtered out
                assert len(result) == 0
                
            finally:
                cleanup_application(storage)
    
    def test_namespace_filtering_with_memory_store(self):
        """Test namespace filtering integration with memory store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            
            # Initialize application
            engine, storage = initialize_application(db_path=db_path, return_storage=True)
            
            try:
                # Store memories
                engine.process_message("Remember conversation point")
                engine.process_message("Remember system setting")
                
                # Retrieve memories
                memory_manager = engine.memory.memory_manager
                all_memories = memory_manager.query_memories(limit=1000)
                
                # Convert to RankedMemory with different namespaces
                ranked_memories = []
                for i, mem in enumerate(all_memories[:2]):
                    namespace = "conversation" if i == 0 else "system"
                    ranked = memory_entry_to_ranked_memory(
                        mem,
                        similarity_score=0.8,
                        namespace=namespace
                    )
                    ranked_memories.append(ranked)
                
                # Create ranking engine with namespace filter
                ranking_engine = create_ranking_engine(
                    alpha=0.5,
                    beta=0.5,
                    gamma=0.0,
                    similarity_threshold=0.0,
                    score_threshold=0.0,
                    namespace="conversation"
                )
                
                result = ranking_engine.rank(ranked_memories)
                
                # Only conversation namespace should be included
                assert len(result) == 1
                assert result[0].namespace == "conversation"
                
            finally:
                cleanup_application(storage)
    
    def test_ranking_with_future_timestamps(self):
        """Test ranking handles future timestamps correctly."""
        # Create memories with future timestamps
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        current_time = datetime.now(timezone.utc)
        
        memories = [
            RankedMemory(
                memory_id="future",
                timestamp=future_time,
                content="test",
                namespace=None,
                similarity_score=0.8,
                importance_score=0.0,
                recency_score=0.0,
                final_score=0.0,
                memory_entry=None
            ),
            RankedMemory(
                memory_id="present",
                timestamp=current_time,
                content="test",
                namespace=None,
                similarity_score=0.8,
                importance_score=0.0,
                recency_score=0.0,
                final_score=0.0,
                memory_entry=None
            )
        ]
        
        ranking_engine = create_ranking_engine(
            alpha=0.0,
            beta=1.0,  # Only recency
            gamma=0.0,
            decay_constant=0.001,
            similarity_threshold=0.0,
            score_threshold=0.0
        )
        
        result = ranking_engine.rank(memories, current_time=current_time)
        
        # Future timestamp should have recency_score = 1.0
        future_mem = next(m for m in result if m.memory_id == "future")
        assert future_mem.recency_score == 1.0
    
