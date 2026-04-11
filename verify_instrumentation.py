#!/usr/bin/env python3
"""Quick verification script for observability instrumentation."""

import sys
from datetime import datetime, timezone
from unittest.mock import Mock

# Test 1: Context Injection Engine
print("=" * 60)
print("1. Testing Context Injection Engine Instrumentation")
print("=" * 60)
try:
    from luma.core.context_injection import inject_memories, InjectionConfig
    from luma.core.metrics_collector import MetricsCollector
    from luma.core.structured_logger import StructuredLogger
    
    metrics = MetricsCollector()
    logger = StructuredLogger()
    
    # Mock memory interface
    mock_memory = Mock()
    mock_memory.retrieve.return_value = {
        "memories": [],
        "total_count": 0,
        "query_metadata": {}
    }
    
    config = InjectionConfig(max_memories=10)
    context = inject_memories("test", mock_memory, config, metrics_collector=metrics, logger=logger)
    
    snapshot = metrics.get_snapshot()
    assert "context_injection_latency_ms" in snapshot["timers"], "Missing context_injection_latency_ms"
    assert "context_injection_count" in snapshot["counters"], "Missing context_injection_count"
    print("✓ Context Injection Engine: INSTRUMENTED")
except Exception as e:
    print(f"✗ Context Injection Engine: FAILED - {e}")
    sys.exit(1)

# Test 2: Reasoning Engine
print("\n" + "=" * 60)
print("2. Testing Reasoning Engine Instrumentation")
print("=" * 60)
try:
    from luma.core.reasoning import ReasoningEngine
    from luma.core.llm_interface import StubLLM
    
    metrics = MetricsCollector()
    logger = StructuredLogger()
    
    engine = ReasoningEngine(llm=StubLLM(), metrics_collector=metrics, logger=logger)
    result = engine.process_message("test message")
    
    snapshot = metrics.get_snapshot()
    assert "reasoning_latency_ms" in snapshot["timers"], "Missing reasoning_latency_ms"
    assert "reasoning_count" in snapshot["counters"], "Missing reasoning_count"
    print("✓ Reasoning Engine: INSTRUMENTED")
except Exception as e:
    print(f"✗ Reasoning Engine: FAILED - {e}")
    sys.exit(1)

# Test 3: Memory Write Engine
print("\n" + "=" * 60)
print("3. Testing Memory Write Engine Instrumentation")
print("=" * 60)
try:
    from luma.core.memory_write.memory_write_engine import MemoryWriteEngine
    from luma.core.memory_write.schemas import MemoryCandidate, ScoredMemory
    
    metrics = MetricsCollector()
    logger = StructuredLogger()
    
    # Mock components
    mock_extractor = Mock()
    mock_extractor.extract_candidates.return_value = []
    
    mock_scorer = Mock()
    mock_writer = Mock()
    
    engine = MemoryWriteEngine(
        extractor=mock_extractor,
        scorer=mock_scorer,
        writer=mock_writer,
        metrics_collector=metrics,
        logger=logger
    )
    
    result = engine.process("test query", "test response")
    
    snapshot = metrics.get_snapshot()
    assert "memory_write_latency_ms" in snapshot["timers"], "Missing memory_write_latency_ms"
    assert "memory_write_count" in snapshot["counters"], "Missing memory_write_count"
    print("✓ Memory Write Engine: INSTRUMENTED")
except Exception as e:
    print(f"✗ Memory Write Engine: FAILED - {e}")
    sys.exit(1)

# Test 4: Injection Engine
print("\n" + "=" * 60)
print("4. Testing Injection Engine Instrumentation")
print("=" * 60)
try:
    from luma.core.injection_engine import InjectionEngine, InjectionConfig
    
    metrics = MetricsCollector()
    logger = StructuredLogger()
    
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=False
    )
    
    engine = InjectionEngine(config, metrics_collector=metrics, logger=logger)
    result = engine.inject([])
    
    snapshot = metrics.get_snapshot()
    assert "injection_engine_latency_ms" in snapshot["timers"], "Missing injection_engine_latency_ms"
    assert "injection_engine_count" in snapshot["counters"], "Missing injection_engine_count"
    print("✓ Injection Engine: INSTRUMENTED")
except Exception as e:
    print(f"✗ Injection Engine: FAILED - {e}")
    sys.exit(1)

# Test 5: Observability Module Structure
print("\n" + "=" * 60)
print("5. Testing Observability Module Structure")
print("=" * 60)
try:
    from luma.observability import (
        MetricsCollector,
        StructuredLogger,
        ObservabilityService,
        TraceContext
    )
    print("✓ Observability Module: PROPERLY STRUCTURED")
except Exception as e:
    print(f"✗ Observability Module: FAILED - {e}")
    sys.exit(1)

# Test 6: Backward Compatibility
print("\n" + "=" * 60)
print("6. Testing Backward Compatibility")
print("=" * 60)
try:
    # Test all engines work without observability
    from luma.core.context_injection import inject_memories, InjectionConfig
    from luma.core.reasoning import ReasoningEngine
    from luma.core.memory_write.memory_write_engine import MemoryWriteEngine
    from luma.core.injection_engine import InjectionEngine, InjectionConfig as InjConfig
    
    # Context injection without observability
    mock_memory = Mock()
    mock_memory.retrieve.return_value = {"memories": [], "total_count": 0, "query_metadata": {}}
    config = InjectionConfig(max_memories=10)
    inject_memories("test", mock_memory, config)
    
    # Reasoning without observability
    engine = ReasoningEngine()
    engine.process_message("test")
    
    # Memory write without observability
    mock_extractor = Mock()
    mock_extractor.extract_candidates.return_value = []
    engine = MemoryWriteEngine(mock_extractor, Mock(), Mock())
    engine.process("test", "test")
    
    # Injection without observability
    config = InjConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=False
    )
    engine = InjectionEngine(config)
    engine.inject([])
    
    print("✓ Backward Compatibility: MAINTAINED")
except Exception as e:
    print(f"✗ Backward Compatibility: FAILED - {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL VERIFICATION CHECKS PASSED ✓")
print("=" * 60)
print("\nSummary:")
print("  ✓ Context Injection Engine instrumented")
print("  ✓ Reasoning Engine instrumented")
print("  ✓ Memory Write Engine instrumented")
print("  ✓ Injection Engine instrumented")
print("  ✓ Observability module properly structured")
print("  ✓ Backward compatibility maintained")
