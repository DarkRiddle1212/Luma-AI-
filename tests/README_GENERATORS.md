# Test Data Generators

This document describes the shared test data generators available in `test_data_generators.py` for property-based testing of the context injection engine.

## Overview

The `test_data_generators.py` module provides reusable Hypothesis strategies for generating test data. These generators create valid, random test data that meets the requirements for the injection engine components.

## Available Generators

### 1. `metadata_strategy(embedding_dim=None)`

Generates random metadata dictionaries with various field types.

**Parameters:**
- `embedding_dim` (Optional[int]): If specified, includes an embedding vector of this dimension in the metadata

**Returns:** Dictionary with metadata fields including:
- `token_count`: Integer between 10-200
- `embedding`: List of floats (if `embedding_dim` is specified)
- Optional custom fields: `source`, `priority`

**Example:**
```python
from hypothesis import given
from tests.test_data_generators import metadata_strategy

@given(metadata=metadata_strategy())
def test_with_metadata(metadata):
    assert 'token_count' in metadata
    assert 10 <= metadata['token_count'] <= 200

@given(metadata=metadata_strategy(embedding_dim=768))
def test_with_embeddings(metadata):
    assert 'embedding' in metadata
    assert len(metadata['embedding']) == 768
```

### 2. `ranked_memory_strategy(embedding_dim=None)`

Generates random RankedMemory objects with all required fields.

**Parameters:**
- `embedding_dim` (Optional[int]): If specified, includes embeddings in metadata for similarity testing

**Returns:** Mock RankedMemory object with:
- `memory_id`: Unique string identifier
- `content`: String with 5-50 words
- `timestamp`: Datetime with UTC timezone
- `namespace`: Optional string
- `category`: Optional string
- `similarity_score`, `importance_score`, `recency_score`, `final_score`: Floats in [0, 1]
- `metadata`: Dictionary (includes embeddings if `embedding_dim` specified)
- `memory_entry`: None

**Example:**
```python
from hypothesis import given
from tests.test_data_generators import ranked_memory_strategy

@given(memory=ranked_memory_strategy())
def test_with_memory(memory):
    assert 0.0 <= memory.final_score <= 1.0
    assert len(memory.content) > 0

@given(memory=ranked_memory_strategy(embedding_dim=768))
def test_with_embeddings(memory):
    assert 'embedding' in memory.metadata
    assert len(memory.metadata['embedding']) == 768
```

### 3. `memory_list_strategy(min_size=0, max_size=50, embedding_dim=None)`

Generates lists of memories with unique memory_ids.

**Parameters:**
- `min_size` (int): Minimum list size (default: 0)
- `max_size` (int): Maximum list size (default: 50)
- `embedding_dim` (Optional[int]): If specified, all memories include embeddings

**Returns:** List of mock RankedMemory objects with unique memory_ids

**Example:**
```python
from hypothesis import given
from tests.test_data_generators import memory_list_strategy

@given(memories=memory_list_strategy(min_size=5, max_size=20))
def test_with_memory_list(memories):
    assert 5 <= len(memories) <= 20
    memory_ids = [m.memory_id for m in memories]
    assert len(memory_ids) == len(set(memory_ids))  # All unique
```

### 4. `sorted_memory_list_strategy(min_size=0, max_size=50, embedding_dim=None)`

Generates lists of memories sorted by final_score in descending order.

**Parameters:**
- `min_size` (int): Minimum list size (default: 0)
- `max_size` (int): Maximum list size (default: 50)
- `embedding_dim` (Optional[int]): If specified, all memories include embeddings

**Returns:** List of mock RankedMemory objects sorted by final_score (highest first)

**Example:**
```python
from hypothesis import given
from tests.test_data_generators import sorted_memory_list_strategy

@given(memories=sorted_memory_list_strategy(min_size=5, max_size=20))
def test_with_sorted_memories(memories):
    # Verify sorted by final_score descending
    for i in range(len(memories) - 1):
        assert memories[i].final_score >= memories[i + 1].final_score
```

### 5. `injection_config_strategy()`

Generates random valid InjectionConfig objects that pass validation.

**Returns:** Mock InjectionConfig object with:
- `max_token_budget`: Integer between 100-10000
- `max_memory_count`: Integer between 1-100
- `redundancy_similarity_threshold`: Float in [0, 1]
- `enable_category_isolation`: Boolean
- `allowed_categories`: List of strings (if isolation enabled) or None
- `token_estimation_factor`: 1.3 (default)

**Example:**
```python
from hypothesis import given
from tests.test_data_generators import injection_config_strategy

@given(config=injection_config_strategy())
def test_with_config(config):
    assert 100 <= config.max_token_budget <= 10000
    if config.enable_category_isolation:
        assert config.allowed_categories is not None
        assert len(config.allowed_categories) > 0
```

## Usage Guidelines

### When to Use Shared Generators

Use the shared generators when:
- Writing new property-based tests for the injection engine
- You need standard RankedMemory or InjectionConfig objects
- You want consistent test data generation across tests

### When to Create Custom Generators

Create custom generators when:
- You need specialized data that doesn't fit the standard patterns
- You need specific constraints not provided by shared generators
- You're testing edge cases that require custom data generation

### Handling Large Embeddings

When using `embedding_dim=768` or other large dimensions, you may need to suppress Hypothesis health checks:

```python
from hypothesis import given, settings, HealthCheck

@given(memory=ranked_memory_strategy(embedding_dim=768))
@settings(suppress_health_check=[HealthCheck.large_base_example])
def test_with_large_embeddings(memory):
    assert len(memory.metadata['embedding']) == 768
```

## Testing the Generators

The generators themselves are validated in `test_generators_validation.py`. Run these tests to verify the generators work correctly:

```bash
python -m pytest tests/test_generators_validation.py -v
```

## Integration with Existing Tests

Existing property tests may have their own local generators. These can optionally be refactored to use the shared generators for consistency, but this is not required. The shared generators are primarily intended for new tests and to reduce duplication.
