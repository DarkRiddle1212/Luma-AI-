"""
Structural smoke tests for the luma.storage module.

Validates:
- All required files exist and are importable (Requirements 1.1, 1.2)
- luma/storage/__init__.py exports all expected names (Requirements 1.3, 1.4)
- No luma.core module imports from luma.storage (import boundary) (Requirements 1.3)
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_FILES = [
    "luma/storage/__init__.py",
    "luma/storage/database.py",
    "luma/storage/models.py",
    "luma/storage/config.py",
    "luma/storage/repositories/__init__.py",
    "luma/storage/repositories/memory_repository.py",
    "luma/storage/repositories/insight_repository.py",
    "luma/storage/repositories/personalization_repository.py",
    "luma/storage/repositories/teacher_repository.py",
    "luma/storage/migrations/__init__.py",
    "luma/storage/migrations/v001_initial_schema.py",
]

EXPECTED_EXPORTS = [
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
    "DatabaseManager",
    "StorageConfig",
    "MigrationRunner",
    # Repositories
    "MemoryRepository",
    "InsightRepository",
    "PersonalizationRepository",
    "TeacherRepository",
]

IMPORT_BOUNDARY_FILES = [
    "luma/core/insight/insight_engine.py",
    "luma/core/personalization/personalization_engine.py",
    "luma/core/teacher/teacher_mode.py",
]

WORKSPACE_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_imports(source: str) -> list[str]:
    """Return all module names imported in *source* (top-level and from-imports)."""
    tree = ast.parse(source)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    return modules


# ---------------------------------------------------------------------------
# Tests: required files exist
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel_path", REQUIRED_FILES)
def test_required_file_exists(rel_path: str) -> None:
    """Each required storage file must exist on disk."""
    path = WORKSPACE_ROOT / rel_path
    assert path.exists(), f"Required file not found: {rel_path}"
    assert path.is_file(), f"Path is not a file: {rel_path}"


# ---------------------------------------------------------------------------
# Tests: required modules are importable
# ---------------------------------------------------------------------------

IMPORTABLE_MODULES = [
    ("luma/storage/__init__.py", "luma.storage"),
    ("luma/storage/database.py", "luma.storage.database"),
    ("luma/storage/models.py", "luma.storage.models"),
    ("luma/storage/config.py", "luma.storage.config"),
    ("luma/storage/repositories/__init__.py", "luma.storage.repositories"),
    ("luma/storage/repositories/memory_repository.py", "luma.storage.repositories.memory_repository"),
    ("luma/storage/repositories/insight_repository.py", "luma.storage.repositories.insight_repository"),
    ("luma/storage/repositories/personalization_repository.py", "luma.storage.repositories.personalization_repository"),
    ("luma/storage/repositories/teacher_repository.py", "luma.storage.repositories.teacher_repository"),
    ("luma/storage/migrations/__init__.py", "luma.storage.migrations"),
    ("luma/storage/migrations/v001_initial_schema.py", "luma.storage.migrations.v001_initial_schema"),
]


@pytest.mark.parametrize("rel_path,module_name", IMPORTABLE_MODULES)
def test_module_is_importable(rel_path: str, module_name: str) -> None:
    """Each required storage module must be importable without errors."""
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(f"Cannot import {module_name} ({rel_path}): {exc}")
    assert mod is not None


# ---------------------------------------------------------------------------
# Tests: __init__.py exports all expected names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", EXPECTED_EXPORTS)
def test_storage_init_exports(name: str) -> None:
    """luma.storage must export every expected public name."""
    import luma.storage as storage_pkg
    assert hasattr(storage_pkg, name), (
        f"luma.storage does not export '{name}'"
    )


def test_storage_all_contains_expected_exports() -> None:
    """luma.storage.__all__ must list every expected export."""
    import luma.storage as storage_pkg
    all_exports = getattr(storage_pkg, "__all__", [])
    missing = [name for name in EXPECTED_EXPORTS if name not in all_exports]
    assert not missing, (
        f"luma.storage.__all__ is missing: {missing}"
    )


# ---------------------------------------------------------------------------
# Tests: import boundary — luma.core must NOT import from luma.storage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel_path", IMPORT_BOUNDARY_FILES)
def test_core_does_not_import_storage(rel_path: str) -> None:
    """luma.core modules must not directly import from luma.storage."""
    path = WORKSPACE_ROOT / rel_path
    assert path.exists(), f"File not found for boundary check: {rel_path}"
    source = path.read_text(encoding="utf-8")
    imports = _get_imports(source)
    violations = [m for m in imports if m.startswith("luma.storage")]
    assert not violations, (
        f"{rel_path} imports from luma.storage (boundary violation): {violations}"
    )
