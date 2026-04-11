# Contributing to Luma Memory Module

Thank you for your interest in contributing to the Luma Memory Module! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We expect all participants to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, trolling, or discriminatory comments
- Personal attacks or insults
- Publishing others' private information without permission
- Other conduct that could reasonably be considered inappropriate

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.8 or higher installed
- Git installed and configured
- A GitHub account
- Familiarity with Python development practices

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/luma-memory.git
   cd luma-memory
   ```

3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/luma/luma-memory.git
   ```

4. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

6. **Verify the setup**:
   ```bash
   pytest
   ```

## Development Workflow

### Creating a Feature Branch

1. **Sync with upstream**:
   ```bash
   git checkout main
   git pull upstream main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

   Branch naming conventions:
   - `feature/` - New features
   - `bugfix/` - Bug fixes
   - `docs/` - Documentation updates
   - `refactor/` - Code refactoring
   - `test/` - Test additions or improvements

### Making Changes

1. **Make your changes** in small, logical commits
2. **Write or update tests** for your changes
3. **Run tests locally** to ensure everything passes:
   ```bash
   pytest
   ```

4. **Check code style**:
   ```bash
   flake8 luma_memory tests
   black --check luma_memory tests
   mypy luma_memory
   ```

5. **Fix formatting issues**:
   ```bash
   black luma_memory tests
   ```

### Keeping Your Branch Updated

Regularly sync your branch with upstream:

```bash
git fetch upstream
git rebase upstream/main
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: Maximum 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Use double quotes for strings
- **Imports**: Group and sort imports (use `isort`)

### Code Formatting

We use [Black](https://black.readthedocs.io/) for code formatting:

```bash
black luma_memory tests
```

### Type Hints

All functions must include type hints:

```python
def create_memory(
    self,
    action: str,
    context: Dict[str, Any],
    device_id: str,
    sensitivity: str = "public",
    tags: Optional[List[str]] = None
) -> str:
    """Create a new memory entry."""
    pass
```

### Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def query_memories(
    self,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    tags: Optional[List[str]] = None,
    limit: int = 100
) -> List[MemoryEntry]:
    """Query memory entries with optional filters.
    
    Args:
        start_time: Filter entries after this timestamp
        end_time: Filter entries before this timestamp
        tags: Filter entries containing these tags
        limit: Maximum number of entries to return
        
    Returns:
        List of matching memory entries
        
    Raises:
        ValueError: If limit is negative or zero
        StorageError: If database query fails
    """
    pass
```

### Error Handling

- Use specific exception types
- Provide descriptive error messages
- Log errors with appropriate context
- Clean up resources in `finally` blocks

```python
try:
    entry = self.storage.get_entry(entry_id)
    if not entry:
        raise ValueError(f"Entry not found: {entry_id}")
except StorageError as e:
    logger.error(f"Failed to retrieve entry {entry_id}: {e}")
    raise
finally:
    # Clean up resources
    pass
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `MemoryManager`)
- **Functions/Methods**: `snake_case` (e.g., `create_memory`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_CACHE_SIZE`)
- **Private methods**: Prefix with `_` (e.g., `_validate_entry`)

## Testing Guidelines

### Test Requirements

- All new features must include tests
- Bug fixes must include regression tests
- Maintain test coverage above 80%
- Tests must be deterministic and isolated

### Writing Tests

Use `pytest` for all tests:

```python
import pytest
from luma_memory import MemoryManager

def test_create_memory_success():
    """Test successful memory creation."""
    manager = MemoryManager()
    
    entry_id = manager.create_memory(
        action="test_action",
        context={"key": "value"},
        device_id="test-device"
    )
    
    assert entry_id is not None
    assert isinstance(entry_id, str)

def test_create_memory_invalid_data():
    """Test memory creation with invalid data."""
    manager = MemoryManager()
    
    with pytest.raises(ValueError):
        manager.create_memory(
            action="",  # Invalid: empty action
            context={},
            device_id="test-device"
        )
```

### Test Organization

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Group related tests in classes
- Use fixtures for common setup

### Running Tests

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_memory_manager.py
```

Run with coverage:
```bash
pytest --cov=luma_memory --cov-report=html
```

Run specific test:
```bash
pytest tests/test_memory_manager.py::test_create_memory_success
```

### Test Coverage

Check coverage report:
```bash
pytest --cov=luma_memory --cov-report=term-missing
```

View HTML coverage report:
```bash
pytest --cov=luma_memory --cov-report=html
open htmlcov/index.html
```

## Commit Guidelines

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Commit Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(storage): add connection pooling for SQLite backend

Implement connection pooling to improve concurrent access
performance and prevent database lock errors.

Closes #123
```

```
fix(encryption): handle key rotation edge case

Fix issue where key rotation failed when no entries existed.
Add regression test to prevent future occurrences.

Fixes #456
```

```
docs(api): update query endpoint examples

Add examples for filtering by tags and time range.
Clarify pagination behavior.
```

### Commit Best Practices

- Keep commits small and focused
- Write clear, descriptive commit messages
- Reference issue numbers when applicable
- Separate refactoring from feature changes
- Avoid mixing unrelated changes

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**:
   ```bash
   pytest
   ```

2. **Check code style**:
   ```bash
   flake8 luma_memory tests
   black --check luma_memory tests
   ```

3. **Update documentation** if needed

4. **Add or update tests** for your changes

5. **Update CHANGELOG.md** with your changes

### Submitting a Pull Request

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub

3. **Fill out the PR template** completely:
   - Describe what changes you made
   - Explain why you made them
   - Reference related issues
   - List any breaking changes

4. **Request review** from maintainers

### PR Title Format

Use the same format as commit messages:

```
feat(storage): add connection pooling
fix(api): handle missing entry_id parameter
docs(readme): update installation instructions
```

### PR Description Template

```markdown
## Description
Brief description of the changes

## Motivation
Why are these changes needed?

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
How were these changes tested?

## Related Issues
Closes #123
Related to #456

## Breaking Changes
List any breaking changes (or "None")

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] CHANGELOG.md updated
```

### Review Process

- Maintainers will review your PR within 3-5 business days
- Address review feedback promptly
- Keep discussions focused and professional
- Be open to suggestions and alternative approaches

### After Approval

Once approved, a maintainer will merge your PR. Your contribution will be included in the next release!

## Issue Reporting

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check the documentation** for solutions
3. **Try the latest version** to see if it's already fixed

### Bug Reports

Include the following information:

- **Description**: Clear description of the bug
- **Steps to reproduce**: Minimal steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: Python version, OS, package version
- **Logs**: Relevant error messages or logs
- **Code sample**: Minimal code that reproduces the issue

### Feature Requests

Include the following information:

- **Description**: Clear description of the feature
- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches you've considered
- **Additional context**: Any other relevant information

### Issue Labels

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Documentation improvements
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `question` - Further information requested

## Documentation

### Documentation Standards

- Write clear, concise documentation
- Include code examples
- Keep documentation up-to-date with code changes
- Use proper Markdown formatting

### Documentation Types

1. **Code documentation**: Docstrings in code
2. **API documentation**: API_DOCUMENTATION.md
3. **User guides**: README.md, CONFIG_GUIDE.md
4. **Architecture docs**: docs/ARCHITECTURE.md
5. **Examples**: examples/ directory

### Updating Documentation

When making changes:

- Update relevant docstrings
- Update API documentation if endpoints change
- Update README if user-facing behavior changes
- Add examples for new features
- Update configuration guide for new settings

## Community

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: team@luma.ai for private inquiries

### Contributing Beyond Code

You can contribute in many ways:

- Report bugs and suggest features
- Improve documentation
- Answer questions from other users
- Review pull requests
- Share your use cases and examples
- Spread the word about the project

### Recognition

Contributors are recognized in:

- CHANGELOG.md for each release
- GitHub contributors page
- Release notes

## Questions?

If you have questions about contributing, feel free to:

- Open a GitHub Discussion
- Ask in an issue
- Email team@luma.ai

Thank you for contributing to Luma Memory Module! 🎉
