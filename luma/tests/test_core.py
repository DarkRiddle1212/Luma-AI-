"""
Unit Tests - Core Modules

Tests for ReasoningEngine and TaskScheduler classes.
"""

import pytest
from datetime import datetime, timedelta
from luma.core.reasoning import ReasoningEngine
from luma.core.scheduler import TaskScheduler, Task


class TestReasoningEngine:
    """Tests for ReasoningEngine class."""
    
    def test_process_returns_expected_structure(self):
        """Test that process() returns the expected result structure."""
        engine = ReasoningEngine()
        input_data = {"key1": "value1", "key2": "value2"}
        
        result = engine.process(input_data)
        
        assert "status" in result
        assert "input_keys" in result
        assert "reasoning" in result
        assert result["status"] == "processed"
        assert set(result["input_keys"]) == {"key1", "key2"}
    
    def test_process_with_empty_input(self):
        """Test that process() handles empty input correctly."""
        engine = ReasoningEngine()
        input_data = {}
        
        result = engine.process(input_data)
        
        assert result["status"] == "processed"
        assert result["input_keys"] == []
    
    def test_process_with_complex_data(self):
        """Test that process() handles complex nested data."""
        engine = ReasoningEngine()
        input_data = {
            "nested": {"inner": "value"},
            "list": [1, 2, 3],
            "number": 42
        }
        
        result = engine.process(input_data)
        
        assert result["status"] == "processed"
        assert len(result["input_keys"]) == 3

    def test_analyze_context_returns_expected_structure(self):
        """Test that analyze_context() returns the expected result structure."""
        engine = ReasoningEngine()
        context = {"user": "test_user", "action": "login"}
        
        result = engine.analyze_context(context)
        
        assert "context_size" in result
        assert "insights" in result
        assert "confidence" in result
        assert result["context_size"] == 2
        assert isinstance(result["insights"], list)
        assert isinstance(result["confidence"], float)
    
    def test_analyze_context_with_empty_context(self):
        """Test that analyze_context() handles empty context correctly."""
        engine = ReasoningEngine()
        context = {}
        
        result = engine.analyze_context(context)
        
        assert result["context_size"] == 0
        assert result["insights"] == []
        assert result["confidence"] == 0.0
    
    def test_analyze_context_with_large_context(self):
        """Test that analyze_context() handles large context data."""
        engine = ReasoningEngine()
        context = {f"key_{i}": f"value_{i}" for i in range(100)}
        
        result = engine.analyze_context(context)
        
        assert result["context_size"] == 100



class TestTaskScheduler:
    """Tests for TaskScheduler class."""
    
    def test_schedule_task_returns_task_id(self):
        """Test that schedule_task() returns the task ID."""
        scheduler = TaskScheduler()
        task = Task(
            id="task_1",
            name="Test Task",
            schedule="daily",
            payload={"data": "test"},
            next_run=datetime.now() + timedelta(days=1)
        )
        
        task_id = scheduler.schedule_task(task)
        
        assert task_id == "task_1"
    
    def test_schedule_task_adds_to_pending_tasks(self):
        """Test that schedule_task() adds task to pending tasks list."""
        scheduler = TaskScheduler()
        task = Task(
            id="task_2",
            name="Another Task",
            schedule="hourly",
            payload={},
            next_run=datetime.now() + timedelta(hours=1)
        )
        
        scheduler.schedule_task(task)
        pending = scheduler.get_pending_tasks()
        
        assert len(pending) == 1
        assert pending[0].id == "task_2"
    
    def test_schedule_multiple_tasks(self):
        """Test scheduling multiple tasks."""
        scheduler = TaskScheduler()
        task1 = Task(
            id="task_1",
            name="Task 1",
            schedule="daily",
            payload={},
            next_run=datetime.now()
        )
        task2 = Task(
            id="task_2",
            name="Task 2",
            schedule="weekly",
            payload={},
            next_run=datetime.now()
        )
        
        scheduler.schedule_task(task1)
        scheduler.schedule_task(task2)
        pending = scheduler.get_pending_tasks()
        
        assert len(pending) == 2

    
    def test_cancel_task_removes_task(self):
        """Test that cancel_task() removes the task from pending tasks."""
        scheduler = TaskScheduler()
        task = Task(
            id="task_to_cancel",
            name="Cancellable Task",
            schedule="daily",
            payload={},
            next_run=datetime.now()
        )
        
        scheduler.schedule_task(task)
        result = scheduler.cancel_task("task_to_cancel")
        pending = scheduler.get_pending_tasks()
        
        assert result is True
        assert len(pending) == 0
    
    def test_cancel_task_returns_false_for_nonexistent_task(self):
        """Test that cancel_task() returns False for non-existent task."""
        scheduler = TaskScheduler()
        
        result = scheduler.cancel_task("nonexistent_task")
        
        assert result is False
    
    def test_cancel_task_only_removes_specified_task(self):
        """Test that cancel_task() only removes the specified task."""
        scheduler = TaskScheduler()
        task1 = Task(
            id="task_1",
            name="Task 1",
            schedule="daily",
            payload={},
            next_run=datetime.now()
        )
        task2 = Task(
            id="task_2",
            name="Task 2",
            schedule="daily",
            payload={},
            next_run=datetime.now()
        )
        
        scheduler.schedule_task(task1)
        scheduler.schedule_task(task2)
        scheduler.cancel_task("task_1")
        pending = scheduler.get_pending_tasks()
        
        assert len(pending) == 1
        assert pending[0].id == "task_2"

    
    def test_get_pending_tasks_returns_empty_list_initially(self):
        """Test that get_pending_tasks() returns empty list for new scheduler."""
        scheduler = TaskScheduler()
        
        pending = scheduler.get_pending_tasks()
        
        assert pending == []
    
    def test_get_pending_tasks_returns_all_tasks(self):
        """Test that get_pending_tasks() returns all scheduled tasks."""
        scheduler = TaskScheduler()
        task1 = Task(
            id="task_1",
            name="Task 1",
            schedule="daily",
            payload={},
            next_run=datetime.now()
        )
        task2 = Task(
            id="task_2",
            name="Task 2",
            schedule="hourly",
            payload={},
            next_run=datetime.now()
        )
        
        scheduler.schedule_task(task1)
        scheduler.schedule_task(task2)
        pending = scheduler.get_pending_tasks()
        
        assert len(pending) == 2
        assert {t.id for t in pending} == {"task_1", "task_2"}
    
    def test_get_pending_tasks_returns_copy(self):
        """Test that get_pending_tasks() returns a copy, not the original list."""
        scheduler = TaskScheduler()
        task = Task(
            id="task_1",
            name="Task 1",
            schedule="daily",
            payload={},
            next_run=datetime.now()
        )
        
        scheduler.schedule_task(task)
        pending1 = scheduler.get_pending_tasks()
        pending1.clear()
        pending2 = scheduler.get_pending_tasks()
        
        assert len(pending2) == 1
