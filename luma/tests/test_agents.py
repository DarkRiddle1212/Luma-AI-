"""
Unit Tests - Agent System

Tests for LaptopAgent class and AgentInterface.
"""

import pytest
from luma.agents.laptop_agent import LaptopAgent, AgentInterface


class TestLaptopAgent:
    """Tests for LaptopAgent class."""
    
    # Tests for execute() method
    
    def test_execute_returns_expected_structure(self):
        """Test that execute() returns the expected result structure."""
        agent = LaptopAgent()
        task = {"type": "test_task", "param": "value"}
        
        result = agent.execute(task)
        
        assert "status" in result
        assert "task_type" in result
        assert "message" in result
        assert result["status"] == "completed"
        assert result["task_type"] == "test_task"
    
    def test_execute_with_empty_task(self):
        """Test that execute() handles empty task dictionary."""
        agent = LaptopAgent()
        task = {}
        
        result = agent.execute(task)
        
        assert result["status"] == "completed"
        assert result["task_type"] == "unknown"
    
    def test_execute_with_different_task_types(self):
        """Test that execute() handles different task types."""
        agent = LaptopAgent()
        task_types = ["system_monitoring", "file_operations", "network_status"]
        
        for task_type in task_types:
            task = {"type": task_type}
            result = agent.execute(task)
            
            assert result["status"] == "completed"
            assert result["task_type"] == task_type
    
    def test_execute_with_additional_parameters(self):
        """Test that execute() accepts tasks with additional parameters."""
        agent = LaptopAgent()
        task = {
            "type": "file_operations",
            "path": "/test/path",
            "action": "read"
        }
        
        result = agent.execute(task)
        
        assert result["status"] == "completed"
        assert result["task_type"] == "file_operations"
    
    # Tests for get_capabilities() method
    
    def test_get_capabilities_returns_list(self):
        """Test that get_capabilities() returns a list."""
        agent = LaptopAgent()
        
        capabilities = agent.get_capabilities()
        
        assert isinstance(capabilities, list)
    
    def test_get_capabilities_contains_expected_capabilities(self):
        """Test that get_capabilities() returns expected capabilities."""
        agent = LaptopAgent()
        expected_capabilities = [
            "system_monitoring",
            "file_operations",
            "application_management",
            "network_status"
        ]
        
        capabilities = agent.get_capabilities()
        
        assert set(capabilities) == set(expected_capabilities)
    
    def test_get_capabilities_is_consistent(self):
        """Test that get_capabilities() returns consistent results."""
        agent = LaptopAgent()
        
        capabilities1 = agent.get_capabilities()
        capabilities2 = agent.get_capabilities()
        
        assert capabilities1 == capabilities2
    
    # Tests for monitor_system() method
    
    def test_monitor_system_returns_expected_structure(self):
        """Test that monitor_system() returns the expected result structure."""
        agent = LaptopAgent()
        
        metrics = agent.monitor_system()
        
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "disk_percent" in metrics
        assert "timestamp" in metrics
    
    def test_monitor_system_returns_numeric_values(self):
        """Test that monitor_system() returns numeric values for metrics."""
        agent = LaptopAgent()
        
        metrics = agent.monitor_system()
        
        assert isinstance(metrics["cpu_percent"], (int, float))
        assert isinstance(metrics["memory_percent"], (int, float))
        assert isinstance(metrics["disk_percent"], (int, float))
    
    def test_monitor_system_is_callable_multiple_times(self):
        """Test that monitor_system() can be called multiple times."""
        agent = LaptopAgent()
        
        metrics1 = agent.monitor_system()
        metrics2 = agent.monitor_system()
        
        assert metrics1 is not None
        assert metrics2 is not None
        assert "cpu_percent" in metrics1
        assert "cpu_percent" in metrics2


class TestAgentInterface:
    """Tests for AgentInterface abstract base class."""
    
    def test_agent_interface_cannot_be_instantiated(self):
        """Test that AgentInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AgentInterface()
    
    def test_laptop_agent_implements_agent_interface(self):
        """Test that LaptopAgent properly implements AgentInterface."""
        agent = LaptopAgent()
        
        assert isinstance(agent, AgentInterface)
        assert hasattr(agent, "execute")
        assert hasattr(agent, "get_capabilities")
        assert callable(agent.execute)
        assert callable(agent.get_capabilities)
