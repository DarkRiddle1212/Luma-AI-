"""
Multi-Agent Coordination Example

This example demonstrates how multiple agents (laptop, phone, smart home)
can coordinate through the Luma Memory Module to provide a unified user experience.

Scenario: User's daily routine tracked across multiple devices
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List


class LumaAgent:
    """Base agent class with common functionality."""
    
    def __init__(self, device_id: str, base_url: str = "http://localhost:8000"):
        self.device_id = device_id
        self.base_url = base_url.rstrip('/')
    
    def create_memory(self, action: str, context: dict, sensitivity: str = "public", tags: list = None) -> str:
        """Create a memory entry."""
        response = requests.post(
            f"{self.base_url}/api/v1/memory",
            json={
                "action": action,
                "context": context,
                "device_id": self.device_id,
                "sensitivity": sensitivity,
                "tags": tags or []
            }
        )
        response.raise_for_status()
        return response.json()["entry_id"]
    
    def query_memories(self, **kwargs) -> Dict[str, Any]:
        """Query memory entries."""
        response = requests.post(
            f"{self.base_url}/api/v1/memory/query",
            json=kwargs
        )
        response.raise_for_status()
        return response.json()
    
    def get_recent_activities(self, hours: int = 1, tags: list = None) -> List[Dict[str, Any]]:
        """Get recent activities from all devices."""
        start_time = datetime.utcnow() - timedelta(hours=hours)
        results = self.query_memories(
            start_time=start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            tags=tags,
            limit=100
        )
        return results['entries']
    
    def get_device_activities(self, device_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get activities from a specific device."""
        # Note: This is a simplified version. In production, you'd filter by device_id
        # which would require querying and filtering the results
        results = self.query_memories(limit=limit)
        if device_id:
            return [e for e in results['entries'] if e['device_id'] == device_id]
        return results['entries']


class LaptopAgent(LumaAgent):
    """Agent running on user's laptop."""
    
    def __init__(self):
        super().__init__(device_id="laptop-001")
    
    def track_work_session(self):
        """Track a work session."""
        print(f"[{self.device_id}] Starting work session...")
        
        # User starts working
        self.create_memory(
            action="User started work session",
            context={
                "time": datetime.utcnow().strftime("%H:%M"),
                "workspace": "/projects/luma"
            },
            tags=["work", "session_start", "productivity"]
        )
        
        # User opens development tools
        self.create_memory(
            action="User opened VS Code",
            context={
                "project": "luma-memory",
                "files_open": 5
            },
            tags=["work", "coding", "vscode"]
        )
        
        # User runs tests
        self.create_memory(
            action="User ran tests",
            context={
                "command": "pytest",
                "tests_passed": 42,
                "duration_seconds": 3.2
            },
            tags=["work", "testing", "success"]
        )
        
        print(f"[{self.device_id}] Work session tracked")


class PhoneAgent(LumaAgent):
    """Agent running on user's phone."""
    
    def __init__(self):
        super().__init__(device_id="phone-001")
    
    def track_commute(self):
        """Track user's commute."""
        print(f"[{self.device_id}] Tracking commute...")
        
        # User leaves home
        self.create_memory(
            action="User left home",
            context={
                "time": datetime.utcnow().strftime("%H:%M"),
                "location_type": "home",
                "destination": "office"
            },
            sensitivity="private",
            tags=["location", "commute", "travel"]
        )
        
        # User listens to podcast
        self.create_memory(
            action="User started podcast",
            context={
                "app": "Spotify",
                "podcast": "Tech Talk Daily",
                "episode": "AI in 2024"
            },
            tags=["media", "podcast", "entertainment"]
        )
        
        # User arrives at office
        self.create_memory(
            action="User arrived at office",
            context={
                "time": datetime.utcnow().strftime("%H:%M"),
                "location_type": "office",
                "wifi_network": "OfficeWiFi"
            },
            sensitivity="private",
            tags=["location", "office", "arrival"]
        )
        
        print(f"[{self.device_id}] Commute tracked")
    
    def check_notifications(self):
        """Track notification interactions."""
        print(f"[{self.device_id}] Checking notifications...")
        
        # User receives meeting reminder
        self.create_memory(
            action="User received notification",
            context={
                "app": "Calendar",
                "type": "meeting_reminder",
                "title": "Team Standup",
                "time": "10:00 AM"
            },
            tags=["notification", "calendar", "meeting"]
        )
        
        # User dismisses notification
        self.create_memory(
            action="User dismissed notification",
            context={
                "app": "Calendar",
                "action": "dismissed"
            },
            tags=["notification", "interaction"]
        )
        
        print(f"[{self.device_id}] Notifications tracked")


class SmartHomeAgent(LumaAgent):
    """Agent running on smart home hub."""
    
    def __init__(self):
        super().__init__(device_id="smarthome-001")
    
    def track_morning_routine(self):
        """Track smart home morning routine."""
        print(f"[{self.device_id}] Tracking morning routine...")
        
        # Lights turned on
        self.create_memory(
            action="Automation triggered: Morning lights",
            context={
                "automation": "morning_routine",
                "devices": ["bedroom_light", "kitchen_light"],
                "trigger": "time_based"
            },
            tags=["automation", "smart_home", "lights"]
        )
        
        # Thermostat adjusted
        self.create_memory(
            action="Thermostat adjusted",
            context={
                "device": "nest_thermostat",
                "temperature": 72,
                "mode": "heat",
                "trigger": "automation"
            },
            tags=["automation", "smart_home", "climate"]
        )
        
        # Coffee maker started
        self.create_memory(
            action="Coffee maker started",
            context={
                "device": "smart_coffee_maker",
                "brew_strength": "medium",
                "cups": 2
            },
            tags=["automation", "smart_home", "appliance"]
        )
        
        print(f"[{self.device_id}] Morning routine tracked")


class CoordinatorAgent(LumaAgent):
    """Coordinator agent that analyzes activities across all devices."""
    
    def __init__(self):
        super().__init__(device_id="coordinator-001")
    
    def analyze_user_context(self):
        """Analyze user's current context from all devices."""
        print(f"\n[{self.device_id}] Analyzing user context...")
        
        # Get recent activities from all devices
        recent = self.get_recent_activities(hours=24)
        
        # Analyze by device
        devices = {}
        for entry in recent:
            device = entry['device_id']
            if device not in devices:
                devices[device] = []
            devices[device].append(entry)
        
        print(f"\n  Activity Summary (last 24 hours):")
        print(f"  Total activities: {len(recent)}")
        print(f"  Active devices: {len(devices)}")
        
        for device, activities in devices.items():
            print(f"\n  {device}:")
            print(f"    - {len(activities)} activities")
            
            # Show most recent activity
            if activities:
                latest = activities[0]
                print(f"    - Latest: {latest['action']}")
        
        # Analyze by tags
        tag_counts = {}
        for entry in recent:
            for tag in entry.get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        print(f"\n  Top Activity Categories:")
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        for tag, count in sorted_tags[:5]:
            print(f"    - {tag}: {count} activities")
    
    def generate_daily_summary(self):
        """Generate a summary of the user's day."""
        print(f"\n[{self.device_id}] Generating daily summary...")
        
        # Get today's activities
        start_of_day = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        results = self.query_memories(
            start_time=start_of_day.strftime("%Y-%m-%dT%H:%M:%SZ"),
            limit=1000
        )
        
        activities = results['entries']
        
        # Categorize activities
        work_activities = [a for a in activities if 'work' in a.get('tags', [])]
        location_changes = [a for a in activities if 'location' in a.get('tags', [])]
        automation_events = [a for a in activities if 'automation' in a.get('tags', [])]
        
        print(f"\n  Daily Summary:")
        print(f"  Total activities: {len(activities)}")
        print(f"  Work activities: {len(work_activities)}")
        print(f"  Location changes: {len(location_changes)}")
        print(f"  Smart home automations: {len(automation_events)}")
        
        # Identify patterns
        if work_activities:
            print(f"\n  Work Pattern:")
            print(f"    - Started work session")
            print(f"    - Completed {len([a for a in work_activities if 'testing' in a.get('tags', [])])} test runs")
        
        if location_changes:
            print(f"\n  Movement Pattern:")
            for loc in location_changes:
                print(f"    - {loc['action']}")
    
    def suggest_next_action(self):
        """Suggest next action based on recent context."""
        print(f"\n[{self.device_id}] Suggesting next action...")
        
        # Get very recent activities (last 30 minutes)
        recent = self.get_recent_activities(hours=0.5)
        
        if not recent:
            print("  No recent activities to analyze")
            return
        
        # Analyze recent patterns
        recent_tags = []
        for entry in recent:
            recent_tags.extend(entry.get('tags', []))
        
        # Simple suggestion logic
        if 'work' in recent_tags and 'testing' in recent_tags:
            print("  Suggestion: Consider committing your changes")
        elif 'location' in recent_tags and 'office' in recent_tags:
            print("  Suggestion: Check your calendar for upcoming meetings")
        elif 'notification' in recent_tags:
            print("  Suggestion: Review and respond to pending notifications")
        else:
            print("  Suggestion: Continue with current activity")


def main():
    """
    Demonstrate multi-agent coordination.
    
    This simulates a user's morning routine tracked across multiple devices.
    """
    print("=" * 70)
    print("Multi-Agent Coordination Example")
    print("=" * 70)
    print("\nSimulating user's morning routine across multiple devices...\n")
    
    # Initialize agents
    laptop = LaptopAgent()
    phone = PhoneAgent()
    smart_home = SmartHomeAgent()
    coordinator = CoordinatorAgent()
    
    # Simulate morning routine
    print("PHASE 1: Morning at Home")
    print("-" * 70)
    smart_home.track_morning_routine()
    time.sleep(0.5)
    
    print("\nPHASE 2: Commute to Office")
    print("-" * 70)
    phone.track_commute()
    time.sleep(0.5)
    
    print("\nPHASE 3: Work Session")
    print("-" * 70)
    laptop.track_work_session()
    time.sleep(0.5)
    
    print("\nPHASE 4: Notifications")
    print("-" * 70)
    phone.check_notifications()
    time.sleep(0.5)
    
    # Coordinator analyzes the context
    print("\n" + "=" * 70)
    print("COORDINATION & ANALYSIS")
    print("=" * 70)
    
    coordinator.analyze_user_context()
    coordinator.generate_daily_summary()
    coordinator.suggest_next_action()
    
    print("\n" + "=" * 70)
    print("✓ Multi-agent coordination example completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Multiple agents can store memories independently")
    print("  2. All memories are accessible through a unified API")
    print("  3. Coordinator agents can analyze cross-device patterns")
    print("  4. Rich context enables intelligent suggestions")
    print()


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Cannot connect to Luma Memory API")
        print("  Make sure the server is running:")
        print("    python -m luma_memory.api.server")
    except Exception as e:
        print(f"\n✗ Error: {e}")
