"""
Unit Tests for Memory Extractor

Tests the MemoryExtractor class to verify:
- Extraction of project goals from user queries
- Extraction of user preferences from responses
- Extraction of facts from declarative statements
- Memory type classification accuracy
- Empty list return for non-memorable interactions
- Graceful handling of empty/None inputs

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 9.1**
"""

import pytest
from luma.core.memory_write.memory_extractor import MemoryExtractor
from luma.core.memory_write.schemas import MemoryCandidate


class TestMemoryExtractorProjectGoals:
    """Test suite for extracting project goals from interactions."""
    
    def test_extract_project_goal_i_want_to(self):
        """Test extraction of project goal with 'I want to' pattern."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I want to build a web application for task management.",
            system_response="That sounds like a great project!"
        )
        
        # Should extract the project goal
        project_goals = [c for c in result if c.type == "project_goal"]
        assert len(project_goals) > 0
        assert any("build a web application" in c.text.lower() for c in project_goals)
    
    def test_extract_project_goal_im_building(self):
        """Test extraction of project goal with 'I'm building' pattern."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I'm building a machine learning model for image classification.",
            system_response="Great! Let me help you with that."
        )
        
        project_goals = [c for c in result if c.type == "project_goal"]
        assert len(project_goals) > 0
        assert any("building a machine learning model" in c.text.lower() for c in project_goals)
    
    def test_extract_project_goal_my_goal_is(self):
        """Test extraction of project goal with 'My goal is to' pattern."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="My goal is to create a mobile app for fitness tracking.",
            system_response="That's an excellent goal!"
        )
        
        project_goals = [c for c in result if c.type == "project_goal"]
        assert len(project_goals) > 0
        assert any("create a mobile app" in c.text.lower() for c in project_goals)
    
    def test_extract_project_goal_im_trying_to(self):
        """Test extraction of project goal with 'I'm trying to' pattern."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I'm trying to develop a REST API for my service.",
            system_response="I can help you with that."
        )
        
        project_goals = [c for c in result if c.type == "project_goal"]
        assert len(project_goals) > 0
        assert any("develop a rest api" in c.text.lower() for c in project_goals)
    
    def test_extract_multiple_project_goals(self):
        """Test extraction of multiple project goals from one interaction."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I want to build a web app. I'm working on a mobile version too.",
            system_response="Both sound great!"
        )
        
        project_goals = [c for c in result if c.type == "project_goal"]
        assert len(project_goals) >= 2


class TestMemoryExtractorUserPreferences:
    """Test suite for extracting user preferences from interactions."""
    
    def test_extract_preference_i_prefer(self):
        """Test extraction of user preference with 'I prefer' pattern."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I prefer using TypeScript over JavaScript.",
            system_response="TypeScript is a great choice!"
        )
        
        preferences = [c for c in result if c.type == "user_preference"]
        assert len(preferences) > 0
        assert any("prefer using typescript" in c.text.lower() for c in preferences)
    
    def test_extract_preference_i_like(self):
        """Test extraction of user preference with 'I like' pattern."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I like writing clean, well-documented code.",
            system_response="That's a good practice!"
        )
        
        preferences = [c for c in result if c.type == "user_preference"]
        assert len(preferences) > 0
        assert any("like writing clean" in c.text.lower() for c in preferences)
    
    def test_extract_preference_i_always(self):
        """Test extraction of user preference with 'I always' pattern."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I always use Git for version control.",
            system_response="Git is essential for modern development."
        )
        
        preferences = [c for c in result if c.type == "user_preference"]
        assert len(preferences) > 0
        assert any("always use git" in c.text.lower() for c in preferences)
    
    def test_extract_preference_i_dont_like(self):
        """Test extraction of negative user preference with 'I don't like' pattern."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I don't like using global variables in my code.",
            system_response="That's a good practice to avoid."
        )
        
        preferences = [c for c in result if c.type == "user_preference"]
        assert len(preferences) > 0
        assert any("don't like using global" in c.text.lower() for c in preferences)
    
    def test_extract_preference_my_preference_is(self):
        """Test extraction of user preference with 'My preference is' pattern."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="My preference is to use functional programming patterns.",
            system_response="Functional programming has many benefits."
        )
        
        preferences = [c for c in result if c.type == "user_preference"]
        assert len(preferences) > 0
        assert any("preference is to use functional" in c.text.lower() for c in preferences)


class TestMemoryExtractorFacts:
    """Test suite for extracting facts from declarative statements."""
    
    def test_extract_fact_i_am(self):
        """Test extraction of fact with 'I am' pattern."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I am a software engineer working on backend systems.",
            system_response="Great to meet you!"
        )
        
        facts = [c for c in result if c.type == "fact"]
        assert len(facts) > 0
        assert any("software engineer" in c.text.lower() for c in facts)
    
    def test_extract_fact_my_name_is(self):
        """Test extraction of fact with 'My name is' pattern."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="My name is Alex and I work at TechCorp.",
            system_response="Nice to meet you, Alex!"
        )
        
        facts = [c for c in result if c.type == "fact"]
        assert len(facts) > 0
        # Should extract at least the name fact
        assert any("alex" in c.text.lower() for c in facts)
    
    def test_extract_fact_i_work_at(self):
        """Test extraction of fact with 'I work at' pattern."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I work at a startup in San Francisco.",
            system_response="Startups are exciting!"
        )
        
        facts = [c for c in result if c.type == "fact"]
        assert len(facts) > 0
        assert any("startup" in c.text.lower() for c in facts)
    
    def test_extract_fact_i_use(self):
        """Test extraction of fact with 'I use' pattern."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I use Python and Go for most of my projects.",
            system_response="Both are excellent languages!"
        )
        
        facts = [c for c in result if c.type == "fact"]
        assert len(facts) > 0
        assert any("python" in c.text.lower() for c in facts)
    
    def test_extract_fact_the_project_is(self):
        """Test extraction of fact with 'The project is' pattern."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="The project is a microservices architecture using Docker.",
            system_response="Microservices are a good choice for scalability."
        )
        
        facts = [c for c in result if c.type == "fact"]
        assert len(facts) > 0
        assert any("microservices" in c.text.lower() for c in facts)


class TestMemoryExtractorStatements:
    """Test suite for extracting important statements."""
    
    def test_extract_statement_with_must_keyword(self):
        """Test extraction of important statement with 'must' keyword."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="The application must support real-time updates.",
            system_response="I understand the requirement."
        )
        
        statements = [c for c in result if c.type == "statement"]
        assert len(statements) > 0
        assert any("must support real-time" in c.text.lower() for c in statements)
    
    def test_extract_statement_with_requirement_keyword(self):
        """Test extraction of important statement with 'requirement' keyword."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="A key requirement is that the system handles 1000 concurrent users.",
            system_response="That's an important constraint."
        )
        
        statements = [c for c in result if c.type == "statement"]
        assert len(statements) > 0
        assert any("requirement" in c.text.lower() for c in statements)
    
    def test_extract_statement_with_critical_keyword(self):
        """Test extraction of important statement with 'critical' keyword."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="It's critical that we maintain backward compatibility.",
            system_response="I'll keep that in mind."
        )
        
        statements = [c for c in result if c.type == "statement"]
        assert len(statements) > 0
        assert any("critical" in c.text.lower() for c in statements)


class TestMemoryExtractorTypeClassification:
    """Test suite for memory type classification accuracy."""
    
    def test_classify_project_goal_correctly(self):
        """Test that project goals are classified with correct type."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I want to create a dashboard for analytics.",
            system_response="That's a great idea!"
        )
        
        project_goals = [c for c in result if c.type == "project_goal"]
        assert len(project_goals) > 0
        for goal in project_goals:
            assert goal.type == "project_goal"
    
    def test_classify_user_preference_correctly(self):
        """Test that user preferences are classified with correct type."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I prefer using React for frontend development.",
            system_response="React is very popular!"
        )
        
        preferences = [c for c in result if c.type == "user_preference"]
        assert len(preferences) > 0
        for pref in preferences:
            assert pref.type == "user_preference"
    
    def test_classify_fact_correctly(self):
        """Test that facts are classified with correct type."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I am a data scientist specializing in NLP.",
            system_response="NLP is a fascinating field!"
        )
        
        facts = [c for c in result if c.type == "fact"]
        assert len(facts) > 0
        for fact in facts:
            assert fact.type == "fact"
    
    def test_classify_statement_correctly(self):
        """Test that statements are classified with correct type."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="The system must be highly available and fault-tolerant.",
            system_response="I understand the requirements."
        )
        
        statements = [c for c in result if c.type == "statement"]
        assert len(statements) > 0
        for stmt in statements:
            assert stmt.type == "statement"
    
    def test_all_candidates_have_valid_types(self):
        """Test that all extracted candidates have valid memory types."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I want to build an app. I prefer Python. I am a developer. The system must be scalable.",
            system_response="I can help with all of that!"
        )
        
        valid_types = ["project_goal", "user_preference", "fact", "statement"]
        for candidate in result:
            assert candidate.type in valid_types


class TestMemoryExtractorNonMemorableInteractions:
    """Test suite for handling non-memorable interactions."""
    
    def test_greeting_returns_empty_list(self):
        """Test that simple greetings return empty list."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="Hello!",
            system_response="Hi there! How can I help you?"
        )
        
        assert result == []
    
    def test_acknowledgement_returns_empty_list(self):
        """Test that simple acknowledgements return empty list."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="Thanks!",
            system_response="You're welcome!"
        )
        
        assert result == []
    
    def test_short_conversation_returns_empty_list(self):
        """Test that very short conversations return empty list."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="Ok",
            system_response="Great!"
        )
        
        assert result == []
    
    def test_goodbye_returns_empty_list(self):
        """Test that goodbyes return empty list."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="Bye!",
            system_response="Goodbye! Have a great day!"
        )
        
        assert result == []
    
    def test_yes_no_returns_empty_list(self):
        """Test that simple yes/no responses return empty list."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="Yes",
            system_response="Okay, got it."
        )
        
        assert result == []
    
    def test_filters_low_value_content(self):
        """Test that low-value content is filtered out."""
        extractor = MemoryExtractor()
        
        # Mix of low-value and valuable content
        result = extractor.extract_candidates(
            user_query="Hello! I want to build a web application.",
            system_response="Hi! That sounds great!"
        )
        
        # Should extract the project goal but not the greeting
        project_goals = [c for c in result if c.type == "project_goal"]
        assert len(project_goals) > 0
        
        # Should not have extracted just "Hello" or "Hi"
        for candidate in result:
            assert candidate.text.lower().strip() not in ["hello", "hi", "hello!", "hi!"]


class TestMemoryExtractorEmptyInputHandling:
    """Test suite for graceful handling of empty/None inputs."""
    
    def test_empty_user_query_returns_empty_list(self):
        """Test that empty user query returns empty list."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="",
            system_response="This is a response."
        )
        
        assert result == []
    
    def test_empty_system_response_returns_empty_list(self):
        """Test that empty system response returns empty list."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="This is a query.",
            system_response=""
        )
        
        assert result == []
    
    def test_both_empty_returns_empty_list(self):
        """Test that both empty inputs return empty list."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="",
            system_response=""
        )
        
        assert result == []
    
    def test_none_user_query_returns_empty_list(self):
        """Test that None user query returns empty list."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query=None,
            system_response="This is a response."
        )
        
        assert result == []
    
    def test_none_system_response_returns_empty_list(self):
        """Test that None system response returns empty list."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="This is a query.",
            system_response=None
        )
        
        assert result == []
    
    def test_both_none_returns_empty_list(self):
        """Test that both None inputs return empty list."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query=None,
            system_response=None
        )
        
        assert result == []
    
    def test_whitespace_only_user_query_returns_empty_list(self):
        """Test that whitespace-only user query returns empty list."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="   \n\t  ",
            system_response="This is a response."
        )
        
        assert result == []
    
    def test_whitespace_only_system_response_returns_empty_list(self):
        """Test that whitespace-only system response returns empty list."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="This is a query.",
            system_response="   \n\t  "
        )
        
        assert result == []


class TestMemoryExtractorReturnTypes:
    """Test suite for verifying return types and data structures."""
    
    def test_returns_list(self):
        """Test that extract_candidates returns a list."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I want to build an app.",
            system_response="Great idea!"
        )
        
        assert isinstance(result, list)
    
    def test_returns_memory_candidate_objects(self):
        """Test that returned list contains MemoryCandidate objects."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I want to build an app.",
            system_response="Great idea!"
        )
        
        for candidate in result:
            assert isinstance(candidate, MemoryCandidate)
    
    def test_candidates_have_text_field(self):
        """Test that all candidates have non-empty text field."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I want to build an app. I prefer Python.",
            system_response="Great choices!"
        )
        
        for candidate in result:
            assert hasattr(candidate, 'text')
            assert isinstance(candidate.text, str)
            assert len(candidate.text) > 0
    
    def test_candidates_have_type_field(self):
        """Test that all candidates have valid type field."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I want to build an app. I prefer Python.",
            system_response="Great choices!"
        )
        
        valid_types = ["project_goal", "user_preference", "fact", "statement"]
        for candidate in result:
            assert hasattr(candidate, 'type')
            assert candidate.type in valid_types


class TestMemoryExtractorEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_very_long_text(self):
        """Test extraction from very long text."""
        extractor = MemoryExtractor()
        
        long_query = "I want to build " + "a " * 100 + "web application."
        result = extractor.extract_candidates(
            user_query=long_query,
            system_response="That's interesting!"
        )
        
        # Should still extract something
        assert isinstance(result, list)
    
    def test_special_characters_in_text(self):
        """Test extraction with special characters."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I want to build a web app with @mentions and #hashtags!",
            system_response="Sounds cool!"
        )
        
        project_goals = [c for c in result if c.type == "project_goal"]
        assert len(project_goals) > 0
    
    def test_multiple_sentences_in_query(self):
        """Test extraction from multiple sentences."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I want to build an app. I prefer using React. I am a frontend developer.",
            system_response="Great! I can help with that."
        )
        
        # Should extract multiple candidates
        assert len(result) >= 3
    
    def test_case_insensitive_pattern_matching(self):
        """Test that pattern matching is case-insensitive."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I WANT TO BUILD AN APP.",
            system_response="Great!"
        )
        
        project_goals = [c for c in result if c.type == "project_goal"]
        assert len(project_goals) > 0
    
    def test_minimum_length_filter(self):
        """Test that very short extractions are filtered out."""
        extractor = MemoryExtractor()
        
        result = extractor.extract_candidates(
            user_query="I want to do it.",
            system_response="Okay!"
        )
        
        # "do it" is too short and vague, should be filtered
        # The implementation has minimum length filters
        for candidate in result:
            # Extracted text should have some substance
            assert len(candidate.text.strip()) > 5
