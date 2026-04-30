"""
Unit tests for SystemPrompt component.

Tests verify:
- Default personality identity contains expected traits
- Identity includes AI assistant instructions
- Identity includes adaptation instructions
- Multiple personality profiles (default, ceo, developer, tutor)
- Deterministic behavior with specific inputs
- Requirements: 2.1-2.7
"""

import pytest

from luma.core.personality.system_prompt import SystemPrompt


@pytest.fixture
def system_prompt():
    """Create a SystemPrompt instance."""
    return SystemPrompt()


class TestSystemPromptDefaultPersonality:
    """Test default personality identity content."""

    def test_default_identity_is_non_empty(self, system_prompt):
        """Test that default identity returns a non-empty string (Requirement 2.1)."""
        identity = system_prompt.get_identity()
        assert identity != ""
        assert len(identity) > 0

    def test_default_identity_contains_luma_name(self, system_prompt):
        """Test that default identity identifies as Luma."""
        identity = system_prompt.get_identity()
        assert "Luma" in identity

    def test_default_identity_contains_ai_assistant_instruction(self, system_prompt):
        """Test that identity includes AI assistant instructions (Requirement 2.3)."""
        identity = system_prompt.get_identity()
        assert "AI assistant" in identity

    def test_default_identity_contains_help_users_instruction(self, system_prompt):
        """Test that identity includes helping users instruction (Requirement 2.3)."""
        identity = system_prompt.get_identity()
        # Should mention helping users learn, solve problems, or achieve goals
        assert any(
            phrase in identity
            for phrase in ["help users", "learn", "solve problems", "achieve"]
        )

    def test_default_identity_contains_adaptation_instruction(self, system_prompt):
        """Test that identity includes adaptation instructions (Requirement 2.4)."""
        identity = system_prompt.get_identity()
        assert "adapt" in identity.lower()
        assert any(
            phrase in identity.lower()
            for phrase in ["communication style", "preferences", "context"]
        )

    def test_default_identity_contains_intelligent_trait(self, system_prompt):
        """Test that default identity contains 'intelligent' trait (Requirement 2.2)."""
        identity = system_prompt.get_identity()
        assert "intelligent" in identity.lower()

    def test_default_identity_contains_practical_trait(self, system_prompt):
        """Test that default identity contains 'practical' trait (Requirement 2.2)."""
        identity = system_prompt.get_identity()
        assert "practical" in identity.lower()

    def test_default_identity_contains_clear_trait(self, system_prompt):
        """Test that default identity contains 'clear' trait (Requirement 2.2)."""
        identity = system_prompt.get_identity()
        assert "clear" in identity.lower()

    def test_default_identity_contains_adaptive_trait(self, system_prompt):
        """Test that default identity contains 'adaptive' trait (Requirement 2.2)."""
        identity = system_prompt.get_identity()
        assert "adaptive" in identity.lower()

    def test_default_identity_contains_structured_trait(self, system_prompt):
        """Test that default identity contains 'structured' trait (Requirement 2.2)."""
        identity = system_prompt.get_identity()
        assert "structured" in identity.lower()

    def test_default_identity_contains_respectful_trait(self, system_prompt):
        """Test that default identity contains 'respectful' trait (Requirement 2.2)."""
        identity = system_prompt.get_identity()
        assert "respectful" in identity.lower()

    def test_default_identity_contains_concise_trait(self, system_prompt):
        """Test that default identity contains 'concise' trait (Requirement 2.2)."""
        identity = system_prompt.get_identity()
        assert "concise" in identity.lower()

    def test_default_identity_contains_actionable_guidance(self, system_prompt):
        """Test that default identity mentions actionable guidance."""
        identity = system_prompt.get_identity()
        assert "actionable" in identity.lower()

    def test_default_identity_avoids_vague_generalities(self, system_prompt):
        """Test that default identity mentions avoiding vague generalities."""
        identity = system_prompt.get_identity()
        assert "vague" in identity.lower() or "generalities" in identity.lower()


class TestSystemPromptCEOPersonality:
    """Test CEO personality profile."""

    def test_ceo_identity_is_non_empty(self, system_prompt):
        """Test that CEO identity returns a non-empty string."""
        identity = system_prompt.get_identity("ceo")
        assert identity != ""
        assert len(identity) > 0

    def test_ceo_identity_contains_luma_name(self, system_prompt):
        """Test that CEO identity identifies as Luma."""
        identity = system_prompt.get_identity("ceo")
        assert "Luma" in identity

    def test_ceo_identity_contains_strategic_trait(self, system_prompt):
        """Test that CEO identity contains 'strategic' trait."""
        identity = system_prompt.get_identity("ceo")
        assert "strategic" in identity.lower()

    def test_ceo_identity_contains_executive_focus(self, system_prompt):
        """Test that CEO identity mentions executive focus."""
        identity = system_prompt.get_identity("ceo")
        assert "executive" in identity.lower() or "leaders" in identity.lower()

    def test_ceo_identity_contains_business_problems(self, system_prompt):
        """Test that CEO identity mentions business problems."""
        identity = system_prompt.get_identity("ceo")
        assert "business" in identity.lower()

    def test_ceo_identity_contains_data_driven_trait(self, system_prompt):
        """Test that CEO identity contains 'data-driven' trait."""
        identity = system_prompt.get_identity("ceo")
        assert "data-driven" in identity.lower() or "data driven" in identity.lower()

    def test_ceo_identity_contains_action_oriented_trait(self, system_prompt):
        """Test that CEO identity contains 'action-oriented' trait."""
        identity = system_prompt.get_identity("ceo")
        assert "action-oriented" in identity.lower() or "action oriented" in identity.lower()

    def test_ceo_identity_contains_adaptation_instruction(self, system_prompt):
        """Test that CEO identity includes adaptation instructions."""
        identity = system_prompt.get_identity("ceo")
        assert "adapt" in identity.lower()


class TestSystemPromptDeveloperPersonality:
    """Test developer personality profile."""

    def test_developer_identity_is_non_empty(self, system_prompt):
        """Test that developer identity returns a non-empty string."""
        identity = system_prompt.get_identity("developer")
        assert identity != ""
        assert len(identity) > 0

    def test_developer_identity_contains_luma_name(self, system_prompt):
        """Test that developer identity identifies as Luma."""
        identity = system_prompt.get_identity("developer")
        assert "Luma" in identity

    def test_developer_identity_contains_technical_trait(self, system_prompt):
        """Test that developer identity contains 'technical' trait."""
        identity = system_prompt.get_identity("developer")
        assert "technical" in identity.lower()

    def test_developer_identity_contains_code_focus(self, system_prompt):
        """Test that developer identity mentions code focus."""
        identity = system_prompt.get_identity("developer")
        assert "code" in identity.lower()

    def test_developer_identity_contains_developers_mention(self, system_prompt):
        """Test that developer identity mentions developers."""
        identity = system_prompt.get_identity("developer")
        assert "developers" in identity.lower()

    def test_developer_identity_contains_precise_trait(self, system_prompt):
        """Test that developer identity contains 'precise' trait."""
        identity = system_prompt.get_identity("developer")
        assert "precise" in identity.lower()

    def test_developer_identity_contains_technically_accurate_trait(self, system_prompt):
        """Test that developer identity contains 'technically accurate' trait."""
        identity = system_prompt.get_identity("developer")
        assert "technically accurate" in identity.lower() or "accurate" in identity.lower()

    def test_developer_identity_contains_practical_trait(self, system_prompt):
        """Test that developer identity contains 'practical' trait."""
        identity = system_prompt.get_identity("developer")
        assert "practical" in identity.lower()

    def test_developer_identity_contains_detail_oriented_trait(self, system_prompt):
        """Test that developer identity contains 'detail-oriented' trait."""
        identity = system_prompt.get_identity("developer")
        assert "detail-oriented" in identity.lower() or "detail oriented" in identity.lower()

    def test_developer_identity_contains_concrete_examples(self, system_prompt):
        """Test that developer identity mentions concrete examples."""
        identity = system_prompt.get_identity("developer")
        assert "concrete" in identity.lower() or "examples" in identity.lower()

    def test_developer_identity_contains_adaptation_instruction(self, system_prompt):
        """Test that developer identity includes adaptation instructions."""
        identity = system_prompt.get_identity("developer")
        assert "adapt" in identity.lower()


class TestSystemPromptTutorPersonality:
    """Test tutor personality profile."""

    def test_tutor_identity_is_non_empty(self, system_prompt):
        """Test that tutor identity returns a non-empty string."""
        identity = system_prompt.get_identity("tutor")
        assert identity != ""
        assert len(identity) > 0

    def test_tutor_identity_contains_luma_name(self, system_prompt):
        """Test that tutor identity identifies as Luma."""
        identity = system_prompt.get_identity("tutor")
        assert "Luma" in identity

    def test_tutor_identity_contains_patient_trait(self, system_prompt):
        """Test that tutor identity contains 'patient' trait."""
        identity = system_prompt.get_identity("tutor")
        assert "patient" in identity.lower()

    def test_tutor_identity_contains_educational_trait(self, system_prompt):
        """Test that tutor identity contains 'educational' trait."""
        identity = system_prompt.get_identity("tutor")
        assert "educational" in identity.lower()

    def test_tutor_identity_contains_learners_mention(self, system_prompt):
        """Test that tutor identity mentions learners."""
        identity = system_prompt.get_identity("tutor")
        assert "learners" in identity.lower() or "learn" in identity.lower()

    def test_tutor_identity_contains_encouraging_trait(self, system_prompt):
        """Test that tutor identity contains 'encouraging' trait."""
        identity = system_prompt.get_identity("tutor")
        assert "encouraging" in identity.lower()

    def test_tutor_identity_contains_clear_trait(self, system_prompt):
        """Test that tutor identity contains 'clear' trait."""
        identity = system_prompt.get_identity("tutor")
        assert "clear" in identity.lower()

    def test_tutor_identity_contains_step_by_step_trait(self, system_prompt):
        """Test that tutor identity contains 'step-by-step' trait."""
        identity = system_prompt.get_identity("tutor")
        assert "step-by-step" in identity.lower() or "step by step" in identity.lower()

    def test_tutor_identity_contains_supportive_trait(self, system_prompt):
        """Test that tutor identity contains 'supportive' trait."""
        identity = system_prompt.get_identity("tutor")
        assert "supportive" in identity.lower()

    def test_tutor_identity_contains_understanding_mention(self, system_prompt):
        """Test that tutor identity mentions building understanding."""
        identity = system_prompt.get_identity("tutor")
        assert "understanding" in identity.lower()

    def test_tutor_identity_contains_adaptation_instruction(self, system_prompt):
        """Test that tutor identity includes adaptation instructions."""
        identity = system_prompt.get_identity("tutor")
        assert "adapt" in identity.lower()


class TestSystemPromptMultipleProfiles:
    """Test multiple personality profiles."""

    def test_all_valid_profiles_return_non_empty_strings(self, system_prompt):
        """Test that all valid personality profiles return non-empty strings."""
        valid_profiles = ["default", "ceo", "developer", "tutor"]
        for profile in valid_profiles:
            identity = system_prompt.get_identity(profile)
            assert identity != ""
            assert len(identity) > 0

    def test_different_profiles_return_different_identities(self, system_prompt):
        """Test that different personality profiles return different identities."""
        default_identity = system_prompt.get_identity("default")
        ceo_identity = system_prompt.get_identity("ceo")
        developer_identity = system_prompt.get_identity("developer")
        tutor_identity = system_prompt.get_identity("tutor")

        # All identities should be different
        assert default_identity != ceo_identity
        assert default_identity != developer_identity
        assert default_identity != tutor_identity
        assert ceo_identity != developer_identity
        assert ceo_identity != tutor_identity
        assert developer_identity != tutor_identity

    def test_invalid_profile_returns_default(self, system_prompt):
        """Test that invalid personality_id returns default identity."""
        default_identity = system_prompt.get_identity("default")
        invalid_identity = system_prompt.get_identity("invalid_profile")
        assert invalid_identity == default_identity

    def test_empty_string_profile_returns_default(self, system_prompt):
        """Test that empty string personality_id returns default identity."""
        default_identity = system_prompt.get_identity("default")
        empty_identity = system_prompt.get_identity("")
        assert empty_identity == default_identity

    def test_none_profile_uses_default_parameter(self, system_prompt):
        """Test that calling without parameter uses default."""
        explicit_default = system_prompt.get_identity("default")
        implicit_default = system_prompt.get_identity()
        assert implicit_default == explicit_default


class TestSystemPromptDeterminism:
    """Test deterministic behavior (Requirement 2.5)."""

    def test_identical_inputs_produce_identical_outputs_default(self, system_prompt):
        """Test that calling get_identity twice with 'default' produces identical results."""
        identity1 = system_prompt.get_identity("default")
        identity2 = system_prompt.get_identity("default")
        assert identity1 == identity2

    def test_identical_inputs_produce_identical_outputs_ceo(self, system_prompt):
        """Test that calling get_identity twice with 'ceo' produces identical results."""
        identity1 = system_prompt.get_identity("ceo")
        identity2 = system_prompt.get_identity("ceo")
        assert identity1 == identity2

    def test_identical_inputs_produce_identical_outputs_developer(self, system_prompt):
        """Test that calling get_identity twice with 'developer' produces identical results."""
        identity1 = system_prompt.get_identity("developer")
        identity2 = system_prompt.get_identity("developer")
        assert identity1 == identity2

    def test_identical_inputs_produce_identical_outputs_tutor(self, system_prompt):
        """Test that calling get_identity twice with 'tutor' produces identical results."""
        identity1 = system_prompt.get_identity("tutor")
        identity2 = system_prompt.get_identity("tutor")
        assert identity1 == identity2

    def test_identical_inputs_produce_identical_outputs_invalid(self, system_prompt):
        """Test that calling get_identity twice with invalid input produces identical results."""
        identity1 = system_prompt.get_identity("invalid")
        identity2 = system_prompt.get_identity("invalid")
        assert identity1 == identity2

    def test_multiple_calls_produce_identical_outputs(self, system_prompt):
        """Test that multiple calls with same input produce identical results."""
        identities = [system_prompt.get_identity("default") for _ in range(10)]
        # All identities should be identical
        assert all(identity == identities[0] for identity in identities)


class TestSystemPromptStatelessness:
    """Test stateless behavior (Requirement 2.6, 2.7)."""

    def test_no_state_mutation_between_calls(self, system_prompt):
        """Test that calling get_identity does not mutate internal state."""
        # Call with different profiles
        system_prompt.get_identity("default")
        system_prompt.get_identity("ceo")
        system_prompt.get_identity("developer")

        # Calling default again should return the same result
        identity1 = system_prompt.get_identity("default")
        identity2 = system_prompt.get_identity("default")
        assert identity1 == identity2

    def test_multiple_instances_produce_identical_outputs(self):
        """Test that different SystemPrompt instances produce identical outputs."""
        prompt1 = SystemPrompt()
        prompt2 = SystemPrompt()

        identity1 = prompt1.get_identity("default")
        identity2 = prompt2.get_identity("default")
        assert identity1 == identity2

    def test_interleaved_calls_produce_consistent_results(self, system_prompt):
        """Test that interleaved calls with different profiles produce consistent results."""
        default1 = system_prompt.get_identity("default")
        ceo1 = system_prompt.get_identity("ceo")
        default2 = system_prompt.get_identity("default")
        ceo2 = system_prompt.get_identity("ceo")

        assert default1 == default2
        assert ceo1 == ceo2
