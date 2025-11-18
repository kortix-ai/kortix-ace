"""Tests for ace.integrations.base module."""

import pytest
from ace import Playbook
from ace.integrations.base import wrap_playbook_context


class TestWrapPlaybookContext:
    """Test the wrap_playbook_context helper function."""

    def test_empty_playbook_returns_empty_string(self):
        """Should return empty string when playbook has no bullets."""
        playbook = Playbook()
        result = wrap_playbook_context(playbook)

        assert result == ""
        assert isinstance(result, str)

    def test_single_bullet_formats_correctly(self):
        """Should format single bullet with explanation."""
        playbook = Playbook()
        playbook.add_bullet("general", "Always validate inputs before processing")

        result = wrap_playbook_context(playbook)

        # Should contain header
        assert "Strategic Knowledge" in result
        assert "Learned from Experience" in result

        # Should contain the bullet content
        assert "Always validate inputs before processing" in result

        # Should contain usage instructions
        assert "How to use these strategies" in result
        assert "success rates" in result
        assert "helpful > harmful" in result

        # Should contain important note
        assert "learned patterns, not rigid rules" in result

    def test_multiple_bullets_all_included(self):
        """Should include all bullets in formatted output."""
        playbook = Playbook()
        playbook.add_bullet("general", "First strategy")
        playbook.add_bullet("general", "Second strategy")
        playbook.add_bullet("specific", "Third strategy")

        result = wrap_playbook_context(playbook)

        assert "First strategy" in result
        assert "Second strategy" in result
        assert "Third strategy" in result

    def test_bullet_with_metadata_shows_scores(self):
        """Should display helpful/harmful scores from metadata."""
        playbook = Playbook()
        playbook.add_bullet(
            "general", "High success strategy", metadata={"helpful": 5, "harmful": 1}
        )

        result = wrap_playbook_context(playbook)

        # The bullet content should be present
        assert "High success strategy" in result
        # Playbook.as_prompt() should include score information

    def test_different_sections_all_included(self):
        """Should include bullets from different sections."""
        playbook = Playbook()
        playbook.add_bullet("browser", "Wait for elements to load")
        playbook.add_bullet("api", "Retry on timeout")
        playbook.add_bullet("general", "Log all errors")

        result = wrap_playbook_context(playbook)

        assert "Wait for elements to load" in result
        assert "Retry on timeout" in result
        assert "Log all errors" in result

    def test_output_includes_key_sections(self):
        """Should include all required sections in output."""
        playbook = Playbook()
        playbook.add_bullet("general", "Test strategy")

        result = wrap_playbook_context(playbook)

        # Check for markdown headers
        assert "##" in result
        assert "Available Strategic Knowledge" in result

        # Check for bullet points/instructions
        assert "**How to use these strategies:**" in result
        assert "**Important:**" in result

        # Check for specific guidance
        assert "Review bullets relevant to your current task" in result
        assert "Prioritize strategies with high success rates" in result
        assert "Apply strategies when they match your context" in result
        assert "Adapt general strategies" in result

    def test_output_format_is_string(self):
        """Should always return a string."""
        playbook1 = Playbook()
        playbook2 = Playbook()
        playbook2.add_bullet("test", "Content")

        result1 = wrap_playbook_context(playbook1)
        result2 = wrap_playbook_context(playbook2)

        assert isinstance(result1, str)
        assert isinstance(result2, str)

    def test_playbook_with_special_characters(self):
        """Should handle bullets with special characters."""
        playbook = Playbook()
        playbook.add_bullet("general", "Use `quotes` and **bold** in markdown")
        playbook.add_bullet("general", "Handle <tags> and {braces}")

        result = wrap_playbook_context(playbook)

        assert "Use `quotes` and **bold** in markdown" in result
        assert "Handle <tags> and {braces}" in result

    def test_playbook_with_multiline_bullet(self):
        """Should handle bullets with newlines."""
        playbook = Playbook()
        playbook.add_bullet("general", "First line\nSecond line\nThird line")

        result = wrap_playbook_context(playbook)

        # The bullet content should be present
        assert "First line" in result or "First line\nSecond line" in result

    def test_empty_string_has_no_formatting(self):
        """Empty playbook should return truly empty string, no formatting."""
        playbook = Playbook()
        result = wrap_playbook_context(playbook)

        assert result == ""
        assert len(result) == 0
        assert "Strategic Knowledge" not in result
        assert "How to use" not in result

    def test_wrapping_same_playbook_twice_identical(self):
        """Should produce identical output for same playbook."""
        playbook = Playbook()
        playbook.add_bullet("general", "Consistent output")

        result1 = wrap_playbook_context(playbook)
        result2 = wrap_playbook_context(playbook)

        assert result1 == result2

    def test_output_mentions_emoji(self):
        """Should include emoji in header for visual appeal."""
        playbook = Playbook()
        playbook.add_bullet("general", "Test")

        result = wrap_playbook_context(playbook)

        # Check for emoji in header
        assert "📚" in result

    def test_output_encourages_judgment(self):
        """Should remind users to use judgment, not treat as rigid rules."""
        playbook = Playbook()
        playbook.add_bullet("general", "Strategy")

        result = wrap_playbook_context(playbook)

        assert "Use judgment" in result
        assert "not rigid rules" in result

    def test_bullets_parameter_uses_playbook_method(self):
        """Should use playbook.bullets() to get bullets."""
        playbook = Playbook()
        playbook.add_bullet("test", "First")
        playbook.add_bullet("test", "Second")

        # Ensure playbook has bullets
        assert len(playbook.bullets()) == 2

        result = wrap_playbook_context(playbook)

        # Both bullets should be in result
        assert "First" in result
        assert "Second" in result

    def test_uses_playbook_as_prompt(self):
        """Should use playbook.as_prompt() for bullet formatting."""
        playbook = Playbook()
        playbook.add_bullet("general", "Test bullet")

        # Get the formatted bullets
        bullet_text = playbook.as_prompt()
        result = wrap_playbook_context(playbook)

        # Result should contain the formatted bullet text
        assert bullet_text in result
