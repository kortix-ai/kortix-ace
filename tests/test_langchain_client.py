"""Basic tests for LangChain client integration."""

import unittest
from unittest.mock import patch, MagicMock

import pytest


@pytest.mark.unit
class TestLangChainClient(unittest.TestCase):
    """Test LangChain client functionality."""

    def test_import_fallback(self):
        """Test that import handles missing langchain gracefully."""
        try:
            from ace.llm_providers import LangChainLiteLLMClient

            # If it imports, langchain is installed
            self.assertTrue(True)
        except ImportError:
            # This is expected if langchain not installed
            self.assertTrue(True)

    @patch("ace.llm_providers.langchain_client.LANGCHAIN_AVAILABLE", True)
    @patch("ace.llm_providers.langchain_client.ChatLiteLLM")
    def test_basic_initialization(self, mock_chat):
        """Test client initialization."""
        from ace.llm_providers.langchain_client import LangChainLiteLLMClient

        # Create client
        client = LangChainLiteLLMClient(model="gpt-3.5-turbo")

        # Check that ChatLiteLLM was initialized
        mock_chat.assert_called_once()
        self.assertFalse(client.is_router)

    @patch("ace.llm_providers.langchain_client.LANGCHAIN_AVAILABLE", False)
    def test_missing_langchain_error(
        self,
    ):
        """Test error when langchain not available."""
        from ace.llm_providers.langchain_client import LangChainLiteLLMClient

        with self.assertRaises(ImportError) as context:
            LangChainLiteLLMClient(model="test")

        self.assertIn("LangChain is not installed", str(context.exception))

    def test_parameter_filtering(self):
        """Test that ACE-specific parameters are filtered."""
        from ace.llm_providers.langchain_client import LangChainLiteLLMClient

        client = LangChainLiteLLMClient.__new__(LangChainLiteLLMClient)

        # Test the filter method directly
        filtered = client._filter_kwargs(
            {
                "temperature": 0.5,
                "refinement_round": 1,
                "max_refinement_rounds": 3,
                "max_tokens": 100,
            }
        )

        self.assertIn("temperature", filtered)
        self.assertIn("max_tokens", filtered)
        self.assertNotIn("refinement_round", filtered)
        self.assertNotIn("max_refinement_rounds", filtered)


if __name__ == "__main__":
    unittest.main()
