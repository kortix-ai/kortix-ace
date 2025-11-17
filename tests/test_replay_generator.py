import unittest

import pytest

from ace import ReplayGenerator, Playbook, Sample


@pytest.mark.unit
class ReplayGeneratorTest(unittest.TestCase):
    """Test ReplayGenerator backward compatibility and new sample-based mode."""

    def test_dict_based_mode_backward_compatibility(self):
        """Test original dict-based mode still works."""
        responses = {"What is 2+2?": "4", "What is Python?": "A programming language"}
        generator = ReplayGenerator(responses)

        # Test successful lookup
        output = generator.generate(
            question="What is 2+2?", context="", playbook=Playbook()
        )

        self.assertEqual(output.final_answer, "4")
        self.assertIn("[Replayed from responses dict]", output.reasoning)
        self.assertEqual(
            output.raw["replay_metadata"]["response_source"], "responses_dict"
        )
        self.assertTrue(output.raw["replay_metadata"]["question_found_in_dict"])

    def test_dict_based_mode_with_default(self):
        """Test fallback to default response when question not found."""
        responses = {"Known question": "Known answer"}
        generator = ReplayGenerator(responses, default_response="I don't know")

        output = generator.generate(
            question="Unknown question", context="", playbook=Playbook()
        )

        self.assertEqual(output.final_answer, "I don't know")
        self.assertEqual(
            output.raw["replay_metadata"]["response_source"], "default_response"
        )
        self.assertFalse(output.raw["replay_metadata"]["question_found_in_dict"])

    def test_sample_based_mode_dict_direct(self):
        """Test new sample-based mode with response in dict."""
        sample = {"question": "What is ACE?", "response": "Agentic Context Engineering"}
        generator = ReplayGenerator()  # No dict needed

        output = generator.generate(
            question=sample["question"], context="", playbook=Playbook(), sample=sample
        )

        self.assertEqual(output.final_answer, "Agentic Context Engineering")
        self.assertEqual(
            output.raw["replay_metadata"]["response_source"], "sample_dict_direct"
        )
        self.assertTrue(output.raw["replay_metadata"]["sample_provided"])

    def test_sample_based_mode_metadata_dict(self):
        """Test sample-based mode with response in metadata dict."""
        sample = {
            "question": "What is the best framework?",
            "metadata": {"response": "ACE Framework"},
        }
        generator = ReplayGenerator()

        output = generator.generate(
            question=sample["question"], context="", playbook=Playbook(), sample=sample
        )

        self.assertEqual(output.final_answer, "ACE Framework")
        self.assertEqual(
            output.raw["replay_metadata"]["response_source"], "sample_dict_metadata"
        )

    def test_sample_based_mode_sample_object(self):
        """Test sample-based mode with Sample dataclass object."""
        sample = Sample(
            question="What is 5+5?",
            ground_truth="10",
            metadata={"response": "The answer is 10"},
        )
        generator = ReplayGenerator()

        output = generator.generate(
            question=sample.question, context="", playbook=Playbook(), sample=sample
        )

        self.assertEqual(output.final_answer, "The answer is 10")
        self.assertEqual(
            output.raw["replay_metadata"]["response_source"], "sample_metadata"
        )

    def test_priority_sample_over_dict(self):
        """Test that sample response takes priority over dict lookup."""
        responses = {"What is 2+2?": "4"}
        generator = ReplayGenerator(responses)

        # Provide both dict and sample - sample should win
        sample = {"question": "What is 2+2?", "response": "5 (from sample)"}

        output = generator.generate(
            question="What is 2+2?", context="", playbook=Playbook(), sample=sample
        )

        self.assertEqual(output.final_answer, "5 (from sample)")
        self.assertEqual(
            output.raw["replay_metadata"]["response_source"], "sample_dict_direct"
        )

    def test_fallback_to_dict_when_sample_has_no_response(self):
        """Test fallback to dict when sample exists but has no response."""
        responses = {"What is 2+2?": "4"}
        generator = ReplayGenerator(responses)

        # Sample without response field
        sample = {"question": "What is 2+2?"}

        output = generator.generate(
            question="What is 2+2?", context="", playbook=Playbook(), sample=sample
        )

        self.assertEqual(output.final_answer, "4")
        self.assertEqual(
            output.raw["replay_metadata"]["response_source"], "responses_dict"
        )

    def test_empty_responses_dict_initialization(self):
        """Test that ReplayGenerator can be initialized without responses dict."""
        generator = ReplayGenerator()

        self.assertIsNotNone(generator.responses)
        self.assertEqual(len(generator.responses), 0)
        self.assertEqual(generator.default_response, "")

    def test_none_responses_dict_initialization(self):
        """Test that ReplayGenerator handles None responses gracefully."""
        generator = ReplayGenerator(responses=None)

        self.assertIsNotNone(generator.responses)
        self.assertEqual(len(generator.responses), 0)


if __name__ == "__main__":
    unittest.main()
