"""
Tests for async learning infrastructure.

Tests for ThreadSafePlaybook, AsyncLearningPipeline, and adapter async mode.
"""

import json
import threading
import time
import unittest
from typing import List

import pytest

from ace import (
    Curator,
    EnvironmentResult,
    Generator,
    OfflineAdapter,
    OnlineAdapter,
    Playbook,
    Reflector,
    Sample,
    TaskEnvironment,
)
from ace.async_learning import (
    AsyncLearningPipeline,
    LearningTask,
    ReflectionResult,
    ThreadSafePlaybook,
)
from ace.delta import DeltaBatch
from ace.roles import BulletTag, GeneratorOutput, ReflectorOutput, CuratorOutput

# Import MockLLMClient from conftest
from tests.conftest import MockLLMClient


# ---------------------------------------------------------------------------
# Test Response Helpers
# ---------------------------------------------------------------------------


def make_generator_response(answer: str = "correct answer") -> str:
    """Create a valid Generator JSON response."""
    return json.dumps(
        {
            "reasoning": "Test reasoning",
            "final_answer": answer,
            "bullet_ids": [],
        }
    )


def make_reflector_response() -> str:
    """Create a valid Reflector JSON response."""
    return json.dumps(
        {
            "reasoning": "Test reflection reasoning",
            "error_identification": "",
            "root_cause_analysis": "",
            "correct_approach": "The approach was correct",
            "key_insight": "Always verify the answer",
            "bullet_tags": [],
        }
    )


def make_curator_response() -> str:
    """Create a valid Curator JSON response with empty deltas."""
    return json.dumps(
        {
            "delta": {"reasoning": "No changes needed", "operations": []},
        }
    )


class SimpleTestEnvironment(TaskEnvironment):
    """Simple environment for testing."""

    def evaluate(self, sample: Sample, generator_output) -> EnvironmentResult:
        """Evaluate if answer contains 'correct'."""
        answer = generator_output.final_answer
        success = "correct" in answer.lower()
        feedback = "✓ Contains 'correct'" if success else "✗ Missing 'correct'"

        return EnvironmentResult(
            feedback=feedback,
            ground_truth="The answer should contain 'correct'",
            metrics={"success": success},
        )


# ---------------------------------------------------------------------------
# ThreadSafePlaybook Tests
# ---------------------------------------------------------------------------


class TestThreadSafePlaybook(unittest.TestCase):
    """Test thread-safe playbook wrapper."""

    def test_lock_free_reads(self):
        """Test that reads work without blocking."""
        playbook = Playbook()
        playbook.add_bullet("Test", "Test content", bullet_id="b1")
        ts_playbook = ThreadSafePlaybook(playbook)

        # Reads should work
        self.assertIn("Test content", ts_playbook.as_prompt())
        self.assertEqual(len(ts_playbook.bullets()), 1)
        self.assertIsNotNone(ts_playbook.get_bullet("b1"))
        # Check actual stats keys
        stats = ts_playbook.stats()
        self.assertIn("bullets", stats)

    def test_locked_writes(self):
        """Test that writes are thread-safe."""
        playbook = Playbook()
        ts_playbook = ThreadSafePlaybook(playbook)

        # Add bullet through thread-safe wrapper
        ts_playbook.add_bullet("Test", "Content 1", bullet_id="b1")
        self.assertEqual(len(ts_playbook.bullets()), 1)

        # Update bullet
        ts_playbook.update_bullet("b1", content="Updated content")
        self.assertEqual(ts_playbook.get_bullet("b1").content, "Updated content")

        # Tag bullet
        ts_playbook.tag_bullet("b1", "helpful")
        self.assertEqual(ts_playbook.get_bullet("b1").helpful, 1)

        # Remove bullet
        ts_playbook.remove_bullet("b1")
        self.assertEqual(len(ts_playbook.bullets()), 0)

    def test_concurrent_writes(self):
        """Test that concurrent writes don't cause race conditions."""
        playbook = Playbook()
        playbook.add_bullet("Test", "Concurrent test", bullet_id="b1")
        ts_playbook = ThreadSafePlaybook(playbook)

        num_threads = 10
        increments_per_thread = 100
        errors: List[Exception] = []

        def increment_tags():
            try:
                for _ in range(increments_per_thread):
                    ts_playbook.tag_bullet("b1", "helpful")
            except Exception as e:
                errors.append(e)

        # Run concurrent increments
        threads = [threading.Thread(target=increment_tags) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        self.assertEqual(len(errors), 0)

        # Final count should be correct
        bullet = ts_playbook.get_bullet("b1")
        expected_count = num_threads * increments_per_thread
        self.assertEqual(bullet.helpful, expected_count)


# ---------------------------------------------------------------------------
# AsyncLearningPipeline Tests
# ---------------------------------------------------------------------------


class TestAsyncLearningPipeline(unittest.TestCase):
    """Test async learning pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.playbook = Playbook()

    def _create_mock_llm(self, responses: List[str]) -> MockLLMClient:
        """Create MockLLMClient with queued responses."""
        llm = MockLLMClient()
        llm.set_responses(responses)
        return llm

    def test_start_stop_lifecycle(self):
        """Test pipeline start/stop lifecycle."""
        # Use separate LLMs for each role
        reflector_llm = self._create_mock_llm([])
        curator_llm = self._create_mock_llm([])

        pipeline = AsyncLearningPipeline(
            playbook=self.playbook,
            reflector=Reflector(reflector_llm),
            curator=Curator(curator_llm),
        )

        # Not running initially
        self.assertFalse(pipeline.is_running())

        # Start
        pipeline.start()
        self.assertTrue(pipeline.is_running())

        # Double start should be safe
        pipeline.start()
        self.assertTrue(pipeline.is_running())

        # Stop
        remaining = pipeline.stop(wait=True, timeout=5.0)
        self.assertFalse(pipeline.is_running())
        self.assertEqual(remaining, 0)

    def test_submit_before_start(self):
        """Test that submit returns None before pipeline starts."""
        reflector_llm = self._create_mock_llm([])
        curator_llm = self._create_mock_llm([])

        pipeline = AsyncLearningPipeline(
            playbook=self.playbook,
            reflector=Reflector(reflector_llm),
            curator=Curator(curator_llm),
        )

        task = self._create_dummy_task()
        result = pipeline.submit(task)
        self.assertIsNone(result)

    def test_submit_and_process(self):
        """Test submitting and processing a task."""
        # Create separate LLMs for each role
        reflector_llm = self._create_mock_llm([make_reflector_response()])
        curator_llm = self._create_mock_llm([make_curator_response()])

        pipeline = AsyncLearningPipeline(
            playbook=self.playbook,
            reflector=Reflector(reflector_llm),
            curator=Curator(curator_llm),
            max_reflector_workers=2,
        )

        pipeline.start()
        try:
            task = self._create_dummy_task()
            future = pipeline.submit(task)

            self.assertIsNotNone(future)

            # Wait for completion
            completed = pipeline.wait_for_completion(timeout=10.0)
            self.assertTrue(completed)

            # Check stats
            stats = pipeline.stats
            self.assertEqual(stats["tasks_submitted"], 1)
            self.assertEqual(stats["reflections_completed"], 1)
            self.assertEqual(stats["curations_completed"], 1)
            self.assertEqual(stats["tasks_failed"], 0)
        finally:
            pipeline.stop(wait=False)

    def test_multiple_tasks(self):
        """Test processing multiple tasks."""
        # Create separate LLMs with 3 responses each
        reflector_llm = self._create_mock_llm(
            [make_reflector_response() for _ in range(3)]
        )
        curator_llm = self._create_mock_llm([make_curator_response() for _ in range(3)])

        pipeline = AsyncLearningPipeline(
            playbook=self.playbook,
            reflector=Reflector(reflector_llm),
            curator=Curator(curator_llm),
            max_reflector_workers=3,
        )

        pipeline.start()
        try:
            # Submit multiple tasks
            for i in range(3):
                task = self._create_dummy_task(i)
                pipeline.submit(task)

            pipeline.wait_for_completion(timeout=15.0)

            # All should be processed
            stats = pipeline.stats
            self.assertEqual(stats["tasks_submitted"], 3)
            self.assertEqual(stats["reflections_completed"], 3)
            self.assertEqual(stats["curations_completed"], 3)
        finally:
            pipeline.stop(wait=False)

    def test_completion_callback(self):
        """Test completion callback invocation."""
        completions: List[tuple] = []

        def on_complete(task, curator_output):
            completions.append((task, curator_output))

        reflector_llm = self._create_mock_llm([make_reflector_response()])
        curator_llm = self._create_mock_llm([make_curator_response()])

        pipeline = AsyncLearningPipeline(
            playbook=self.playbook,
            reflector=Reflector(reflector_llm),
            curator=Curator(curator_llm),
            on_complete=on_complete,
        )

        pipeline.start()
        try:
            task = self._create_dummy_task()
            pipeline.submit(task)

            pipeline.wait_for_completion(timeout=10.0)

            # Completion callback should be invoked
            self.assertEqual(len(completions), 1)
            self.assertEqual(completions[0][0], task)
        finally:
            pipeline.stop(wait=False)

    def _create_dummy_task(self, step_index: int = 0) -> LearningTask:
        """Create a dummy learning task for testing."""
        sample = Sample(
            question=f"Test question {step_index}",
            context="Test context",
            ground_truth="correct",
        )
        generator_output = GeneratorOutput(
            reasoning="Test reasoning",
            final_answer="Test answer correct",
            bullet_ids=[],
        )
        env_result = EnvironmentResult(
            feedback="Test feedback",
            ground_truth="correct",
            metrics={"success": True},
        )
        return LearningTask(
            sample=sample,
            generator_output=generator_output,
            environment_result=env_result,
            epoch=1,
            step_index=step_index,
            total_epochs=1,
            total_steps=1,
        )


# ---------------------------------------------------------------------------
# Adapter Async Mode Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestOfflineAdapterAsyncMode(unittest.TestCase):
    """Test OfflineAdapter with async learning mode."""

    def setUp(self):
        """Set up test fixtures."""
        self.playbook = Playbook()
        self.environment = SimpleTestEnvironment()

    def _create_mock_llm(self, responses: List[str]) -> MockLLMClient:
        """Create MockLLMClient with queued responses."""
        llm = MockLLMClient()
        llm.set_responses(responses)
        return llm

    def test_sync_mode_unchanged(self):
        """Test that sync mode still works as before."""
        # Each role gets its own LLM with appropriate responses
        generator_llm = self._create_mock_llm([make_generator_response()])
        reflector_llm = self._create_mock_llm([make_reflector_response()])
        curator_llm = self._create_mock_llm([make_curator_response()])

        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=Generator(generator_llm),
            reflector=Reflector(reflector_llm),
            curator=Curator(curator_llm),
            async_learning=False,  # Sync mode (default)
        )

        samples = [Sample(question="What is 2+2?", context="Math", ground_truth="4")]

        results = adapter.run(samples, self.environment, epochs=1)

        # Verify results have all fields populated (sync mode)
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].generator_output)
        self.assertIsNotNone(results[0].reflection)  # Populated in sync mode
        self.assertIsNotNone(results[0].curator_output)  # Populated in sync mode

    def test_async_mode_basic(self):
        """Test async mode returns results with None reflection/curator."""
        # Each role gets its own LLM with appropriate responses
        generator_llm = self._create_mock_llm(
            [make_generator_response() for _ in range(3)]
        )
        reflector_llm = self._create_mock_llm(
            [make_reflector_response() for _ in range(3)]
        )
        curator_llm = self._create_mock_llm([make_curator_response() for _ in range(3)])

        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=Generator(generator_llm),
            reflector=Reflector(reflector_llm),
            curator=Curator(curator_llm),
            async_learning=True,  # Async mode
        )

        samples = [
            Sample(question=f"Q{i}", context="", ground_truth="correct")
            for i in range(3)
        ]

        results = adapter.run(
            samples,
            self.environment,
            epochs=1,
            wait_for_learning=False,  # Don't wait
        )

        # Results should be returned
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsNotNone(result.generator_output)
            # In async mode, these are None (processing in background)
            self.assertIsNone(result.reflection)
            self.assertIsNone(result.curator_output)

        # Clean up async pipeline
        adapter.stop_async_learning(wait=False)

    def test_async_mode_with_wait(self):
        """Test async mode that waits for learning completion."""
        generator_llm = self._create_mock_llm(
            [make_generator_response() for _ in range(3)]
        )
        reflector_llm = self._create_mock_llm(
            [make_reflector_response() for _ in range(3)]
        )
        curator_llm = self._create_mock_llm([make_curator_response() for _ in range(3)])

        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=Generator(generator_llm),
            reflector=Reflector(reflector_llm),
            curator=Curator(curator_llm),
            async_learning=True,
        )

        samples = [
            Sample(question=f"Q{i}", context="", ground_truth="correct")
            for i in range(3)
        ]

        results = adapter.run(
            samples,
            self.environment,
            epochs=1,
            wait_for_learning=True,  # Wait for completion
        )

        self.assertEqual(len(results), 3)

        # Check learning stats
        stats = adapter.learning_stats
        self.assertEqual(stats["tasks_submitted"], 3)
        self.assertEqual(stats["curations_completed"], 3)

    def test_wait_for_learning_method(self):
        """Test explicit wait_for_learning method."""
        generator_llm = self._create_mock_llm([make_generator_response()])
        reflector_llm = self._create_mock_llm([make_reflector_response()])
        curator_llm = self._create_mock_llm([make_curator_response()])

        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=Generator(generator_llm),
            reflector=Reflector(reflector_llm),
            curator=Curator(curator_llm),
            async_learning=True,
        )

        samples = [Sample(question="Q1", context="", ground_truth="correct")]

        # Run without waiting
        adapter.run(samples, self.environment, epochs=1, wait_for_learning=False)

        # Explicitly wait
        completed = adapter.wait_for_learning(timeout=10.0)
        self.assertTrue(completed)

        # Now stats should show completion
        stats = adapter.learning_stats
        self.assertEqual(stats["curations_completed"], 1)

        # Clean up
        adapter.stop_async_learning(wait=False)

    def test_learning_stats_property(self):
        """Test learning_stats property."""
        generator_llm = self._create_mock_llm([make_generator_response()])
        reflector_llm = self._create_mock_llm([make_reflector_response()])
        curator_llm = self._create_mock_llm([make_curator_response()])

        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=Generator(generator_llm),
            reflector=Reflector(reflector_llm),
            curator=Curator(curator_llm),
            async_learning=True,
        )

        # Before starting, stats should show defaults
        stats = adapter.learning_stats
        self.assertEqual(stats["tasks_submitted"], 0)
        self.assertFalse(stats["is_running"])

        samples = [Sample(question="Q1", context="", ground_truth="correct")]
        adapter.run(samples, self.environment, epochs=1, wait_for_learning=True)

        # After running, stats should be updated
        stats = adapter.learning_stats
        self.assertGreater(stats["tasks_submitted"], 0)


@pytest.mark.integration
class TestOnlineAdapterAsyncMode(unittest.TestCase):
    """Test OnlineAdapter with async learning mode."""

    def setUp(self):
        """Set up test fixtures."""
        self.playbook = Playbook()
        self.environment = SimpleTestEnvironment()

    def _create_mock_llm(self, responses: List[str]) -> MockLLMClient:
        """Create MockLLMClient with queued responses."""
        llm = MockLLMClient()
        llm.set_responses(responses)
        return llm

    def test_online_async_mode(self):
        """Test OnlineAdapter async mode."""
        generator_llm = self._create_mock_llm(
            [make_generator_response() for _ in range(3)]
        )
        reflector_llm = self._create_mock_llm(
            [make_reflector_response() for _ in range(3)]
        )
        curator_llm = self._create_mock_llm([make_curator_response() for _ in range(3)])

        adapter = OnlineAdapter(
            playbook=self.playbook,
            generator=Generator(generator_llm),
            reflector=Reflector(reflector_llm),
            curator=Curator(curator_llm),
            async_learning=True,
        )

        samples = [
            Sample(question=f"Q{i}", context="", ground_truth="correct")
            for i in range(3)
        ]

        results = adapter.run(samples, self.environment, wait_for_learning=True)

        self.assertEqual(len(results), 3)

        stats = adapter.learning_stats
        self.assertEqual(stats["tasks_submitted"], 3)


# ---------------------------------------------------------------------------
# Data Classes Tests
# ---------------------------------------------------------------------------


class TestLearningTask(unittest.TestCase):
    """Test LearningTask dataclass."""

    def test_creation_with_defaults(self):
        """Test LearningTask creation with default values."""
        sample = Sample(question="Q", context="C", ground_truth="A")
        gen_output = GeneratorOutput(reasoning="R", final_answer="A", bullet_ids=[])
        env_result = EnvironmentResult(feedback="F", ground_truth="A", metrics={})

        task = LearningTask(
            sample=sample,
            generator_output=gen_output,
            environment_result=env_result,
            epoch=1,
            step_index=0,
        )

        self.assertEqual(task.epoch, 1)
        self.assertEqual(task.step_index, 0)
        self.assertEqual(task.total_epochs, 1)  # Default
        self.assertEqual(task.total_steps, 1)  # Default
        self.assertIsNotNone(task.timestamp)
        self.assertEqual(task.metadata, {})


class TestReflectionResult(unittest.TestCase):
    """Test ReflectionResult dataclass."""

    def test_creation(self):
        """Test ReflectionResult creation."""
        sample = Sample(question="Q", context="C", ground_truth="A")
        gen_output = GeneratorOutput(reasoning="R", final_answer="A", bullet_ids=[])
        env_result = EnvironmentResult(feedback="F", ground_truth="A", metrics={})
        task = LearningTask(
            sample=sample,
            generator_output=gen_output,
            environment_result=env_result,
            epoch=1,
            step_index=0,
        )
        # Create proper ReflectorOutput with all required fields
        reflection = ReflectorOutput(
            reasoning="Analysis reasoning",
            correct_approach="The correct approach",
            key_insight="Key insight learned",
            bullet_tags=[],
        )

        result = ReflectionResult(task=task, reflection=reflection)

        self.assertEqual(result.task, task)
        self.assertEqual(result.reflection, reflection)
        self.assertIsNotNone(result.timestamp)


if __name__ == "__main__":
    unittest.main()
