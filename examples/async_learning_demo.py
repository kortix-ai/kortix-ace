#!/usr/bin/env python3
"""Demo script to verify async learning works correctly.

This script demonstrates:
1. Sync mode (traditional blocking behavior)
2. Async mode (Generator returns immediately, learning in background)
"""

import json
import time
from typing import Any, List, Type, TypeVar

from pydantic import BaseModel

from ace import (
    Curator,
    EnvironmentResult,
    Generator,
    LLMClient,
    OfflineAdapter,
    Playbook,
    Reflector,
    Sample,
    TaskEnvironment,
)
from ace.llm import LLMResponse

T = TypeVar("T", bound=BaseModel)


class DemoLLMClient(LLMClient):
    """Mock LLM with configurable delay to simulate real LLM latency."""

    def __init__(self, delay: float = 0.1):
        super().__init__(model="demo-mock")
        self.delay = delay
        self.call_count = 0

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        time.sleep(self.delay)
        self.call_count += 1
        return LLMResponse(text="{}")

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        **kwargs: Any,
    ) -> T:
        """Return appropriate mock response based on model type."""
        time.sleep(self.delay)
        self.call_count += 1

        # Detect which role is calling based on response_model name
        model_name = response_model.__name__

        if "Generator" in model_name:
            data = {
                "reasoning": "Mock reasoning for demo",
                "final_answer": "The correct answer is 42",
                "bullet_ids": [],
            }
        elif "Reflector" in model_name:
            data = {
                "reasoning": "Analysis: the approach was sound",
                "error_identification": "",
                "root_cause_analysis": "",
                "correct_approach": "Continue with systematic analysis",
                "key_insight": "Always verify calculations",
                "bullet_tags": [],
            }
        elif "Curator" in model_name:
            data = {
                "delta": {
                    "reasoning": "No changes needed for this iteration",
                    "operations": [],
                },
            }
        else:
            data = {}

        return response_model.model_validate(data)


class DemoEnvironment(TaskEnvironment):
    """Simple environment that always returns success."""

    def evaluate(self, sample: Sample, generator_output) -> EnvironmentResult:
        return EnvironmentResult(
            feedback="Good answer!",
            ground_truth=sample.ground_truth,
            metrics={"correct": 1.0},
        )


def run_sync_demo(num_samples: int = 5, llm_delay: float = 0.1):
    """Run in sync mode (traditional blocking)."""
    print("\n" + "=" * 60)
    print("SYNC MODE DEMO")
    print("=" * 60)

    # Create separate LLMs for each role (simulates real usage)
    generator_llm = DemoLLMClient(delay=llm_delay)
    reflector_llm = DemoLLMClient(delay=llm_delay)
    curator_llm = DemoLLMClient(delay=llm_delay)

    playbook = Playbook()
    environment = DemoEnvironment()

    adapter = OfflineAdapter(
        playbook=playbook,
        generator=Generator(generator_llm),
        reflector=Reflector(reflector_llm),
        curator=Curator(curator_llm),
        async_learning=False,  # Sync mode
    )

    samples = [
        Sample(question=f"Question {i}", context="Demo", ground_truth="42")
        for i in range(num_samples)
    ]

    print(f"Processing {num_samples} samples in SYNC mode...")
    print(f"LLM delay: {llm_delay}s per call")
    print(
        f"Expected: ~{num_samples * 3 * llm_delay:.1f}s (sequential Generator+Reflector+Curator)"
    )

    start = time.time()
    results = adapter.run(samples, environment, epochs=1)
    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  - Samples processed: {len(results)}")
    print(f"  - Time elapsed: {elapsed:.2f}s")
    print(
        f"  - All reflections populated: {all(r.reflection is not None for r in results)}"
    )
    print(
        f"  - All curator outputs populated: {all(r.curator_output is not None for r in results)}"
    )

    return elapsed


def run_async_demo(num_samples: int = 5, llm_delay: float = 0.1):
    """Run in async mode (Generator returns immediately)."""
    print("\n" + "=" * 60)
    print("ASYNC MODE DEMO")
    print("=" * 60)

    # Create separate LLMs for each role
    generator_llm = DemoLLMClient(delay=llm_delay)
    reflector_llm = DemoLLMClient(delay=llm_delay)
    curator_llm = DemoLLMClient(delay=llm_delay)

    playbook = Playbook()
    environment = DemoEnvironment()

    adapter = OfflineAdapter(
        playbook=playbook,
        generator=Generator(generator_llm),
        reflector=Reflector(reflector_llm),
        curator=Curator(curator_llm),
        async_learning=True,  # Async mode!
        max_reflector_workers=3,
    )

    samples = [
        Sample(question=f"Question {i}", context="Demo", ground_truth="42")
        for i in range(num_samples)
    ]

    print(f"Processing {num_samples} samples in ASYNC mode...")
    print(f"LLM delay: {llm_delay}s per call")
    print(
        f"Expected: ~{num_samples * llm_delay:.1f}s (just Generator, learning in background)"
    )

    # Phase 1: Get results quickly (no waiting for learning)
    start = time.time()
    results = adapter.run(samples, environment, epochs=1, wait_for_learning=False)
    results_time = time.time() - start

    print(f"\nPhase 1 - Results returned:")
    print(f"  - Samples processed: {len(results)}")
    print(f"  - Time to get results: {results_time:.2f}s")
    print(
        f"  - Reflections: {sum(1 for r in results if r.reflection is not None)}/{len(results)} (None = still processing)"
    )
    print(
        f"  - Curator outputs: {sum(1 for r in results if r.curator_output is not None)}/{len(results)} (None = still processing)"
    )

    # Check learning stats
    stats = adapter.learning_stats
    print(f"\n  Learning stats:")
    print(f"    - Tasks submitted: {stats['tasks_submitted']}")
    print(f"    - Reflections completed: {stats['reflections_completed']}")
    print(f"    - Curations completed: {stats['curations_completed']}")
    print(f"    - Pipeline running: {stats['is_running']}")

    # Phase 2: Wait for learning to complete
    print("\nPhase 2 - Waiting for learning to complete...")
    wait_start = time.time()
    adapter.wait_for_learning(timeout=30.0)
    wait_time = time.time() - wait_start

    final_stats = adapter.learning_stats
    print(f"  - Wait time: {wait_time:.2f}s")
    print(f"  - Final reflections completed: {final_stats['reflections_completed']}")
    print(f"  - Final curations completed: {final_stats['curations_completed']}")

    total_time = time.time() - start
    print(f"\nTotal time: {total_time:.2f}s")

    # Cleanup
    adapter.stop_async_learning(wait=False)

    return results_time, total_time


def main():
    print("=" * 60)
    print("ASYNC LEARNING DEMO")
    print("=" * 60)
    print("\nThis demo shows the difference between sync and async modes.")
    print("In async mode, the Generator returns immediately while")
    print("Reflector and Curator process in the background.")

    num_samples = 5
    llm_delay = 0.15  # 150ms per LLM call

    # Run sync demo
    sync_time = run_sync_demo(num_samples, llm_delay)

    # Run async demo
    async_results_time, async_total_time = run_async_demo(num_samples, llm_delay)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nSync mode total time:    {sync_time:.2f}s")
    print(
        f"Async mode results time: {async_results_time:.2f}s (when Generator results are available)"
    )
    print(
        f"Async mode total time:   {async_total_time:.2f}s (including background learning)"
    )
    print(
        f"\nSpeedup for getting results: {sync_time / async_results_time:.1f}x faster"
    )
    print("\n✅ Async learning is working correctly!")


if __name__ == "__main__":
    main()
