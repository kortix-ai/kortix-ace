"""Example demonstrating playbook save and load functionality."""

import os
import time

from ace import Playbook, Sample, OfflineAdapter, Generator, Reflector, Curator
from ace.adaptation import TaskEnvironment, EnvironmentResult
from ace.llm_providers import LiteLLMClient


class SimpleTaskEnvironment(TaskEnvironment):
    """A simple environment for demonstration."""

    def evaluate(self, sample: Sample, generator_output) -> EnvironmentResult:
        # Simple evaluation: check if answer contains expected keyword
        is_correct = (
            sample.ground_truth.lower() in generator_output.final_answer.lower()
        )

        feedback = "Correct!" if is_correct else f"Expected '{sample.ground_truth}'"

        return EnvironmentResult(feedback=feedback, ground_truth=sample.ground_truth)


def train_and_save_playbook():
    """Train a playbook and save it to file."""
    print("\n" + "=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)

    # Initialize components (v2.1 prompts are now the default)
    client = LiteLLMClient(model="claude-sonnet-4-5-20250929")
    generator = Generator(client)
    reflector = Reflector(client)
    curator = Curator(client)

    # Create offline adapter
    adapter = OfflineAdapter(generator=generator, reflector=reflector, curator=curator)

    # Create training samples - reasoning questions that benefit from learned strategies
    samples = [
        Sample(
            question="A store sells apples for $2 each. If you buy 5 or more, you get 20% off. How much do 7 apples cost?",
            ground_truth="$11.20",
            context="Multi-step: calculate base price, apply discount",
        ),
        Sample(
            question="A train leaves Station A at 9:00 AM traveling at 60 mph. Another train leaves Station B (180 miles away) at 10:00 AM traveling at 90 mph toward Station A. When do they meet?",
            ground_truth="11:00 AM",
            context="Relative motion problem requiring setup of equations",
        ),
        Sample(
            question="If all birds can fly, and penguins are birds, can penguins fly? Explain the flaw.",
            ground_truth="The premise is false - penguins are birds but cannot fly",
            context="Logic and commonsense reasoning",
        ),
        Sample(
            question="You have 3 boxes labeled apples, oranges, mixed. All labels are wrong. Pick one fruit from one box to correctly label all. Which box?",
            ground_truth="Pick from the box labeled 'mixed' - it must contain only one type",
            context="Logic puzzle requiring deductive reasoning",
        ),
    ]

    # Train with 2 epochs
    environment = SimpleTaskEnvironment()
    print(f"\nTraining on {len(samples)} samples for 2 epochs...")

    start = time.time()
    results = adapter.run(samples, environment, epochs=2)
    train_time = time.time() - start

    # Save the trained playbook
    playbook_path = "trained_playbook.json"
    adapter.playbook.save_to_file(playbook_path)

    print(f"\n✅ Training complete in {train_time:.2f}s")
    print(f"   - Samples processed: {len(results)}")
    print(f"   - Strategies learned: {len(adapter.playbook.bullets())}")
    print(f"   - Playbook saved to: {playbook_path}")

    return playbook_path, train_time


def load_and_use_playbook(playbook_path):
    """Load a pre-trained playbook and use it."""
    print("\n" + "=" * 60)
    print("INFERENCE PHASE (using saved playbook)")
    print("=" * 60)

    # Load the saved playbook
    print(f"\nLoading playbook from {playbook_path}...")
    playbook = Playbook.load_from_file(playbook_path)

    print(f"✅ Loaded playbook with {len(playbook.bullets())} strategies")

    # Use the loaded playbook with a new adapter
    client = LiteLLMClient(model="claude-sonnet-4-5-20250929")
    generator = Generator(client)

    # Test with a new reasoning question (similar type to training)
    test_question = "A book costs $15. With a 30% discount, how much do you pay?"
    print(f"\nTest question: {test_question}")

    start = time.time()
    test_output = generator.generate(
        question=test_question,
        context="",
        playbook=playbook,
        reflection=None,
    )
    inference_time = time.time() - start

    print(f"   Answer: {test_output.final_answer}")
    print(f"   Strategies used: {len(test_output.bullet_ids)}")
    print(f"   Inference time: {inference_time:.2f}s")

    return playbook, inference_time


def demonstrate_playbook_inspection(playbook):
    """Show how to inspect a loaded playbook."""
    print("\n" + "=" * 50)
    print("PLAYBOOK INSPECTION")
    print("=" * 50)

    # Print playbook statistics
    stats = playbook.stats()
    print(f"\nPlaybook Statistics:")
    print(f"  - Sections: {stats['sections']}")
    print(f"  - Total bullets: {stats['bullets']}")
    print(f"  - Helpful tags: {stats['tags']['helpful']}")
    print(f"  - Harmful tags: {stats['tags']['harmful']}")
    print(f"  - Neutral tags: {stats['tags']['neutral']}")

    # Show playbook as prompt (first 500 chars)
    prompt_view = playbook.as_prompt()
    if prompt_view:
        print(f"\nPlaybook as prompt (preview):")
        print("-" * 40)
        print(prompt_view[:500] + "..." if len(prompt_view) > 500 else prompt_view)

    # Show individual bullets
    print(f"\nIndividual bullets:")
    for bullet in playbook.bullets()[:3]:  # Show first 3 bullets
        print(f"  [{bullet.id}] {bullet.content[:60]}...")
        print(f"    Helpful: {bullet.helpful}, Harmful: {bullet.harmful}")


if __name__ == "__main__":
    print("=" * 60)
    print("PLAYBOOK PERSISTENCE DEMO")
    print("=" * 60)
    print("\nThis demo shows how to save and load trained playbooks.")
    print("A trained playbook can be reused across sessions.")

    try:
        # Step 1: Train and save a playbook
        playbook_path, train_time = train_and_save_playbook()

        # Step 2: Load and use the saved playbook
        loaded_playbook, inference_time = load_and_use_playbook(playbook_path)

        # Step 3: Inspect the playbook
        demonstrate_playbook_inspection(loaded_playbook)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"\n📊 TIMING:")
        print(f"   - Training time: {train_time:.2f}s")
        print(f"   - Inference time: {inference_time:.2f}s")
        print(f"\n📚 PLAYBOOK:")
        print(f"   - Strategies learned: {len(loaded_playbook.bullets())}")
        print(f"   - Sections: {list(loaded_playbook._sections.keys())}")
        print(f"\n✅ Playbook can now be reused without retraining!")

        # Clean up
        if os.path.exists(playbook_path):
            os.remove(playbook_path)
            print(f"\n(Cleaned up {playbook_path})")

    except ImportError:
        print("Note: This example requires an LLM provider to be configured.")
        print("Set your API key: export OPENAI_API_KEY='your-key'")
    except Exception as e:
        print(f"Example requires API keys to be set: {e}")

        # Demonstrate without API calls
        print("\n" + "=" * 50)
        print("DEMONSTRATING SAVE/LOAD WITHOUT API CALLS")
        print("=" * 50)

        # Create a playbook manually
        playbook = Playbook()
        playbook.add_bullet(
            section="general",
            content="Always provide step-by-step explanations",
            metadata={"helpful": 3, "harmful": 0},
        )
        playbook.add_bullet(
            section="math",
            content="Show your calculations clearly",
            metadata={"helpful": 5, "harmful": 0},
        )

        # Save it
        test_path = "test_playbook.json"
        playbook.save_to_file(test_path)
        print(f"\n✓ Saved playbook to {test_path}")

        # Load it back
        loaded = Playbook.load_from_file(test_path)
        print(f"✓ Loaded playbook with {len(loaded.bullets())} bullets")

        # Verify content matches
        for original, loaded_bullet in zip(playbook.bullets(), loaded.bullets()):
            assert original.content == loaded_bullet.content
            assert original.helpful == loaded_bullet.helpful
        print("✓ Content verified - save/load working correctly!")

        # Show the JSON structure
        print(f"\nJSON structure of saved playbook:")
        with open(test_path, "r") as f:
            print(f.read())

        # Clean up
        # os.remove(test_path)
        print(f"\n✓ Cleaned up {test_path}")
