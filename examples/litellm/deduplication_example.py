#!/usr/bin/env python3
"""
ACE Deduplication Example

Demonstrates that deduplication works during learning by:
1. Loading a playbook with known duplicate bullets
2. Running a learning cycle (ask + learn)
3. Verifying that similar bullets were detected and consolidated

Requires: OPENAI_API_KEY (for embeddings), ANTHROPIC_API_KEY (for LLM)
"""

import os
from dotenv import load_dotenv

from ace import ACELiteLLM, Sample, SimpleEnvironment, DeduplicationConfig, Playbook

load_dotenv()


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY for embeddings")
        return

    print("=" * 60)
    print("ACELiteLLM DEDUPLICATION DEMO")
    print("=" * 60)

    # Step 1: Load playbook with known duplicates
    playbook_path = os.path.join(os.path.dirname(__file__), "test_duplicates.json")

    print("\n1. Loading playbook with known duplicates...")
    playbook = Playbook.load_from_file(playbook_path)
    bullets_before = playbook.bullets()

    print(f"   Loaded {len(bullets_before)} bullets:")
    for b in bullets_before:
        print(f"   [{b.section}] {b.content}")

    # Step 2: Configure agent with deduplication
    dedup_config = DeduplicationConfig(
        enabled=True,
        similarity_threshold=0.70,  # Lowered to catch semantic duplicates
        embedding_model="text-embedding-3-small",
    )

    agent = ACELiteLLM(
        model="claude-sonnet-4-5-20250929",
        dedup_config=dedup_config,
        is_learning=True,
    )
    agent.playbook = playbook  # Use our duplicate playbook

    print(f"\n2. Running learning with deduplication enabled...")
    print(f"   - Similarity threshold: {dedup_config.similarity_threshold}")
    print(f"   - Embedding model: {dedup_config.embedding_model}")

    # Step 3: Run learning (this triggers deduplication)
    samples = [
        Sample(question="What is the capital of France?", ground_truth="Paris"),
    ]
    environment = SimpleEnvironment()

    results = agent.learn(samples, environment, epochs=1)

    # Step 4: Check results
    bullets_after = agent.playbook.bullets()

    print(f"\n3. Results:")
    print(f"   - Bullets before: {len(bullets_before)}")
    print(f"   - Bullets after:  {len(bullets_after)}")

    print(f"\n   Current playbook:")
    for b in bullets_after:
        print(f"   [{b.section}] {b.content}")

    # Step 5: Verify
    print("\n" + "=" * 60)
    if len(bullets_after) < len(bullets_before):
        reduction = len(bullets_before) - len(bullets_after)
        print(f"SUCCESS: Deduplication removed {reduction} duplicate bullet(s)")
    elif len(bullets_after) == len(bullets_before):
        print("INFO: No duplicates removed (similarity may be below threshold)")
        print(
            "   This is expected if embeddings find the bullets sufficiently different"
        )
    else:
        print("INFO: Learning added new bullets")
    print("=" * 60)


if __name__ == "__main__":
    main()
