#!/usr/bin/env python3
"""
Simple LangChain chain example with ACE learning.

This example demonstrates how to wrap a basic LangChain chain with ACELangChain
to add learning capabilities. The chain learns from Q&A tasks and improves
over time.

Requirements:
    pip install ace-framework[langchain]
    # or: pip install langchain-core langchain-anthropic

Environment:
    export ANTHROPIC_API_KEY="your-api-key"
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check availability
try:
    from ace.integrations import ACELangChain, LANGCHAIN_AVAILABLE

    if not LANGCHAIN_AVAILABLE:
        print("LangChain is not installed.")
        print("Install with: pip install ace-framework[langchain]")
        print("or: pip install langchain-core langchain-anthropic")
        exit(1)
except ImportError:
    print("ACE framework not installed.")
    print("Install with: pip install ace-framework")
    exit(1)

from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic


def example_basic_chain():
    """Example 1: Basic LLM chain with ACE learning."""
    print("\n" + "=" * 60)
    print("Example 1: Basic LLM Chain with ACE")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    # Create a simple LangChain chain
    prompt = ChatPromptTemplate.from_template("Answer briefly: {input}")
    llm = ChatAnthropic(temperature=0, model="claude-sonnet-4-5-20250929")
    chain = prompt | llm

    # Wrap with ACE learning
    ace_chain = ACELangChain(
        runnable=chain, ace_model="claude-sonnet-4-5-20250929", is_learning=True
    )

    # Run some questions
    questions = [
        "What is the capital of France?",
        "What is the largest planet in our solar system?",
        "Who painted the Mona Lisa?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        result = ace_chain.invoke({"input": question})
        print(f"Answer: {result.content}")

    # Check learned strategies
    print(f"\nLearned {len(ace_chain.playbook.bullets())} strategies")

    # Save playbook
    ace_chain.save_playbook("simple_chain_learned.json")
    print("\nPlaybook saved to: simple_chain_learned.json")


def example_reuse_playbook():
    """Example 2: Reusing learned strategies from a previous session."""
    print("\n" + "=" * 60)
    print("Example 2: Reusing Learned Strategies")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    # Check if playbook exists
    if not os.path.exists("simple_chain_learned.json"):
        print("No playbook found. Run Example 1 first!")
        return

    # Create chain
    prompt = ChatPromptTemplate.from_template("Answer briefly: {input}")
    llm = ChatAnthropic(temperature=0, model="claude-sonnet-4-5-20250929")
    chain = prompt | llm

    # Load existing playbook
    ace_chain = ACELangChain(
        runnable=chain,
        ace_model="claude-sonnet-4-5-20250929",
        playbook_path="simple_chain_learned.json",
        is_learning=True,
    )

    print(f"Loaded {len(ace_chain.playbook.bullets())} strategies from playbook")

    # Test with a new question
    question = "What is the smallest prime number?"
    print(f"\nQuestion: {question}")
    result = ace_chain.invoke({"input": question})
    print(f"Answer: {result.content}")

    # Show learned strategies
    print("\n" + "-" * 60)
    print("Current Strategies:")
    print("-" * 60)
    for i, bullet in enumerate(ace_chain.playbook.bullets()[:3], 1):
        score = f"+{bullet.helpful}/-{bullet.harmful}"
        print(f"{i}. [{bullet.id}] {bullet.content[:80]}...")
        print(f"   Score: {score}")


def example_learning_control():
    """Example 3: Controlling when learning occurs."""
    print("\n" + "=" * 60)
    print("Example 3: Learning Control (Enable/Disable)")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    # Create chain
    prompt = ChatPromptTemplate.from_template("Answer briefly: {input}")
    llm = ChatAnthropic(temperature=0, model="claude-sonnet-4-5-20250929")
    chain = prompt | llm

    # Start with learning disabled
    ace_chain = ACELangChain(
        runnable=chain, ace_model="claude-sonnet-4-5-20250929", is_learning=False
    )

    print("Initial strategies:", len(ace_chain.playbook.bullets()))

    # Run without learning
    print("\nRunning WITHOUT learning (warm-up phase):")
    result = ace_chain.invoke({"input": "What is 2+2?"})
    print(f"Answer: {result.content}")
    print("Strategies after:", len(ace_chain.playbook.bullets()))

    # Enable learning
    ace_chain.enable_learning()
    print("\nLearning ENABLED")

    # Run with learning
    print("\nRunning WITH learning:")
    result = ace_chain.invoke({"input": "What is 3+3?"})
    print(f"Answer: {result.content}")
    print("Strategies after:", len(ace_chain.playbook.bullets()))


async def example_async_chain():
    """Example 4: Using async chain execution."""
    print("\n" + "=" * 60)
    print("Example 4: Async Chain Execution")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    # Create chain
    prompt = ChatPromptTemplate.from_template("Answer in one sentence: {input}")
    llm = ChatAnthropic(temperature=0.7, model="claude-sonnet-4-5-20250929")
    chain = prompt | llm

    # Wrap with ACE
    ace_chain = ACELangChain(runnable=chain, ace_model="claude-sonnet-4-5-20250929")

    # Async execution
    questions = [
        "What is quantum computing?",
        "Explain machine learning briefly.",
        "What is blockchain technology?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        result = await ace_chain.ainvoke({"input": question})
        print(f"Answer: {result.content}")


def example_custom_output_parser():
    """Example 5: Using a custom output parser."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Output Parser")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    # Custom parser that extracts specific format
    def parse_output(result):
        """Extract just the text content from LangChain message."""
        if hasattr(result, "content"):
            return result.content
        return str(result)

    # Create chain
    prompt = ChatPromptTemplate.from_template(
        "List 3 facts about {topic}. Format as JSON."
    )
    llm = ChatAnthropic(temperature=0, model="claude-sonnet-4-5-20250929")
    chain = prompt | llm

    # Wrap with custom parser
    ace_chain = ACELangChain(
        runnable=chain,
        ace_model="claude-sonnet-4-5-20250929",
        output_parser=parse_output,
    )

    # Test
    result = ace_chain.invoke({"topic": "Python programming"})
    print(f"Result: {result.content[:200]}...")
    print("\nCustom parser used for learning!")


def main():
    """Run all examples."""
    print("=" * 60)
    print("LangChain + ACE Integration Examples")
    print("=" * 60)

    # Run sync examples
    example_basic_chain()
    example_reuse_playbook()
    example_learning_control()
    example_custom_output_parser()

    # Run async example
    import asyncio

    if os.getenv("ANTHROPIC_API_KEY"):
        asyncio.run(example_async_chain())

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
