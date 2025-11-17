# 🧪 ACE Framework Testing Guide

Complete guide for testing ACE agents and validating performance.

## Table of Contents
- [Testing Philosophy](#testing-philosophy)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [LangGraph Testing](#langgraph-testing)
- [Performance Testing](#performance-testing)
- [Testing Without API Calls](#testing-without-api-calls)
- [Continuous Integration](#continuous-integration)

## Testing Philosophy

ACE testing focuses on three key areas:
1. **Correctness**: Does the agent produce accurate answers?
2. **Learning**: Does the playbook improve over time?
3. **Robustness**: Does the system handle edge cases?

## Unit Testing

### Testing Individual Components

```python
import unittest
from ace import Generator, Reflector, Curator
from ace import Playbook, DummyLLMClient

class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.client = DummyLLMClient()
        self.generator = Generator(self.client)
        self.playbook = Playbook()

    def test_generate_with_empty_playbook(self):
        output = self.generator.generate(
            question="What is 2+2?",
            context="Calculate",
            playbook=self.playbook
        )

        self.assertIsNotNone(output.final_answer)
        self.assertIsNotNone(output.reasoning)
        self.assertEqual(len(output.bullet_ids), 0)

    def test_generate_with_strategies(self):
        # Add strategies to playbook
        bullet = self.playbook.add_bullet(
            section="Math",
            content="Break down calculations step by step"
        )

        output = self.generator.generate(
            question="What is 10*5?",
            context="Show work",
            playbook=self.playbook
        )

        self.assertIn(bullet.id, output.bullet_ids)
```

### Testing Playbook Operations

```python
class TestPlaybook(unittest.TestCase):
    def test_add_and_retrieve_bullet(self):
        playbook = Playbook()

        bullet = playbook.add_bullet(
            section="Test Section",
            content="Test strategy"
        )

        retrieved = playbook.get_bullet(bullet.id)
        self.assertEqual(retrieved.content, "Test strategy")

    def test_save_and_load(self):
        playbook = Playbook()
        playbook.add_bullet("Section", "Content")

        # Save
        playbook.save_to_file("test_playbook.json")

        # Load
        loaded = Playbook.load_from_file("test_playbook.json")

        self.assertEqual(len(loaded.bullets()), 1)
```

## Integration Testing

### Testing the Full ACE Loop

```python
from ace import OfflineAdapter, TaskEnvironment, EnvironmentResult, Sample

# Create a simple environment for testing
class SimpleEnvironment(TaskEnvironment):
    """Basic environment for testing - checks if ground truth appears in answer."""

    def evaluate(self, sample, generator_output):
        correct = str(sample.ground_truth).lower() in str(generator_output.final_answer).lower()
        return EnvironmentResult(
            feedback="Correct!" if correct else "Incorrect",
            ground_truth=sample.ground_truth,
        )

class TestACEIntegration(unittest.TestCase):
    def test_offline_learning(self):
        # Setup
        client = DummyLLMClient()
        generator = Generator(client)
        reflector = Reflector(client)
        curator = Curator(client)
        adapter = OfflineAdapter(generator, reflector, curator)
        environment = SimpleEnvironment()

        # Training samples
        samples = [
            Sample("What is 2+2?", "Calculate", "4"),
            Sample("What is 5*3?", "Calculate", "15")
        ]

        # Train
        results = adapter.run(samples, environment, epochs=2)

        # Verify learning
        self.assertGreater(len(adapter.playbook.bullets()), 0)

        # Test on new sample
        test_output = generator.generate(
            question="What is 3+3?",
            context="Calculate",
            playbook=adapter.playbook
        )

        self.assertEqual(test_output.final_answer, "6")
```

### Testing Custom Environments

```python
class TestCustomEnvironment(unittest.TestCase):
    def test_math_environment(self):
        class MathEnvironment(TaskEnvironment):
            def evaluate(self, sample, output):
                try:
                    result = eval(output.final_answer)
                    expected = eval(sample.ground_truth)
                    correct = (result == expected)

                    return EnvironmentResult(
                        feedback="Correct!" if correct else f"Wrong: expected {expected}, got {result}",
                        ground_truth=sample.ground_truth,
                        metrics={"accuracy": 1.0 if correct else 0.0}
                    )
                except:
                    return EnvironmentResult(
                        feedback="Invalid expression",
                        ground_truth=sample.ground_truth,
                        metrics={"accuracy": 0.0}
                    )

        env = MathEnvironment()
        sample = Sample("What is 2+2?", "Calculate", "4")
        output = GeneratorOutput(
            reasoning="2+2=4",
            final_answer="4",
            bullet_ids=[],
            raw={}
        )

        result = env.evaluate(sample, output)
        self.assertEqual(result.metrics["accuracy"], 1.0)
```

## LangGraph Testing

### Setting Up LangGraph Tests

```python
from langchain_core.messages import HumanMessage
from langgraph.graph import Graph, END
from ace import Generator, Reflector, Curator

def create_ace_graph():
    """Create a LangGraph workflow for ACE."""

    # Define the graph
    workflow = Graph()

    # Add nodes
    workflow.add_node("generate", generate_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("curate", curate_node)

    # Add edges
    workflow.add_edge("generate", "reflect")
    workflow.add_edge("reflect", "curate")
    workflow.add_edge("curate", END)

    # Set entry point
    workflow.set_entry_point("generate")

    return workflow.compile()

def generate_node(state):
    """Generator node for LangGraph."""
    generator = state["generator"]
    output = generator.generate(
        question=state["question"],
        context=state["context"],
        playbook=state["playbook"]
    )
    state["generator_output"] = output
    return state

def reflect_node(state):
    """Reflector node for LangGraph."""
    reflector = state["reflector"]
    reflection = reflector.reflect(
        question=state["question"],
        generator_output=state["generator_output"],
        playbook=state["playbook"],
        ground_truth=state.get("ground_truth"),
        feedback=state.get("feedback")
    )
    state["reflection"] = reflection
    return state

def curate_node(state):
    """Curator node for LangGraph."""
    curator = state["curator"]
    curator_output = curator.curate(
        reflection=state["reflection"],
        playbook=state["playbook"],
        question_context=state["context"],
        progress="Testing"
    )
    state["playbook"].apply_delta(curator_output.delta)
    return state
```

### Testing the LangGraph Workflow

```python
class TestLangGraphIntegration(unittest.TestCase):
    def test_ace_workflow(self):
        # Create components
        client = DummyLLMClient()
        generator = Generator(client)
        reflector = Reflector(client)
        curator = Curator(client)
        playbook = Playbook()

        # Create graph
        app = create_ace_graph()

        # Run workflow
        state = {
            "question": "What is 2+2?",
            "context": "Calculate",
            "ground_truth": "4",
            "feedback": "Correct",
            "generator": generator,
            "reflector": reflector,
            "curator": curator,
            "playbook": playbook
        }

        result = app.invoke(state)

        # Verify results
        self.assertIn("generator_output", result)
        self.assertIn("reflection", result)
        self.assertGreater(len(result["playbook"].bullets()), 0)
```

## Performance Testing

### Benchmarking ACE Performance

```python
import time
from statistics import mean, stdev

class PerformanceTester:
    def __init__(self, adapter, environment):
        self.adapter = adapter
        self.environment = environment
        self.metrics = []

    def benchmark(self, samples, epochs=3):
        """Run performance benchmark."""
        start_time = time.time()

        # Track accuracy over epochs
        epoch_accuracies = []

        for epoch in range(epochs):
            correct = 0
            total = 0

            for sample in samples:
                output = self.adapter.generator.generate(
                    question=sample.question,
                    context=sample.context,
                    playbook=self.adapter.playbook
                )

                result = self.environment.evaluate(sample, output)
                if result.metrics.get("accuracy", 0) == 1.0:
                    correct += 1
                total += 1

                # Learn from this sample
                reflection = self.adapter.reflector.reflect(
                    question=sample.question,
                    generator_output=output,
                    playbook=self.adapter.playbook,
                    ground_truth=sample.ground_truth,
                    feedback=result.feedback
                )

                curator_output = self.adapter.curator.curate(
                    reflection=reflection,
                    playbook=self.adapter.playbook,
                    question_context=sample.context,
                    progress=f"{correct}/{total}"
                )

                self.adapter.playbook.apply_delta(curator_output.delta)

            accuracy = correct / total if total > 0 else 0
            epoch_accuracies.append(accuracy)
            print(f"Epoch {epoch + 1}: Accuracy = {accuracy:.2%}")

        end_time = time.time()

        return {
            "epochs": epochs,
            "samples": len(samples),
            "final_accuracy": epoch_accuracies[-1],
            "accuracy_improvement": epoch_accuracies[-1] - epoch_accuracies[0],
            "mean_accuracy": mean(epoch_accuracies),
            "stdev_accuracy": stdev(epoch_accuracies) if len(epoch_accuracies) > 1 else 0,
            "total_time": end_time - start_time,
            "playbook_size": len(self.adapter.playbook.bullets())
        }
```

### Load Testing

```python
import concurrent.futures
import threading

class LoadTester:
    def __init__(self, client):
        self.client = client
        self.lock = threading.Lock()
        self.results = []

    def test_concurrent_requests(self, num_requests=10):
        """Test handling concurrent requests."""

        def make_request(i):
            try:
                start = time.time()
                response = self.client.complete(f"Test request {i}")
                duration = time.time() - start

                with self.lock:
                    self.results.append({
                        "request": i,
                        "duration": duration,
                        "success": True
                    })
            except Exception as e:
                with self.lock:
                    self.results.append({
                        "request": i,
                        "error": str(e),
                        "success": False
                    })

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            concurrent.futures.wait(futures)

        # Calculate metrics
        successful = [r for r in self.results if r["success"]]
        if successful:
            avg_duration = mean([r["duration"] for r in successful])
            success_rate = len(successful) / len(self.results)
        else:
            avg_duration = 0
            success_rate = 0

        return {
            "total_requests": num_requests,
            "successful": len(successful),
            "failed": num_requests - len(successful),
            "success_rate": success_rate,
            "avg_duration": avg_duration
        }
```

## Testing Without API Calls

### Using DummyLLMClient

```python
from ace.llm import DummyLLMClient

class TestWithDummy(unittest.TestCase):
    def setUp(self):
        # Create dummy client with predefined responses
        self.client = DummyLLMClient(
            responses=[
                '{"reasoning": "2+2=4", "final_answer": "4", "bullet_ids": []}',
                '{"reasoning": "Correct", "error_identification": "", "root_cause_analysis": "", "correct_approach": "Direct calculation", "key_insight": "Addition works", "bullet_tags": []}',
                '{"operations": [{"type": "ADD", "section": "Math", "content": "Use direct calculation"}]}'
            ]
        )

    def test_full_cycle(self):
        generator = Generator(self.client)
        reflector = Reflector(self.client)
        curator = Curator(self.client)

        playbook = Playbook()

        # Generate
        gen_output = generator.generate(
            question="What is 2+2?",
            context="Calculate",
            playbook=playbook
        )

        # Reflect
        reflection = reflector.reflect(
            question="What is 2+2?",
            generator_output=gen_output,
            playbook=playbook,
            ground_truth="4",
            feedback="Correct"
        )

        # Curate
        curator_output = curator.curate(
            reflection=reflection,
            playbook=playbook,
            question_context="Math",
            progress="1/1"
        )

        # Apply delta
        playbook.apply_delta(curator_output.delta)

        # Verify
        self.assertEqual(len(playbook.bullets()), 1)
```

### Mocking LLM Responses

```python
from unittest.mock import Mock, patch

class TestWithMocks(unittest.TestCase):
    @patch('ace.llm_providers.litellm_client.completion')
    def test_with_mocked_llm(self, mock_completion):
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"final_answer": "42"}'
        mock_completion.return_value = mock_response

        # Test
        client = LiteLLMClient(model="gpt-4")
        response = client.complete("What is the meaning of life?")

        # Verify
        self.assertIn("42", response.text)
        mock_completion.assert_called_once()
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install uv
        uv sync  # Installs dev dependencies via [dependency-groups]

    - name: Run tests with coverage
      run: |
        uv run pytest  # Respects pyproject.toml config (coverage thresholds, markers)

    - name: Upload coverage reports (Linux only)
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12'
      uses: actions/upload-artifact@v4
      with:
        name: coverage-report
        path: htmlcov/
        retention-days: 30
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
```

## Test Data Management

### Creating Test Datasets

```python
def create_math_dataset():
    """Create a dataset for testing math capabilities."""
    return [
        Sample("What is 2+2?", "Calculate", "4"),
        Sample("What is 10*5?", "Calculate", "50"),
        Sample("What is 100/4?", "Calculate", "25"),
        Sample("What is 7-3?", "Calculate", "4"),
        Sample("What is 2^3?", "Calculate", "8"),
    ]

def create_qa_dataset():
    """Create a dataset for testing Q&A capabilities."""
    return [
        Sample(
            "What is the capital of France?",
            "Provide city name only",
            "Paris"
        ),
        Sample(
            "Who wrote Romeo and Juliet?",
            "Provide author name",
            "William Shakespeare"
        ),
        Sample(
            "What year did WW2 end?",
            "Provide year only",
            "1945"
        ),
    ]
```

### Validation Metrics

```python
class MetricsCalculator:
    @staticmethod
    def calculate_metrics(results):
        """Calculate comprehensive metrics from test results."""

        accuracies = [r.get("accuracy", 0) for r in results]

        return {
            "mean_accuracy": mean(accuracies),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "stdev_accuracy": stdev(accuracies) if len(accuracies) > 1 else 0,
            "total_samples": len(results),
            "perfect_scores": sum(1 for a in accuracies if a == 1.0),
            "failures": sum(1 for a in accuracies if a == 0.0),
        }
```

## Best Practices

1. **Always test with DummyLLMClient first** - Validate logic without API costs
2. **Use fixtures for common setup** - Reduce code duplication
3. **Test edge cases** - Empty playbooks, invalid inputs, etc.
4. **Monitor performance metrics** - Track accuracy improvements
5. **Use continuous integration** - Catch issues early
6. **Test different model providers** - Ensure compatibility
7. **Create domain-specific test sets** - Math, coding, Q&A, etc.
8. **Version your test data** - Track what changes affect performance

## Next Steps

- Review [API Reference](API_REFERENCE.md)
- Explore [example tests](../tests/)
- Set up [CI/CD pipeline](#continuous-integration)
- Join our [testing community](#)