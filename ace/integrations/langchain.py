"""
ACE + LangChain integration for learning from chain/agent execution.

This module provides ACELangChain, a wrapper that adds ACE learning capabilities
to any LangChain Runnable (chains, agents, custom runnables).

When to Use ACELangChain:
- Complex workflows: Multi-step LangChain chains
- Tool-using agents: LangChain agents with tools
- Custom runnables: Your own LangChain components
- Production workflows: LangChain orchestration with learning

When NOT to Use ACELangChain:
- Simple Q&A → Use ACELiteLLM
- Browser automation → Use ACEAgent (browser-use)
- Custom agent (non-LangChain) → Use integration pattern (see docs)

Example:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from ace.integrations import ACELangChain

    # Create your LangChain chain
    prompt = PromptTemplate.from_template("Answer: {question}")
    llm = ChatOpenAI(temperature=0)
    chain = prompt | llm

    # Wrap with ACE learning
    ace_chain = ACELangChain(runnable=chain)

    # Use it (learns automatically)
    result = ace_chain.invoke({"question": "What is ACE?"})

    # Save learned strategies
    ace_chain.save_playbook("chain_expert.json")
"""

from typing import TYPE_CHECKING, Any, Optional, Dict, Callable, List
import asyncio
import logging

from ..playbook import Playbook
from ..roles import Reflector, Curator, GeneratorOutput
from ..prompts_v2_1 import PromptManager
from .base import wrap_playbook_context

if TYPE_CHECKING:
    from ..deduplication import DeduplicationConfig

try:
    from langchain_core.runnables import Runnable

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Runnable = None  # type: ignore

# Try to import AgentExecutor for rich tracing
try:
    from langchain.agents import AgentExecutor

    AGENT_EXECUTOR_AVAILABLE = True
except ImportError:
    AGENT_EXECUTOR_AVAILABLE = False
    AgentExecutor = None  # type: ignore

logger = logging.getLogger(__name__)


class ACELangChain:
    """
    LangChain Runnable wrapper with ACE learning.

    Wraps any LangChain Runnable (chain, agent, custom) and adds ACE learning
    capabilities. The runnable executes normally, but ACE learns from results
    to improve future executions.

    Pattern:
        1. INJECT: Add playbook context to input
        2. EXECUTE: Run the LangChain runnable
        3. LEARN: Update playbook via Reflector + Curator

    Attributes:
        runnable: The LangChain Runnable being wrapped
        playbook: Learned strategies (Playbook instance)
        is_learning: Whether learning is enabled

    Example:
        # Basic usage
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template("Answer: {input}")
        chain = prompt | ChatOpenAI(temperature=0)

        ace_chain = ACELangChain(runnable=chain)
        result = ace_chain.invoke({"input": "What is 2+2?"})

        # With existing playbook
        ace_chain = ACELangChain(
            runnable=chain,
            playbook_path="expert.json"
        )

        # Async execution
        result = await ace_chain.ainvoke({"input": "Question"})
    """

    def __init__(
        self,
        runnable: Any,
        ace_model: str = "gpt-4o-mini",
        playbook_path: Optional[str] = None,
        is_learning: bool = True,
        async_learning: bool = False,
        output_parser: Optional[Callable[[Any], str]] = None,
        dedup_config: Optional["DeduplicationConfig"] = None,
    ):
        """
        Initialize ACELangChain wrapper.

        Args:
            runnable: LangChain Runnable (chain, agent, custom)
            ace_model: Model for ACE learning (Reflector/Curator)
            playbook_path: Path to existing playbook (optional)
            is_learning: Enable/disable learning (default: True)
            async_learning: Run learning in background for ainvoke() (default: False)
                           When True, ainvoke() returns immediately while
                           Reflector/Curator process in background.
            output_parser: Custom function to parse runnable output to string
                          (default: converts to string)
            dedup_config: Optional DeduplicationConfig for bullet deduplication

        Raises:
            ImportError: If LangChain is not installed

        Example:
            # Basic
            ace_chain = ACELangChain(my_chain)

            # With async learning
            ace_chain = ACELangChain(my_chain, async_learning=True)
            result = await ace_chain.ainvoke(input)
            # Result returns immediately, learning continues in background
            await ace_chain.wait_for_learning()

            # With custom output parser
            def parse_output(result):
                return result["output"]["final_answer"]

            ace_chain = ACELangChain(
                my_chain,
                output_parser=parse_output
            )

            # With deduplication
            from ace import DeduplicationConfig
            ace_chain = ACELangChain(
                my_chain,
                dedup_config=DeduplicationConfig(similarity_threshold=0.85)
            )
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install with:\n"
                "pip install ace-framework[langchain]\n"
                "or: pip install langchain-core"
            )

        self.runnable = runnable
        self.is_learning = is_learning
        self._async_learning = async_learning
        self._learning_tasks: List[asyncio.Task] = []
        self._tasks_submitted_count: int = 0
        self._tasks_completed_count: int = 0
        self.output_parser = output_parser or self._default_output_parser

        # Load or create playbook
        if playbook_path:
            self.playbook = Playbook.load_from_file(playbook_path)
        else:
            self.playbook = Playbook()

        # Setup ACE learning components
        try:
            from ..llm_providers import LiteLLMClient
        except ImportError:
            raise ImportError(
                "ACELangChain requires LiteLLM. Install with:\n"
                "pip install ace-framework  # (includes LiteLLM by default)"
            )

        self.llm = LiteLLMClient(model=ace_model, max_tokens=2048)

        # Create ACE components with v2.1 prompts
        prompt_mgr = PromptManager()
        self.reflector = Reflector(
            self.llm, prompt_template=prompt_mgr.get_reflector_prompt()
        )

        # Create DeduplicationManager if config provided
        dedup_manager = None
        if dedup_config is not None:
            from ..deduplication import DeduplicationManager

            dedup_manager = DeduplicationManager(dedup_config)

        self.curator = Curator(
            self.llm,
            prompt_template=prompt_mgr.get_curator_prompt(),
            dedup_manager=dedup_manager,
        )

    def invoke(self, input: Any, **kwargs) -> Any:
        """
        Execute runnable with ACE learning (sync).

        For AgentExecutor runnables, automatically enables intermediate step
        extraction to provide richer reasoning traces to the Reflector.

        Args:
            input: Input for the runnable (string, dict, etc.)
            **kwargs: Additional arguments passed to runnable.invoke()

        Returns:
            Output from the runnable

        Example:
            # String input
            result = ace_chain.invoke("What is ACE?")

            # Dict input
            result = ace_chain.invoke({"question": "What is ACE?"})
        """
        # Step 1: Inject playbook context
        enhanced_input = self._inject_context(input)

        # Step 2: Configure AgentExecutor for intermediate steps (if applicable)
        is_agent = self._is_agent_executor()
        original_setting = False
        if is_agent:
            original_setting = getattr(
                self.runnable, "return_intermediate_steps", False
            )
            self.runnable.return_intermediate_steps = True

        # Step 3: Execute runnable
        try:
            result = self.runnable.invoke(enhanced_input, **kwargs)
        except Exception as e:
            logger.error(f"Error executing runnable: {e}")
            # Restore original setting
            if is_agent:
                self.runnable.return_intermediate_steps = original_setting
            # Learn from failure
            if self.is_learning:
                self._learn_from_failure(input, str(e))
            raise
        finally:
            # Restore original setting
            if is_agent:
                self.runnable.return_intermediate_steps = original_setting

        # Step 4: Learn from result
        if self.is_learning:
            if is_agent and isinstance(result, dict) and "intermediate_steps" in result:
                # Rich trace from AgentExecutor
                self._learn_with_trace(input, result)
            else:
                # Basic learning
                self._learn(input, result)

        # Step 5: Return clean output
        if is_agent and isinstance(result, dict) and "output" in result:
            return result["output"]
        return result

    async def ainvoke(self, input: Any, **kwargs) -> Any:
        """
        Execute runnable with ACE learning (async).

        For AgentExecutor runnables, automatically enables intermediate step
        extraction to provide richer reasoning traces to the Reflector.

        Args:
            input: Input for the runnable (string, dict, etc.)
            **kwargs: Additional arguments passed to runnable.ainvoke()

        Returns:
            Output from the runnable

        Example:
            result = await ace_chain.ainvoke({"input": "Question"})

            # With async_learning=True, learning happens in background
            ace_chain = ACELangChain(chain, async_learning=True)
            result = await ace_chain.ainvoke({"input": "Question"})
            # Result returns immediately
            await ace_chain.wait_for_learning()  # Wait for learning to complete
        """
        # Step 1: Inject playbook context
        enhanced_input = self._inject_context(input)

        # Step 2: Configure AgentExecutor for intermediate steps (if applicable)
        is_agent = self._is_agent_executor()
        original_setting = False
        if is_agent:
            original_setting = getattr(
                self.runnable, "return_intermediate_steps", False
            )
            self.runnable.return_intermediate_steps = True

        # Step 3: Execute runnable (async)
        try:
            result = await self.runnable.ainvoke(enhanced_input, **kwargs)
        except Exception as e:
            logger.error(f"Error executing runnable: {e}")
            # Restore original setting
            if is_agent:
                self.runnable.return_intermediate_steps = original_setting
            # Learn from failure
            if self.is_learning:
                if self._async_learning:
                    task = asyncio.create_task(self._alearn_from_failure(input, str(e)))
                    task.add_done_callback(self._on_task_done)
                    self._learning_tasks.append(task)
                    self._tasks_submitted_count += 1
                else:
                    await self._alearn_from_failure(input, str(e))
            raise
        finally:
            # Restore original setting
            if is_agent:
                self.runnable.return_intermediate_steps = original_setting

        # Step 4: Learn from result
        if self.is_learning:
            # Determine which learning method to use
            use_trace = (
                is_agent and isinstance(result, dict) and "intermediate_steps" in result
            )

            if self._async_learning:
                if use_trace:
                    task = asyncio.create_task(self._alearn_with_trace(input, result))
                else:
                    task = asyncio.create_task(self._alearn(input, result))
                task.add_done_callback(self._on_task_done)
                self._learning_tasks.append(task)
                self._tasks_submitted_count += 1
            else:
                if use_trace:
                    await self._alearn_with_trace(input, result)
                else:
                    await self._alearn(input, result)

        # Step 5: Return clean output
        if is_agent and isinstance(result, dict) and "output" in result:
            return result["output"]
        return result

    def _inject_context(self, input: Any) -> Any:
        """
        Add playbook context to input.

        Handles common input formats:
        - String: Append playbook context
        - Dict with "input" key: Enhance input field
        - Dict without: Add playbook_context key
        - Other: Return unchanged (no playbook strategies yet)

        Args:
            input: Original input

        Returns:
            Enhanced input with playbook context
        """
        # No context if no strategies yet
        if not self.playbook or not self.playbook.bullets():
            return input

        playbook_context = wrap_playbook_context(self.playbook)

        # String input - append context
        if isinstance(input, str):
            return f"{input}\n\n{playbook_context}"

        # Dict input with "input" key - enhance that field
        if isinstance(input, dict) and "input" in input:
            enhanced = input.copy()
            enhanced["input"] = f"{input['input']}\n\n{playbook_context}"
            return enhanced

        # Dict input without "input" key - add playbook_context key
        if isinstance(input, dict):
            enhanced = input.copy()
            enhanced["playbook_context"] = playbook_context
            return enhanced

        # Other types - return unchanged
        return input

    def _learn(self, original_input: Any, result: Any):
        """
        Learn from successful execution.

        Args:
            original_input: Original input to runnable
            result: Output from runnable
        """
        try:
            # Parse output to string
            output_str = self.output_parser(result)

            # Build task description
            if isinstance(original_input, str):
                task = original_input
            elif isinstance(original_input, dict):
                # Try common keys
                task = (
                    original_input.get("input")
                    or original_input.get("question")
                    or original_input.get("query")
                    or str(original_input)
                )
            else:
                task = str(original_input)

            # Create adapter for Reflector interface
            # Note: LangChain chains don't expose intermediate reasoning, so we
            # focus the learning on task patterns and output quality rather than
            # reasoning steps. This prevents the Reflector from generating
            # meta-strategies about the learning system itself.
            generator_output = GeneratorOutput(
                reasoning=f"""Question/Task: {task}

Chain Output: {output_str}

Note: This is an external LangChain chain execution. Learning should focus on task patterns and output quality, not internal system behavior.""",
                final_answer=output_str,
                bullet_ids=[],  # LangChain runnables don't cite bullets
                raw={"input": original_input, "output": result},
            )

            # Build feedback - focus on task outcome, not system behavior
            feedback = f"External chain completed for task: {task[:200]}"

            # Reflect: Analyze execution
            reflection = self.reflector.reflect(
                question=task,
                generator_output=generator_output,
                playbook=self.playbook,
                ground_truth=None,
                feedback=feedback,
            )

            # Curate: Generate playbook updates
            curator_output = self.curator.curate(
                reflection=reflection,
                playbook=self.playbook,
                question_context=f"task: {task}",
                progress=task,
            )

            # Apply updates
            self.playbook.apply_delta(curator_output.delta)

        except Exception as e:
            logger.error(f"ACE learning failed: {e}")
            # Don't crash - continue without learning

    def _learn_from_failure(self, original_input: Any, error_msg: str):
        """
        Learn from execution failure.

        Args:
            original_input: Original input to runnable
            error_msg: Error message
        """
        try:
            # Build task description
            if isinstance(original_input, str):
                task = original_input
            elif isinstance(original_input, dict):
                task = (
                    original_input.get("input")
                    or original_input.get("question")
                    or str(original_input)
                )
            else:
                task = str(original_input)

            # Create adapter for Reflector interface
            # Note: Focus learning on task failure patterns, not system behavior
            generator_output = GeneratorOutput(
                reasoning=f"""Question/Task: {task}

Execution Result: FAILED
Error: {error_msg}

Note: This is an external LangChain chain execution that failed. Learning should focus on task patterns that may have caused the failure, not internal system behavior.""",
                final_answer=f"Failed: {error_msg}",
                bullet_ids=[],
                raw={"input": original_input, "error": error_msg},
            )

            # Build failure feedback - focus on task, not system
            feedback = f"Chain execution failed for task: {task[:200]}. Error: {error_msg[:200]}"

            # Reflect on failure
            reflection = self.reflector.reflect(
                question=task,
                generator_output=generator_output,
                playbook=self.playbook,
                ground_truth=None,
                feedback=feedback,
            )

            # Curate: Learn from failure patterns
            curator_output = self.curator.curate(
                reflection=reflection,
                playbook=self.playbook,
                question_context=f"task: {task}",
                progress=f"Failed: {task}",
            )

            # Apply updates
            self.playbook.apply_delta(curator_output.delta)

        except Exception as e:
            logger.error(f"ACE failure learning failed: {e}")
            # Don't crash

    def _learn_with_trace(self, original_input: Any, result: Dict[str, Any]):
        """
        Learn from AgentExecutor result with intermediate_steps.

        Extracts rich reasoning trace from AgentExecutor's intermediate_steps
        including agent thoughts, tool calls, and observations.

        Args:
            original_input: Original input to runnable
            result: AgentExecutor result dict with 'output' and 'intermediate_steps'
        """
        try:
            task = self._get_task_str(original_input)
            output = result.get("output", "")
            steps = result.get("intermediate_steps", [])

            # Build rich reasoning from steps
            parts = [f"Question/Task: {task}", ""]
            parts.append(f"=== AGENT EXECUTION TRACE ({len(steps)} steps) ===")

            for i, step_tuple in enumerate(steps, 1):
                if len(step_tuple) != 2:
                    continue
                action, observation = step_tuple

                parts.append(f"\n--- Step {i} ---")
                # Extract thought from action.log (agent's reasoning)
                if hasattr(action, "log") and action.log:
                    parts.append(f"Thought: {action.log}")
                # Extract tool and input
                if hasattr(action, "tool"):
                    parts.append(f"Action: {action.tool}")
                    tool_input = str(action.tool_input)[:300]
                    parts.append(f"Action Input: {tool_input}")
                # Extract observation (tool output)
                observation_str = str(observation)[:300]
                parts.append(f"Observation: {observation_str}")

            parts.append("\n=== END TRACE ===")
            parts.append(f"\nFinal Answer: {output}")

            reasoning = "\n".join(parts)

            generator_output = GeneratorOutput(
                reasoning=reasoning,
                final_answer=output,
                bullet_ids=[],
                raw={"input": original_input, "output": result},
            )

            feedback = f"Agent completed task in {len(steps)} steps"

            # Reflect and Curate
            reflection = self.reflector.reflect(
                question=task,
                generator_output=generator_output,
                playbook=self.playbook,
                ground_truth=None,
                feedback=feedback,
            )

            curator_output = self.curator.curate(
                reflection=reflection,
                playbook=self.playbook,
                question_context=f"task: {task}",
                progress=task,
            )

            self.playbook.apply_delta(curator_output.delta)

        except Exception as e:
            logger.error(f"ACE learning with trace failed: {e}")
            # Don't crash - continue without learning

    async def _alearn(self, original_input: Any, result: Any):
        """
        Async version of _learn for background execution.

        Uses asyncio.to_thread() to run sync learning in a thread pool,
        preventing event loop blocking when async_learning=True.
        """
        await asyncio.to_thread(self._learn, original_input, result)

    async def _alearn_from_failure(self, original_input: Any, error_msg: str):
        """
        Async version of _learn_from_failure for background execution.

        Uses asyncio.to_thread() to run sync learning in a thread pool,
        preventing event loop blocking when async_learning=True.
        """
        await asyncio.to_thread(self._learn_from_failure, original_input, error_msg)

    async def _alearn_with_trace(self, original_input: Any, result: Dict[str, Any]):
        """
        Async version of _learn_with_trace for background execution.

        Uses asyncio.to_thread() to run sync learning in a thread pool,
        preventing event loop blocking when async_learning=True.
        """
        await asyncio.to_thread(self._learn_with_trace, original_input, result)

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Callback when a learning task completes (success or failure)."""
        self._tasks_completed_count += 1

    async def wait_for_learning(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all background learning tasks to complete.

        Only relevant when using async_learning=True.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            True if all learning completed, False if timeout reached

        Example:
            ace_chain = ACELangChain(chain, async_learning=True)
            result = await ace_chain.ainvoke(input)
            # Do other work while learning happens...
            success = await ace_chain.wait_for_learning(timeout=60.0)
            if success:
                print("Learning complete!")
        """
        if not self._learning_tasks:
            return True

        # Clean up completed tasks
        self._learning_tasks = [t for t in self._learning_tasks if not t.done()]
        if not self._learning_tasks:
            return True

        try:
            await asyncio.wait_for(
                asyncio.gather(*self._learning_tasks, return_exceptions=True),
                timeout=timeout,
            )
            self._learning_tasks.clear()
            return True
        except asyncio.TimeoutError:
            return False

    @property
    def learning_stats(self) -> Dict[str, Any]:
        """
        Get async learning statistics.

        Returns:
            Dictionary with learning progress info:
            - tasks_submitted: Total tasks ever created
            - pending: Number of tasks still running
            - completed: Number of tasks finished
            - async_learning: Whether async mode is enabled

        Example:
            stats = ace_chain.learning_stats
            print(f"Pending: {stats['pending']}")
        """
        pending = len([t for t in self._learning_tasks if not t.done()])
        return {
            "tasks_submitted": self._tasks_submitted_count,
            "pending": pending,
            "completed": self._tasks_completed_count,
            "async_learning": self._async_learning,
        }

    def stop_async_learning(self):
        """
        Cancel all pending learning tasks.

        Call this to stop background learning early.

        Example:
            ace_chain = ACELangChain(chain, async_learning=True)
            await ace_chain.ainvoke(input)
            # Decide to stop early...
            ace_chain.stop_async_learning()
        """
        for task in self._learning_tasks:
            if not task.done():
                task.cancel()
        self._learning_tasks.clear()

    @staticmethod
    def _default_output_parser(result: Any) -> str:
        """
        Default output parser - converts result to string.

        Args:
            result: Runnable output

        Returns:
            String representation
        """
        # String - return as is
        if isinstance(result, str):
            return result

        # LangChain messages (AIMessage, etc.) have .content attribute
        if hasattr(result, "content"):
            return str(result.content)

        # Dict - try common output keys
        if isinstance(result, dict):
            for key in ["output", "answer", "result", "text"]:
                if key in result:
                    return str(result[key])
            return str(result)

        # Fallback to string representation
        return str(result)

    def _is_agent_executor(self) -> bool:
        """
        Check if the wrapped runnable is an AgentExecutor.

        AgentExecutor supports rich tracing via return_intermediate_steps=True.
        """
        if not AGENT_EXECUTOR_AVAILABLE or AgentExecutor is None:
            return False
        return isinstance(self.runnable, AgentExecutor)

    def _get_task_str(self, original_input: Any) -> str:
        """Extract task string from input."""
        if isinstance(original_input, str):
            return original_input
        elif isinstance(original_input, dict):
            return (
                original_input.get("input")
                or original_input.get("question")
                or original_input.get("query")
                or str(original_input)
            )
        return str(original_input)

    def save_playbook(self, path: str):
        """
        Save learned playbook to file.

        Args:
            path: File path to save to

        Example:
            ace_chain.save_playbook("chain_expert.json")
        """
        self.playbook.save_to_file(path)

    def load_playbook(self, path: str):
        """
        Load playbook from file (replaces current).

        Args:
            path: File path to load from

        Example:
            ace_chain.load_playbook("expert.json")
        """
        self.playbook = Playbook.load_from_file(path)

    def enable_learning(self):
        """Enable learning (allows learn() to update playbook)."""
        self.is_learning = True

    def disable_learning(self):
        """Disable learning (prevents learn() from updating playbook)."""
        self.is_learning = False

    def __repr__(self) -> str:
        """String representation."""
        bullets_count = len(self.playbook.bullets()) if self.playbook else 0
        return (
            f"ACELangChain("
            f"runnable={self.runnable.__class__.__name__}, "
            f"strategies={bullets_count}, "
            f"learning={'enabled' if self.is_learning else 'disabled'})"
        )


__all__ = ["ACELangChain", "LANGCHAIN_AVAILABLE"]
