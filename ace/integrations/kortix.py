"""
Kortix (OpenCode) integration for ACE framework.

This module provides ACEKortix, a wrapper for Kortix agents via the OpenCode
HTTP API that automatically learns from execution feedback.

Example:
    from ace.integrations import ACEKortix

    agent = ACEKortix(base_url="http://localhost:8000")
    result = agent.run(task="Create a Python fibonacci script")
    agent.save_skillbook("learned.json")

    # Post-hoc learning from existing session
    agent.learn_from_session("session-abc-123")

    # With async learning
    agent = ACEKortix(base_url="http://localhost:8000", async_learning=True)
    result = agent.run(task="Task 1")
    agent.wait_for_learning()
"""

import json
import queue
import threading
import time
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import requests

from ..llm_providers import LiteLLMClient
from ..skillbook import Skillbook
from ..roles import Reflector, SkillManager, AgentOutput
from ..prompt_manager import PromptManager
from .base import wrap_skillbook_context

if TYPE_CHECKING:
    from ..deduplication import DeduplicationConfig, DeduplicationManager

logger = logging.getLogger(__name__)

# Kortix skill file template (YAML frontmatter + markdown body)
SKILL_TEMPLATE = """---
name: kortix-ace-strategies
description: "Learned execution strategies from ACE. Patterns extracted from past task analysis — what worked, what failed, and reusable approaches. Load when starting complex tasks."
---

# ACE Learned Strategies

{strategies}
"""


@dataclass
class KortixResult:
    """Result from Kortix agent execution."""

    success: bool
    output: str
    execution_trace: str
    session_id: str
    cost: float = 0.0
    tokens: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    feedback_rating: Optional[int] = None
    feedback_text: Optional[str] = None


class KortixClient:
    """
    HTTP client for the OpenCode REST API.

    Handles session management, message sending, and file operations
    for communicating with a Kortix agent running in a sandbox.

    Args:
        base_url: OpenCode API base URL (e.g. "http://localhost:8000")
        timeout: Default request timeout in seconds
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def create_session(self, agent: str = "kortix-main") -> str:
        """Create a new agent session.

        Args:
            agent: Agent identifier to use for the session.

        Returns:
            Session ID string.
        """
        resp = self.session.post(
            f"{self.base_url}/session",
            json={"agent": agent},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["id"]

    def send_message(self, session_id: str, text: str) -> None:
        """Send a message to the session (fire-and-forget).

        Args:
            session_id: Target session ID.
            text: Message text to send.
        """
        resp = self.session.post(
            f"{self.base_url}/session/{session_id}/prompt-async",
            json={"text": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()

    def wait_for_idle(
        self, session_id: str, timeout: int = 600, poll_interval: float = 2.0
    ) -> bool:
        """Poll until the session is idle (agent finished processing).

        Args:
            session_id: Session to monitor.
            timeout: Maximum wait time in seconds.
            poll_interval: Seconds between polls.

        Returns:
            True if session became idle, False if timeout reached.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = self.session.get(
                    f"{self.base_url}/session/{session_id}",
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                # Session is idle when not actively processing
                if data.get("idle", False) or data.get("status") == "idle":
                    return True
            except requests.RequestException:
                pass
            time.sleep(poll_interval)
        return False

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all messages from a session.

        Args:
            session_id: Session to fetch messages from.

        Returns:
            List of message dicts with info/parts structure.
        """
        resp = self.session.get(
            f"{self.base_url}/session/{session_id}/messages",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def write_file(self, path: str, content: str) -> None:
        """Write a file into the sandbox.

        Args:
            path: Absolute path inside the sandbox.
            content: File content to write.
        """
        resp = self.session.post(
            f"{self.base_url}/file/write",
            json={"path": path, "content": content},
            timeout=self.timeout,
        )
        resp.raise_for_status()

    def read_file(self, path: str) -> str:
        """Read a file from the sandbox.

        Args:
            path: Absolute path inside the sandbox.

        Returns:
            File content string.
        """
        resp = self.session.get(
            f"{self.base_url}/file/read",
            params={"path": path},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.text


class ACEKortix:
    """
    Kortix agent with ACE learning capabilities.

    Executes tasks via Kortix (OpenCode HTTP API) and learns from execution.
    Drop-in wrapper that automatically:
    - Injects learned strategies into the sandbox as a skill file
    - Reflects on execution results
    - Updates skillbook with new learnings

    Usage:
        # Simple usage
        agent = ACEKortix(base_url="http://localhost:8000")
        result = agent.run(task="Create a Python fibonacci script")

        # Reuse across tasks (learns from each)
        agent.run(task="Task 1")
        agent.run(task="Task 2")  # Uses Task 1 learnings
        agent.save_skillbook("expert.json")

        # Post-hoc learning from existing session
        agent.learn_from_session("session-abc-123")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        ace_model: str = "gpt-4o-mini",
        ace_llm: Optional[LiteLLMClient] = None,
        ace_max_tokens: int = 2048,
        skillbook: Optional[Skillbook] = None,
        skillbook_path: Optional[str] = None,
        is_learning: bool = True,
        timeout: int = 600,
        async_learning: bool = False,
        max_reflector_workers: int = 3,
        dedup_config: Optional["DeduplicationConfig"] = None,
        skill_path: str = "/opt/opencode/skills/KORTIX-ace/SKILL.md",
        session_agent: str = "kortix-main",
        client: Optional[KortixClient] = None,
    ):
        """
        Initialize ACEKortix.

        Args:
            base_url: OpenCode API base URL.
            ace_model: Model for ACE learning (Reflector/SkillManager).
            ace_llm: Custom LLM client for ACE (overrides ace_model).
            ace_max_tokens: Max tokens for ACE learning LLM.
            skillbook: Existing Skillbook instance.
            skillbook_path: Path to load skillbook from.
            is_learning: Enable/disable ACE learning.
            timeout: Execution timeout in seconds.
            async_learning: Run learning in background.
            max_reflector_workers: Parallel Reflector threads (unused, reserved).
            dedup_config: Optional DeduplicationConfig for skill deduplication.
            skill_path: Path inside sandbox for the ACE skill file.
            session_agent: Agent identifier for new sessions.
            client: Optional pre-configured KortixClient (for testing).
        """
        self.base_url = base_url
        self.is_learning = is_learning
        self.timeout = timeout
        self.async_learning = async_learning
        self.max_reflector_workers = max_reflector_workers
        self.dedup_config = dedup_config
        self.skill_path = skill_path
        self.session_agent = session_agent

        # HTTP client
        self.client = client or KortixClient(base_url=base_url)

        # Load or create skillbook
        if skillbook_path:
            self.skillbook = Skillbook.load_from_file(skillbook_path)
        elif skillbook:
            self.skillbook = skillbook
        else:
            self.skillbook = Skillbook()

        # Create ACE LLM (for Reflector/SkillManager)
        self.ace_llm = ace_llm or LiteLLMClient(
            model=ace_model, max_tokens=ace_max_tokens
        )

        # Create ACE learning components with v2.1 prompts
        prompt_mgr = PromptManager()
        self.reflector = Reflector(
            self.ace_llm, prompt_template=prompt_mgr.get_reflector_prompt()
        )
        self.skill_manager = SkillManager(
            self.ace_llm, prompt_template=prompt_mgr.get_skill_manager_prompt()
        )

        # Initialize deduplication manager if config provided
        self._dedup_manager: Optional["DeduplicationManager"] = None
        if dedup_config:
            from ..deduplication import DeduplicationManager

            self._dedup_manager = DeduplicationManager(dedup_config)

        # Async learning state
        self._learning_queue: queue.Queue = queue.Queue()
        self._learning_thread: Optional[threading.Thread] = None
        self._stop_learning = threading.Event()
        self._tasks_submitted = 0
        self._tasks_completed = 0
        self._lock = threading.Lock()

        # Start async learning thread if enabled
        if async_learning:
            self._start_async_learning()

    def run(
        self,
        task: str,
        context: str = "",
        session_id: Optional[str] = None,
        feedback_rating: Optional[int] = None,
        feedback_text: Optional[str] = None,
    ) -> KortixResult:
        """
        Execute task via Kortix with ACE learning.

        Args:
            task: Task description for the Kortix agent.
            context: Additional context (optional).
            session_id: Existing session ID to reuse (creates new if None).
            feedback_rating: Optional 1-5 star rating for learning.
            feedback_text: Optional text feedback for learning.

        Returns:
            KortixResult with execution details.
        """
        # 1. INJECT: Sync skillbook to sandbox as skill file
        if self.is_learning and self.skillbook.skills():
            try:
                self._sync_skill_to_sandbox()
            except Exception as e:
                logger.warning(f"Failed to sync skill to sandbox: {e}")

        # Build prompt
        prompt = f"{task}\n\n{context}" if context else task

        # 2. EXECUTE: Run via Kortix
        result = self._execute_kortix(prompt, session_id)
        result.feedback_rating = feedback_rating
        result.feedback_text = feedback_text

        # 3. LEARN: Run ACE learning if enabled
        if self.is_learning:
            if self.async_learning:
                with self._lock:
                    self._tasks_submitted += 1
                self._learning_queue.put((task, result))
            else:
                self._learn_from_execution(task, result)

        return result

    def _execute_kortix(
        self, prompt: str, session_id: Optional[str] = None
    ) -> KortixResult:
        """Execute a task via the Kortix HTTP API."""
        try:
            # Create or reuse session
            sid = session_id or self.client.create_session(agent=self.session_agent)

            # Send message
            self.client.send_message(sid, prompt)

            # Wait for completion
            idle = self.client.wait_for_idle(sid, timeout=self.timeout)
            if not idle:
                return KortixResult(
                    success=False,
                    output="",
                    execution_trace="",
                    session_id=sid,
                    error=f"Session timed out after {self.timeout}s",
                )

            # Get messages and parse
            messages = self.client.get_messages(sid)
            output, trace, cost, tokens = self._parse_messages(messages)

            return KortixResult(
                success=True,
                output=output,
                execution_trace=trace,
                session_id=sid,
                cost=cost,
                tokens=tokens,
            )

        except requests.RequestException as e:
            return KortixResult(
                success=False,
                output="",
                execution_trace="",
                session_id=session_id or "",
                error=f"HTTP error: {e}",
            )
        except Exception as e:
            return KortixResult(
                success=False,
                output="",
                execution_trace="",
                session_id=session_id or "",
                error=str(e),
            )

    def _parse_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[str, str, float, Dict[str, Any]]:
        """
        Parse OpenCode messages into execution trace and output.

        Converts the OpenCode message/part JSON format into a chronological
        trace string suitable for the ACE Reflector.

        Args:
            messages: List of message dicts from GET /session/{id}/messages.

        Returns:
            Tuple of (final_output, execution_trace, total_cost, token_counts).
        """
        trace_parts: List[str] = []
        final_text = ""
        step_num = 0
        total_cost = 0.0
        total_tokens: Dict[str, Any] = {}

        for msg in messages:
            info = msg.get("info", {})
            role = info.get("role", "")

            # Accumulate cost from assistant messages
            if role == "assistant":
                msg_cost = info.get("cost", 0)
                if msg_cost:
                    total_cost += float(msg_cost)
                msg_tokens = info.get("tokens", {})
                if msg_tokens:
                    for k, v in msg_tokens.items():
                        total_tokens[k] = total_tokens.get(k, 0) + v

            # Only process assistant parts for trace
            if role != "assistant":
                continue

            parts = msg.get("parts", [])
            for part in parts:
                part_type = part.get("type", "")

                if part_type == "text":
                    text = part.get("text", "").strip()
                    if text:
                        trace_parts.append(f"[Reasoning] {text[:300]}")
                        final_text = text

                elif part_type == "tool":
                    step_num += 1
                    tool_name = part.get("tool", "unknown")
                    state = part.get("state", {})
                    tool_input = state.get("input", {})
                    tool_output = state.get("output", "")
                    tool_status = state.get("status", "")
                    tool_error = state.get("error", "")

                    # Format tool input summary
                    if isinstance(tool_input, dict):
                        # Extract most meaningful field
                        input_summary = (
                            tool_input.get("command")
                            or tool_input.get("file_path")
                            or tool_input.get("pattern")
                            or tool_input.get("url")
                            or tool_input.get("query")
                            or str(tool_input)[:80]
                        )
                    else:
                        input_summary = str(tool_input)[:80]

                    trace_parts.append(f"[Step {step_num}] {tool_name}: {input_summary}")

                    # Add output or error
                    if tool_error:
                        trace_parts.append(f"  -> ERROR: {str(tool_error)[:200]}")
                    elif tool_output:
                        trace_parts.append(f"  -> {str(tool_output)[:200]}")

                # Skip reasoning/thinking parts, step-start, step-finish

        execution_trace = (
            "\n".join(trace_parts) if trace_parts else "(No trace captured)"
        )

        # Extract final output
        if final_text:
            paragraphs = [p.strip() for p in final_text.split("\n\n") if p.strip()]
            output = paragraphs[-1][:500] if paragraphs else final_text[:500]
        else:
            output = f"Completed {step_num} steps"

        return output, execution_trace, total_cost, total_tokens

    def _learn_from_execution(self, task: str, result: KortixResult) -> None:
        """Run ACE learning pipeline after execution."""
        agent_output = AgentOutput(
            reasoning=result.execution_trace,
            final_answer=result.output,
            skill_ids=[],
            raw={
                "success": result.success,
                "session_id": result.session_id,
                "cost": result.cost,
            },
        )

        # Build feedback
        status = "succeeded" if result.success else "failed"
        feedback = f"Kortix task {status}"
        if result.error:
            feedback += f"\nError: {result.error}"
        if result.feedback_rating is not None:
            feedback += f"\nUser rating: {result.feedback_rating}/5"
        if result.feedback_text:
            feedback += f"\nUser feedback: {result.feedback_text}"

        # Run Reflector
        reflection = self.reflector.reflect(
            question=task,
            agent_output=agent_output,
            skillbook=self.skillbook,
            ground_truth=None,
            feedback=feedback,
        )

        # Run SkillManager
        skill_manager_output = self.skill_manager.update_skills(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=f"task: {task}",
            progress=f"Kortix: {task}",
        )

        # Update skillbook
        self.skillbook.apply_update(skill_manager_output.update)

        # Apply consolidation if deduplication enabled
        if self._dedup_manager and skill_manager_output.raw:
            self._dedup_manager.apply_operations_from_response(
                skill_manager_output.raw, self.skillbook
            )

        # Re-sync updated strategies to sandbox
        try:
            self._sync_skill_to_sandbox()
        except Exception as e:
            logger.warning(f"Failed to sync updated skill to sandbox: {e}")

    def _sync_skill_to_sandbox(self) -> None:
        """Write the current skillbook to the sandbox as a Kortix skill file."""
        if self.skillbook.skills():
            strategies = wrap_skillbook_context(self.skillbook)
        else:
            strategies = (
                "No strategies learned yet. This skill is automatically updated "
                "by ACE as it analyzes your completed sessions."
            )
        skill_md = SKILL_TEMPLATE.format(strategies=strategies)
        self.client.write_file(self.skill_path, skill_md)

    def learn_from_session(
        self,
        session_id: str,
        feedback_rating: Optional[int] = None,
        feedback_text: Optional[str] = None,
    ) -> bool:
        """
        Learn from an existing Kortix session (post-hoc).

        Fetches the session transcript, extracts the task and trace,
        and runs the ACE learning pipeline.

        Args:
            session_id: ID of a completed session.
            feedback_rating: Optional 1-5 star rating.
            feedback_text: Optional text feedback.

        Returns:
            True if learning succeeded, False otherwise.
        """
        try:
            messages = self.client.get_messages(session_id)
            if not messages:
                logger.warning(f"No messages found for session {session_id}")
                return False

            # Extract task from first user message
            task = ""
            for msg in messages:
                if msg.get("info", {}).get("role") == "user":
                    parts = msg.get("parts", [])
                    for part in parts:
                        if part.get("type") == "text":
                            task = part.get("text", "")
                            break
                    if task:
                        break

            if not task:
                logger.warning(f"No user message found in session {session_id}")
                return False

            # Parse trace
            output, trace, cost, tokens = self._parse_messages(messages)

            result = KortixResult(
                success=True,
                output=output,
                execution_trace=trace,
                session_id=session_id,
                cost=cost,
                tokens=tokens,
                feedback_rating=feedback_rating,
                feedback_text=feedback_text,
            )

            self._learn_from_execution(task, result)
            return True

        except Exception as e:
            logger.error(f"Failed to learn from session {session_id}: {e}")
            return False

    def save_skillbook(self, path: str) -> None:
        """Save learned skillbook to file."""
        self.skillbook.save_to_file(path)

    def load_skillbook(self, path: str) -> None:
        """Load skillbook from file."""
        self.skillbook = Skillbook.load_from_file(path)

    def get_strategies(self) -> str:
        """Get current skillbook strategies as formatted text."""
        if not self.skillbook.skills():
            return ""
        return wrap_skillbook_context(self.skillbook)

    def enable_learning(self) -> None:
        """Enable ACE learning."""
        self.is_learning = True

    def disable_learning(self) -> None:
        """Disable ACE learning (execution only)."""
        self.is_learning = False

    # --- Async learning ---

    def _start_async_learning(self) -> None:
        """Start the background learning thread."""
        if self._learning_thread is not None and self._learning_thread.is_alive():
            return
        self._stop_learning.clear()
        self._learning_thread = threading.Thread(
            target=self._learning_worker, daemon=True
        )
        self._learning_thread.start()

    def _learning_worker(self) -> None:
        """Background worker that processes learning tasks."""
        while not self._stop_learning.is_set():
            try:
                task, result = self._learning_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._learn_from_execution(task, result)
            finally:
                with self._lock:
                    self._tasks_completed += 1
                self._learning_queue.task_done()

    def wait_for_learning(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for async learning to complete.

        Args:
            timeout: Maximum seconds to wait (None = wait forever).

        Returns:
            True if all learning completed, False if timeout reached.
        """
        if not self.async_learning:
            return True
        try:
            if timeout is not None:
                start = time.time()
                while not self._learning_queue.empty():
                    if time.time() - start >= timeout:
                        return False
                    time.sleep(0.1)
                return True
            else:
                self._learning_queue.join()
                return True
        except Exception:
            return False

    def stop_async_learning(self, wait: bool = True) -> None:
        """
        Stop async learning pipeline.

        Args:
            wait: If True, wait for current tasks to complete.
        """
        if not self.async_learning:
            return
        if wait:
            self.wait_for_learning()
        self._stop_learning.set()
        if self._learning_thread and self._learning_thread.is_alive():
            self._learning_thread.join(timeout=5.0)

    @property
    def learning_stats(self) -> Dict[str, Any]:
        """Get async learning statistics."""
        with self._lock:
            submitted = self._tasks_submitted
            completed = self._tasks_completed
        return {
            "async_learning": self.async_learning,
            "tasks_submitted": submitted,
            "tasks_completed": completed,
            "pending": submitted - completed,
            "queue_size": self._learning_queue.qsize(),
        }

    def __del__(self):
        """Cleanup async learning resources on deletion."""
        try:
            self.stop_async_learning(wait=False)
        except Exception:
            pass


__all__ = ["ACEKortix", "KortixResult", "KortixClient"]
