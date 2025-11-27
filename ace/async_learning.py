"""Async learning infrastructure for ACE.

This module provides parallel Reflector execution with serialized Curator
for efficient learning without playbook conflicts.

Architecture:
    Sample 1 ──► Generator ──► Env ──► Reflector ─┐
    Sample 2 ──► Generator ──► Env ──► Reflector ─┼──► [Queue] ──► Curator ──► Playbook
    Sample 3 ──► Generator ──► Env ──► Reflector ─┘              (serialized)
               (parallel)           (parallel)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from .adaptation import EnvironmentResult, Sample
    from .playbook import Playbook
    from .roles import Curator, GeneratorOutput, Reflector, ReflectorOutput

from .delta import DeltaBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class LearningTask:
    """Input to Reflector (from main thread).

    Contains all data needed to run reflection on a sample's results.
    """

    sample: "Sample"
    generator_output: "GeneratorOutput"
    environment_result: "EnvironmentResult"
    epoch: int
    step_index: int
    total_epochs: int = 1
    total_steps: int = 1
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionResult:
    """Output from Reflector (goes to Curator queue).

    Contains the original task plus the reflection analysis.
    """

    task: LearningTask
    reflection: "ReflectorOutput"
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Thread-Safe Playbook Wrapper
# ---------------------------------------------------------------------------


class ThreadSafePlaybook:
    """Thread-safe wrapper for Playbook with RWLock semantics.

    Provides:
    - Lock-free reads for eventual consistency (Generator can read anytime)
    - Locked writes to ensure atomic playbook updates (Curator serialized)

    Example:
        >>> ts_playbook = ThreadSafePlaybook(playbook)
        >>> # Reads are lock-free
        >>> prompt = ts_playbook.as_prompt()
        >>> # Writes are locked
        >>> ts_playbook.apply_delta(delta_batch)
    """

    def __init__(self, playbook: "Playbook") -> None:
        self._playbook = playbook
        self._lock = threading.RLock()

    @property
    def playbook(self) -> "Playbook":
        """Direct access to underlying playbook (for read operations)."""
        return self._playbook

    # -----------------------------------------------------------------------
    # Lock-free reads (eventual consistency)
    # -----------------------------------------------------------------------

    def as_prompt(self) -> str:
        """Get TOON-encoded playbook for LLM prompts (lock-free)."""
        return self._playbook.as_prompt()

    def bullets(self) -> List[Any]:
        """Get all bullets (lock-free)."""
        return self._playbook.bullets()

    def get_bullet(self, bullet_id: str) -> Optional[Any]:
        """Get a bullet by ID (lock-free)."""
        return self._playbook.get_bullet(bullet_id)

    def stats(self) -> Dict[str, object]:
        """Get playbook statistics (lock-free)."""
        return self._playbook.stats()

    # -----------------------------------------------------------------------
    # Locked writes (serialized for thread safety)
    # -----------------------------------------------------------------------

    def apply_delta(self, delta: DeltaBatch) -> None:
        """Apply delta operations to playbook (thread-safe)."""
        with self._lock:
            self._playbook.apply_delta(delta)

    def tag_bullet(self, bullet_id: str, tag: str, increment: int = 1) -> Optional[Any]:
        """Tag a bullet (thread-safe)."""
        with self._lock:
            return self._playbook.tag_bullet(bullet_id, tag, increment)

    def add_bullet(
        self,
        section: str,
        content: str,
        bullet_id: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> Any:
        """Add a bullet (thread-safe)."""
        with self._lock:
            return self._playbook.add_bullet(section, content, bullet_id, metadata)

    def update_bullet(
        self,
        bullet_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> Optional[Any]:
        """Update a bullet (thread-safe)."""
        with self._lock:
            return self._playbook.update_bullet(
                bullet_id, content=content, metadata=metadata
            )

    def remove_bullet(self, bullet_id: str) -> None:
        """Remove a bullet (thread-safe)."""
        with self._lock:
            self._playbook.remove_bullet(bullet_id)


# ---------------------------------------------------------------------------
# Async Learning Pipeline
# ---------------------------------------------------------------------------


class AsyncLearningPipeline:
    """Parallel Reflectors + Serialized Curator pipeline.

    This class orchestrates async learning with:
    1. ThreadPoolExecutor for parallel Reflector.reflect() calls
    2. Single Curator thread processing a queue sequentially
    3. Thread-safe playbook wrapper for safe concurrent access

    Flow:
        1. Main thread submits LearningTask via submit()
        2. ThreadPoolExecutor runs Reflector.reflect() in parallel
        3. ReflectionResult queued to Curator
        4. Single Curator thread processes queue sequentially

    Example:
        >>> pipeline = AsyncLearningPipeline(
        ...     playbook=playbook,
        ...     reflector=reflector,
        ...     curator=curator,
        ...     max_reflector_workers=3,
        ... )
        >>> pipeline.start()
        >>> pipeline.submit(task)  # Non-blocking
        >>> pipeline.wait_for_completion()
        >>> pipeline.stop()
    """

    def __init__(
        self,
        playbook: "Playbook",
        reflector: "Reflector",
        curator: "Curator",
        *,
        max_reflector_workers: int = 3,
        curator_queue_size: int = 100,
        max_refinement_rounds: int = 1,
        on_error: Optional[Callable[[Exception, LearningTask], None]] = None,
        on_complete: Optional[Callable[[LearningTask, Any], None]] = None,
    ) -> None:
        """Initialize the async learning pipeline.

        Args:
            playbook: Playbook instance to update
            reflector: Reflector instance for analysis
            curator: Curator instance for playbook updates
            max_reflector_workers: Max concurrent Reflector threads (default: 3)
            curator_queue_size: Max pending Curator tasks (default: 100)
            max_refinement_rounds: Reflector refinement rounds (default: 1)
            on_error: Callback for task errors (task, exception)
            on_complete: Callback for task completion (task, curator_output)
        """
        self._playbook = ThreadSafePlaybook(playbook)
        self._reflector = reflector
        self._curator = curator
        self._max_reflector_workers = max_reflector_workers
        self._max_refinement_rounds = max_refinement_rounds
        self._on_error = on_error
        self._on_complete = on_complete

        # Thread pool for parallel Reflectors
        self._reflector_pool: Optional[ThreadPoolExecutor] = None

        # Queue for Curator (serialized processing)
        self._curator_queue: Queue[ReflectionResult] = Queue(maxsize=curator_queue_size)

        # Curator thread
        self._curator_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Stats
        self._tasks_submitted = 0
        self._reflections_completed = 0
        self._curations_completed = 0
        self._tasks_failed = 0
        self._stats_lock = threading.Lock()

        # Track pending futures for wait_for_completion
        self._pending_futures: List[Future] = []
        self._futures_lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Start the async learning pipeline."""
        if self._reflector_pool is not None:
            logger.warning("AsyncLearningPipeline already started")
            return

        self._stop_event.clear()

        # Start thread pool for Reflectors
        self._reflector_pool = ThreadPoolExecutor(
            max_workers=self._max_reflector_workers,
            thread_name_prefix="ace-reflector",
        )

        # Start Curator thread
        self._curator_thread = threading.Thread(
            target=self._curator_loop,
            daemon=True,
            name="ace-curator",
        )
        self._curator_thread.start()

        logger.info(
            f"AsyncLearningPipeline started with {self._max_reflector_workers} "
            f"Reflector workers"
        )

    def stop(self, wait: bool = True, timeout: float = 30.0) -> int:
        """Stop the async learning pipeline.

        Args:
            wait: If True, wait for pending tasks to complete
            timeout: Max seconds to wait for completion

        Returns:
            Number of tasks remaining in queues
        """
        if self._reflector_pool is None:
            return 0

        if wait:
            self.wait_for_completion(timeout=timeout)

        # Signal stop
        self._stop_event.set()

        # Shutdown thread pool
        self._reflector_pool.shutdown(wait=False)
        self._reflector_pool = None

        # Wait for Curator thread
        if self._curator_thread and self._curator_thread.is_alive():
            self._curator_thread.join(timeout=min(timeout, 5.0))

        remaining = self._curator_queue.qsize()
        logger.info(f"AsyncLearningPipeline stopped, {remaining} tasks remaining")
        return remaining

    def is_running(self) -> bool:
        """Check if the pipeline is running."""
        return self._reflector_pool is not None and not self._stop_event.is_set()

    # -----------------------------------------------------------------------
    # Task Submission
    # -----------------------------------------------------------------------

    def submit(self, task: LearningTask) -> Optional[Future]:
        """Submit a learning task (non-blocking).

        Args:
            task: LearningTask containing sample results to learn from

        Returns:
            Future for tracking completion, or None if pipeline not running
        """
        if self._reflector_pool is None:
            logger.warning("Cannot submit task: pipeline not started")
            return None

        with self._stats_lock:
            self._tasks_submitted += 1

        # Submit to thread pool for Reflector processing
        future = self._reflector_pool.submit(self._reflector_worker, task)

        # Track future for wait_for_completion
        with self._futures_lock:
            self._pending_futures.append(future)
            # Clean up completed futures
            self._pending_futures = [f for f in self._pending_futures if not f.done()]

        return future

    # -----------------------------------------------------------------------
    # Synchronization
    # -----------------------------------------------------------------------

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all pending learning tasks to complete.

        Args:
            timeout: Max seconds to wait (None = wait forever)

        Returns:
            True if all tasks completed, False if timeout
        """
        start_time = time.time()

        # Wait for all Reflector futures to complete
        with self._futures_lock:
            pending = list(self._pending_futures)

        for future in pending:
            remaining_timeout = None
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout - elapsed)
                if remaining_timeout <= 0:
                    return False
            try:
                future.result(timeout=remaining_timeout)
            except Exception:
                pass  # Errors already handled in worker

        # Wait for Curator queue to drain
        try:
            remaining_timeout = None
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout - elapsed)

            # Poll queue until empty or timeout
            poll_start = time.time()
            while not self._curator_queue.empty():
                if remaining_timeout is not None:
                    if time.time() - poll_start > remaining_timeout:
                        return False
                time.sleep(0.1)

            return True
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        with self._stats_lock:
            return {
                "tasks_submitted": self._tasks_submitted,
                "reflections_completed": self._reflections_completed,
                "curations_completed": self._curations_completed,
                "tasks_failed": self._tasks_failed,
                "curator_queue_size": self._curator_queue.qsize(),
                "is_running": self.is_running(),
            }

    # -----------------------------------------------------------------------
    # Internal Workers
    # -----------------------------------------------------------------------

    def _reflector_worker(self, task: LearningTask) -> Optional[ReflectionResult]:
        """Run Reflector.reflect() - executes in thread pool.

        Can have multiple instances running concurrently.
        """
        try:
            # Run reflection (safe to parallelize - reads only)
            reflection = self._reflector.reflect(
                question=task.sample.question,
                generator_output=task.generator_output,
                playbook=self._playbook.playbook,  # Read-only access
                ground_truth=task.environment_result.ground_truth,
                feedback=task.environment_result.feedback,
                max_refinement_rounds=self._max_refinement_rounds,
            )

            # Create result
            result = ReflectionResult(task=task, reflection=reflection)

            # Queue for Curator
            try:
                self._curator_queue.put(result, timeout=10.0)
            except Full:
                logger.warning(
                    f"Curator queue full, dropping reflection for sample "
                    f"{task.step_index}"
                )
                with self._stats_lock:
                    self._tasks_failed += 1
                return None

            with self._stats_lock:
                self._reflections_completed += 1

            return result

        except Exception as e:
            logger.warning(f"Reflector failed for sample {task.step_index}: {e}")
            with self._stats_lock:
                self._tasks_failed += 1

            if self._on_error:
                try:
                    self._on_error(e, task)
                except Exception:
                    pass  # Don't let callback errors propagate

            return None

    def _curator_loop(self) -> None:
        """Single Curator thread - processes ReflectionResults sequentially.

        Only one instance runs at a time to serialize playbook updates.
        """
        while not self._stop_event.is_set():
            try:
                # Block with timeout to check stop_event periodically
                result = self._curator_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                self._process_curation(result)
            except Exception as e:
                logger.warning(
                    f"Curator failed for sample {result.task.step_index}: {e}"
                )
                with self._stats_lock:
                    self._tasks_failed += 1

                if self._on_error:
                    try:
                        self._on_error(e, result.task)
                    except Exception:
                        pass
            finally:
                self._curator_queue.task_done()

    def _process_curation(self, result: ReflectionResult) -> None:
        """Process a single reflection result through Curator.

        Runs in the single Curator thread - serialized execution.
        """
        task = result.task
        reflection = result.reflection

        # Apply bullet tags (thread-safe)
        for tag in reflection.bullet_tags:
            try:
                self._playbook.tag_bullet(tag.id, tag.tag)
            except ValueError:
                continue  # Bullet not found, skip

        # Build question context
        question_context = self._build_question_context(task)
        progress = self._build_progress_string(task)

        # Run Curator (sees latest playbook state)
        curator_output = self._curator.curate(
            reflection=reflection,
            playbook=self._playbook.playbook,  # Pass underlying playbook
            question_context=question_context,
            progress=progress,
        )

        # Apply delta (thread-safe)
        self._playbook.apply_delta(curator_output.delta)

        with self._stats_lock:
            self._curations_completed += 1

        # Completion callback
        if self._on_complete:
            try:
                self._on_complete(task, curator_output)
            except Exception:
                pass  # Don't let callback errors propagate

    def _build_question_context(self, task: LearningTask) -> str:
        """Build question context string for Curator."""
        parts = [
            f"question: {task.sample.question}",
            f"context: {task.sample.context}",
            f"metadata: {json.dumps(task.sample.metadata)}",
            f"feedback: {task.environment_result.feedback}",
            f"ground_truth: {task.environment_result.ground_truth}",
        ]
        return "\n".join(parts)

    def _build_progress_string(self, task: LearningTask) -> str:
        """Build progress string for Curator."""
        return (
            f"epoch {task.epoch}/{task.total_epochs} · "
            f"sample {task.step_index}/{task.total_steps}"
        )
