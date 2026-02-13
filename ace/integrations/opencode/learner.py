"""OpenCode ACE Learning — learns from OpenCode sessions via REST API.

This module enables ACE learning from OpenCode sessions by fetching
messages via the OpenCode API, running a Reflector + SkillManager pipeline,
and persisting strategies to a project-scoped skillbook and SKILL.md.

Usage:
    1. Run an OpenCode session (autonomous or interactive)
    2. Trigger /ace-learning on the thread (or run ace-learn-opencode)
    3. Skillbook is updated and SKILL.md is written for future sessions
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Load .env
_env_paths = [
    Path.home() / ".ace" / ".env",
    Path.cwd() / ".env",
]
for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(_env_path)
        break

from ...skillbook import Skillbook
from ...roles import Reflector, SkillManager, AgentOutput
from ...prompt_manager import PromptManager
from .client import OpenCodeClient, OpenCodeClientError
from .transcript import (
    filter_messages,
    messages_to_toon,
    extract_feedback,
    extract_last_user_prompt,
    extract_session_stats,
)
from .prompts import OPENCODE_REFLECTOR_PROMPT
from .skill_writer import write_skill_md, write_memory_summary

logger = logging.getLogger(__name__)

# Minimum filtered parts to consider a session worth learning from
MIN_PARTS_FOR_LEARNING = 5


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
        ) as tmp:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


class OpenCodeLearner:
    """Main class for learning from OpenCode sessions.

    Fetches messages from the OpenCode API, runs Reflector + SkillManager,
    and persists the skillbook and SKILL.md.

    Skillbook scoping: user x project. Each user has independent skillbooks
    per project, stored at:
        {workspace}/.kortix/ace/users/{user_id}/projects/{project_id}/skillbook.json

    Usage:
        learner = OpenCodeLearner(opencode_url="http://localhost:3111")
        learner.learn_from_session("session-id-here")
    """

    def __init__(
        self,
        opencode_url: str = "http://localhost:3111",
        workspace_path: str = "/workspace",
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        ace_llm: Optional[Any] = None,
        model: str = "anthropic/claude-sonnet-4-20250514",
        skill_output_path: Optional[str] = None,
    ):
        """Initialize the learner.

        Args:
            opencode_url: OpenCode API base URL.
            workspace_path: Workspace root inside the sandbox.
            user_id: User ID for scoped skillbooks (default: from env or "default").
            project_id: Project ID (auto-detected from session if not set).
            ace_llm: Custom LLM client (default: LiteLLMClient with model).
            model: LLM model for Reflector/SkillManager.
            skill_output_path: Custom path for SKILL.md output.
        """
        self.opencode_url = opencode_url
        self.workspace = Path(workspace_path)
        self.user_id = user_id or os.environ.get("ACE_USER_ID", "default")
        self.project_id = project_id
        self.model = model
        self.skill_output_path = skill_output_path

        # OpenCode API client
        self.client = OpenCodeClient(base_url=opencode_url)

        # ACE base dir
        self.ace_base = self.workspace / ".kortix" / "ace"

        # LLM client
        if ace_llm:
            self.ace_llm = ace_llm
        else:
            from ...llm_providers.litellm_client import LiteLLMClient
            self.ace_llm = LiteLLMClient(model=model)

        # Roles
        prompt_mgr = PromptManager()
        self.reflector = Reflector(
            self.ace_llm, prompt_template=OPENCODE_REFLECTOR_PROMPT
        )
        self.skill_manager = SkillManager(
            self.ace_llm, prompt_template=prompt_mgr.get_skill_manager_prompt()
        )

        # Learned sessions tracking
        self._learned_sessions_path = self.ace_base / "learned_sessions.json"

    # ------------------------------------------------------------------
    # Skillbook path resolution (user x project)
    # ------------------------------------------------------------------

    def _get_skillbook_path(self, project_id: str) -> Path:
        """Get the skillbook path for a user+project pair."""
        return (
            self.ace_base
            / "users"
            / self.user_id
            / "projects"
            / project_id
            / "skillbook.json"
        )

    def _load_skillbook(self, project_id: str) -> Skillbook:
        """Load or create the skillbook for a user+project."""
        path = self._get_skillbook_path(project_id)
        if path.exists():
            sb = Skillbook.load_from_file(str(path))
            logger.info(f"Loaded skillbook with {len(sb.skills())} skills for project={project_id}")
            return sb
        logger.info(f"Creating new skillbook for user={self.user_id}, project={project_id}")
        return Skillbook()

    def _save_skillbook(self, skillbook: Skillbook, project_id: str) -> Path:
        """Save the skillbook atomically."""
        path = self._get_skillbook_path(project_id)
        _atomic_write_text(path, skillbook.dumps())
        logger.info(f"Saved skillbook ({len(skillbook.skills())} skills) to {path}")
        return path

    # ------------------------------------------------------------------
    # Learned sessions tracking
    # ------------------------------------------------------------------

    def _is_already_learned(self, session_id: str) -> bool:
        """Check if a session has already been processed."""
        if not self._learned_sessions_path.exists():
            return False
        try:
            data = json.loads(self._learned_sessions_path.read_text())
            return session_id in data.get("sessions", [])
        except (json.JSONDecodeError, IOError):
            return False

    def _mark_as_learned(self, session_id: str) -> None:
        """Mark a session as learned."""
        data = {"sessions": []}
        if self._learned_sessions_path.exists():
            try:
                data = json.loads(self._learned_sessions_path.read_text())
            except (json.JSONDecodeError, IOError):
                pass

        if session_id not in data.get("sessions", []):
            data.setdefault("sessions", []).append(session_id)
        _atomic_write_text(self._learned_sessions_path, json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Reflector + SkillManager with retry
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _run_reflector_with_retry(
        self, task: str, agent_output: AgentOutput, feedback: str, skillbook: Skillbook
    ):
        return self.reflector.reflect(
            question=task,
            agent_output=agent_output,
            skillbook=skillbook,
            ground_truth=None,
            feedback=feedback,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _run_skill_manager_with_retry(
        self, reflection, skillbook: Skillbook, context: str, progress: str
    ):
        return self.skill_manager.update_skills(
            reflection=reflection,
            skillbook=skillbook,
            question_context=context,
            progress=progress,
        )

    # ------------------------------------------------------------------
    # Main learning methods
    # ------------------------------------------------------------------

    def learn_from_session(self, session_id: str, force: bool = False) -> bool:
        """Learn from a specific OpenCode session.

        Args:
            session_id: The session ID to learn from.
            force: If True, relearn even if already processed.

        Returns:
            True if learning succeeded, False otherwise.
        """
        try:
            # Check if already learned
            if not force and self._is_already_learned(session_id):
                logger.info(f"Session {session_id} already learned, skipping (use force=True to override)")
                return True

            # Fetch session metadata
            logger.info(f"Fetching session {session_id}...")
            session = self.client.get_session(session_id)
            project_id = self.project_id or session.get("projectID") or "global"

            # Fetch messages
            logger.info("Fetching messages...")
            messages = self.client.get_messages(session_id)

            if not messages:
                logger.info("No messages in session, skipping")
                return True

            # Filter and check for trivial sessions
            stats = extract_session_stats(messages)
            total_parts = stats["text_parts"] + stats["tool_count"]
            if total_parts < MIN_PARTS_FOR_LEARNING:
                logger.info(
                    f"Skipping trivial session ({total_parts} parts, minimum {MIN_PARTS_FOR_LEARNING})"
                )
                return True

            # TOON-encode the filtered trace
            logger.info("Compressing session trace...")
            toon_trace = messages_to_toon(messages)

            # Extract metadata
            task = extract_last_user_prompt(messages)
            feedback = extract_feedback(messages)

            # Load skillbook for this user+project
            skillbook = self._load_skillbook(project_id)

            # Build AgentOutput for the Reflector
            agent_output = AgentOutput(
                reasoning=toon_trace,
                final_answer="(see trace)",
                skill_ids=[],
                raw={
                    "session_id": session_id,
                    "project_id": project_id,
                    "stats": stats,
                },
            )

            # Run Reflector
            logger.info("Running Reflector...")
            reflection = self._run_reflector_with_retry(
                task=task,
                agent_output=agent_output,
                feedback=feedback,
                skillbook=skillbook,
            )

            # Run SkillManager
            logger.info("Running SkillManager...")
            sm_output = self._run_skill_manager_with_retry(
                reflection=reflection,
                skillbook=skillbook,
                context=f"OpenCode session in project {project_id}",
                progress=f"session {session_id[:8]}..., {stats['message_count']} messages",
            )

            # Apply update
            skillbook.apply_update(sm_output.update)

            # Persist
            self._save_skillbook(skillbook, project_id)

            # Write SKILL.md (OpenCode skills live at /opt/opencode/skills/)
            skill_path = self.skill_output_path or "/opt/opencode/skills/KORTIX-ace/SKILL.md"
            write_skill_md(skillbook, skill_path)

            # Write memory summary
            memory_path = str(self.workspace / ".kortix" / "memory" / "ace-strategies.md")
            write_memory_summary(skillbook, memory_path)

            # Mark as learned
            self._mark_as_learned(session_id)

            logger.info(
                f"Learning complete! Skillbook now has {len(skillbook.skills())} skills"
            )
            return True

        except OpenCodeClientError as e:
            logger.error(f"OpenCode API error: {e}")
            return False
        except Exception as e:
            logger.error(f"Learning failed: {e}", exc_info=True)
            return False

    def learn_from_latest(self, project_id: Optional[str] = None) -> bool:
        """Learn from the most recent OpenCode session.

        Args:
            project_id: If provided, filter to sessions for this project.

        Returns:
            True if learning succeeded, False otherwise.
        """
        session = self.client.get_latest_session(project_id=project_id)
        if not session:
            logger.warning("No sessions found")
            return False

        session_id = session.get("id")
        if not session_id:
            logger.error("Session has no ID")
            return False

        logger.info(f"Latest session: {session_id}")
        return self.learn_from_session(session_id)

    # ------------------------------------------------------------------
    # Insight management
    # ------------------------------------------------------------------

    def get_insights(self, project_id: Optional[str] = None) -> dict:
        """Get current insights for a project.

        Returns:
            Dict with 'skills' list and 'sections' grouped dict.
        """
        pid = project_id or self.project_id or "global"
        skillbook = self._load_skillbook(pid)
        skills = skillbook.skills()

        sections: dict = {}
        for skill in skills:
            section = skill.section
            if section not in sections:
                sections[section] = []
            sections[section].append(skill)

        return {"skills": skills, "sections": sections, "count": len(skills)}

    def remove_insight(self, insight_id: str, project_id: Optional[str] = None) -> bool:
        """Remove a specific insight by ID."""
        pid = project_id or self.project_id or "global"
        skillbook = self._load_skillbook(pid)
        skills = skillbook.skills()

        target = None
        for s in skills:
            if s.id == insight_id or insight_id in s.id or insight_id.lower() in s.content.lower():
                target = s
                break

        if not target:
            return False

        skillbook.remove_skill(target.id)
        self._save_skillbook(skillbook, pid)

        # Rewrite outputs
        skill_path = self.skill_output_path or "/opt/opencode/skills/KORTIX-ace/SKILL.md"
        write_skill_md(skillbook, skill_path)

        memory_path = str(self.workspace / ".kortix" / "memory" / "ace-strategies.md")
        write_memory_summary(skillbook, memory_path)

        return True

    def clear_insights(self, project_id: Optional[str] = None) -> None:
        """Clear all insights for a project."""
        pid = project_id or self.project_id or "global"
        skillbook = Skillbook()
        self._save_skillbook(skillbook, pid)

        skill_path = self.skill_output_path or "/opt/opencode/skills/KORTIX-ace/SKILL.md"
        write_skill_md(skillbook, skill_path)

        memory_path = str(self.workspace / ".kortix" / "memory" / "ace-strategies.md")
        write_memory_summary(skillbook, memory_path)

    def inject_skill(self, project_id: Optional[str] = None) -> Optional[str]:
        """Regenerate SKILL.md from the current skillbook.

        Returns:
            Path to the written SKILL.md, or None if no skills.
        """
        pid = project_id or self.project_id or "global"
        skillbook = self._load_skillbook(pid)

        if not skillbook.skills():
            return None

        skill_path = self.skill_output_path or "/opt/opencode/skills/KORTIX-ace/SKILL.md"
        write_skill_md(skillbook, skill_path)

        memory_path = str(self.workspace / ".kortix" / "memory" / "ace-strategies.md")
        write_memory_summary(skillbook, memory_path)

        return skill_path
