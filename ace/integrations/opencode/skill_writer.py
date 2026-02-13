"""SKILL.md and memory output writers for OpenCode ACE integration.

Writes learned strategies to:
1. /opt/opencode/skills/KORTIX-ace/SKILL.md — OpenCode skill format
2. /workspace/.kortix/memory/ace-strategies.md — Memory file for grep/semantic search
"""

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ...skillbook import Skillbook

logger = logging.getLogger(__name__)

# SKILL.md YAML frontmatter
SKILL_FRONTMATTER = """\
---
name: kortix-ace
description: "ACE learned strategies from past sessions. Actionable rules and
  procedures extracted from previous task executions. Load when starting complex
  tasks to benefit from accumulated experience."
---"""

# Memory summary template
MEMORY_TEMPLATE = """\
# ACE Learned Strategies

{n} strategies learned from past sessions.
Load the `kortix-ace` skill for full details.

Last updated: {timestamp}.
Managed by ACE. Do not edit manually.
"""


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


def write_skill_md(skillbook: Skillbook, output_path: str) -> Path:
    """Write the SKILL.md file in OpenCode skill format.

    Uses markdown (not TOON) because SKILL.md body is only loaded on-demand
    when the skill is triggered — token compression is less critical here.

    Args:
        skillbook: Skillbook with learned strategies.
        output_path: Full path to write SKILL.md.

    Returns:
        Path to the written file.
    """
    path = Path(output_path)
    skills = skillbook.skills()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build skill body
    lines = [SKILL_FRONTMATTER, "", "# ACE Learned Strategies", ""]

    if not skills:
        lines.append("*No strategies learned yet. Run /ace-learning after a session to start learning.*")
    else:
        # Group by section
        sections: dict = {}
        for skill in skills:
            section = skill.section
            if section not in sections:
                sections[section] = []
            sections[section].append(skill)

        for section_name, section_skills in sorted(sections.items()):
            display_name = section_name.replace("_", " ").title()
            lines.append(f"## {display_name}")
            for s in section_skills:
                score = f"*(+{s.helpful}/-{s.harmful})*"
                lines.append(f"- **[{s.id}]** {s.content} {score}")
            lines.append("")

    lines.append(f"*Last updated: {now}. {len(skills)} strategies.*")
    lines.append("*Managed by ACE. Do not edit manually.*")
    lines.append("")

    content = "\n".join(lines)
    _atomic_write_text(path, content)
    logger.info(f"Wrote SKILL.md ({len(skills)} strategies) to {path}")
    return path


def write_memory_summary(skillbook: Skillbook, output_path: str) -> Path:
    """Write a brief memory summary pointing to the skill.

    Keeps the memory system aware of ACE strategies without bloating it.

    Args:
        skillbook: Skillbook with learned strategies.
        output_path: Full path to write the memory file.

    Returns:
        Path to the written file.
    """
    path = Path(output_path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    content = MEMORY_TEMPLATE.format(
        n=len(skillbook.skills()),
        timestamp=now,
    )

    _atomic_write_text(path, content)
    logger.info(f"Wrote memory summary to {path}")
    return path
