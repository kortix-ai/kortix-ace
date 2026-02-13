"""OpenCode integration for ACE — learns from autonomous agent sessions.

This package provides:
- OpenCodeLearner: Learns from OpenCode sessions via REST API
- OpenCodeClient: HTTP client for the OpenCode API
- write_skill_md: Writes learned strategies as an OpenCode skill (SKILL.md)

Quick Start:
    # Learn from the latest OpenCode session
    ace-learn-opencode

    # Learn from a specific session
    ace-learn-opencode --session <session-id>

    # Check prerequisites
    ace-learn-opencode doctor

    # Show learned strategies
    ace-learn-opencode insights

Programmatic usage:
    from ace.integrations.opencode import OpenCodeLearner

    learner = OpenCodeLearner(opencode_url="http://localhost:3111")
    learner.learn_from_session("session-id")
"""

from .learner import OpenCodeLearner
from .client import OpenCodeClient, OpenCodeClientError
from .skill_writer import write_skill_md, write_memory_summary

__all__ = [
    "OpenCodeLearner",
    "OpenCodeClient",
    "OpenCodeClientError",
    "write_skill_md",
    "write_memory_summary",
]
