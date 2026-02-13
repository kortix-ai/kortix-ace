"""OpenCode message filtering and TOON encoding.

Converts raw OpenCode messages (from GET /session/{id}/message) to a
compressed trace suitable for the ACE Reflector.

Filtering rules are based on OpenCode's part type system:
- text: Keep (strip <system-reminder> blocks and ACE recursive content)
- tool (completed): Keep (serialize as {tool, input_summary, compressed_output, status})
- tool (error): Keep in full (errors are high-signal for learning)
- step-start / step-finish: Skip (LLM invocation metadata)
- reasoning: Skip (encrypted content, not readable)
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Tool result compression settings (same as Claude Code integration)
MAX_RESULT_SIZE = 1000  # chars
HEAD_SIZE = 500
TAIL_SIZE = 200

# Part types to skip entirely
_SKIP_PART_TYPES = {
    "step-start",
    "step-finish",
    "reasoning",
}

# Patterns that indicate ACE recursive content
_ACE_LEARN_PATTERNS = (
    "ace-learn",
    "ace.integrations.opencode",
    "ace.integrations.claude_code",
    "OpenCodeClientError",
    "Learning from session",
    "Learning complete",
    "Learning failed",
    "Running Reflector",
    "Running SkillManager",
    "ACE Doctor",
)


def _compress_tool_result(content: str) -> str:
    """Truncate large tool results, keeping head and tail."""
    if not isinstance(content, str) or len(content) <= MAX_RESULT_SIZE:
        return content
    truncated = len(content) - HEAD_SIZE - TAIL_SIZE
    return f"{content[:HEAD_SIZE]}\n... [{truncated} chars truncated] ...\n{content[-TAIL_SIZE:]}"


def _contains_ace_content(text: str) -> bool:
    """Check if text contains ACE recursive content."""
    return any(pattern in text for pattern in _ACE_LEARN_PATTERNS)


def _strip_system_reminders(text: str) -> str:
    """Remove <system-reminder> blocks from text."""
    return re.sub(
        r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL
    ).strip()


def _filter_text_part(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Filter a text part, stripping system content and ACE recursion."""
    content = part.get("content", "")
    if not isinstance(content, str):
        return None

    content = _strip_system_reminders(content)

    if not content:
        return None
    if _contains_ace_content(content):
        return None

    return {"type": "text", "role": part.get("role", "unknown"), "content": content}


def _filter_tool_part(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Filter a tool-call part, compressing outputs."""
    state = part.get("state", "")

    # Skip pending/running tool calls — only completed ones have results
    if state not in ("completed", "error"):
        return None

    tool_name = part.get("name", "unknown")
    tool_input = part.get("input", {})
    tool_output = part.get("output", "")
    is_error = state == "error" or part.get("error")

    # Summarize input (keep it short)
    if isinstance(tool_input, dict):
        # Keep first few keys as summary
        input_summary = {
            k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v)
            for k, v in list(tool_input.items())[:5]
        }
    elif isinstance(tool_input, str):
        input_summary = tool_input[:200]
    else:
        input_summary = str(tool_input)[:200]

    # Compress output (keep errors in full for high-signal learning)
    if isinstance(tool_output, str):
        compressed_output = tool_output if is_error else _compress_tool_result(tool_output)
    elif isinstance(tool_output, dict):
        output_str = json.dumps(tool_output, separators=(",", ":"))
        compressed_output = output_str if is_error else _compress_tool_result(output_str)
    else:
        compressed_output = str(tool_output)[:MAX_RESULT_SIZE]

    return {
        "type": "tool",
        "name": tool_name,
        "input": input_summary,
        "output": compressed_output,
        "status": "error" if is_error else "ok",
    }


def filter_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter raw OpenCode messages into a clean trace for the Reflector.

    Each message has 'info' (metadata) and 'parts' (content parts).
    We filter parts by type and compress tool outputs.

    Args:
        messages: Raw messages from OpenCode API (GET /session/{id}/message).

    Returns:
        List of filtered message dicts with only relevant parts.
    """
    filtered = []

    for msg in messages:
        parts = msg.get("parts", [])
        info = msg.get("info", {})
        role = info.get("role", "unknown")

        filtered_parts = []
        for part in parts:
            part_type = part.get("type", "")

            # Skip meta part types
            if part_type in _SKIP_PART_TYPES:
                continue

            if part_type == "text":
                # Add role context to text parts
                part_with_role = {**part, "role": role}
                result = _filter_text_part(part_with_role)
                if result:
                    filtered_parts.append(result)

            elif part_type == "tool":
                result = _filter_tool_part(part)
                if result:
                    filtered_parts.append(result)

            # Skip unknown part types silently

        if filtered_parts:
            filtered.append({"role": role, "parts": filtered_parts})

    return filtered


def messages_to_toon(messages: List[Dict[str, Any]]) -> str:
    """Filter messages and encode to TOON format for the Reflector.

    Args:
        messages: Raw messages from OpenCode API.

    Returns:
        TOON-encoded string of filtered messages, or compact JSON fallback.
    """
    filtered = filter_messages(messages)

    try:
        from toon import encode
        return encode(filtered, {"delimiter": "\t"})
    except ImportError:
        logger.warning("TOON not installed, using compact JSON")
        return json.dumps(filtered, separators=(",", ":"))


def extract_feedback(messages: List[Dict[str, Any]]) -> str:
    """Extract feedback summary from messages.

    Counts tool calls, success/error rates, etc.

    Args:
        messages: Raw messages from OpenCode API.

    Returns:
        Human-readable feedback string.
    """
    total_tools = 0
    failed_tools = 0
    tool_names: Dict[str, int] = {}

    for msg in messages:
        for part in msg.get("parts", []):
            if part.get("type") == "tool" and part.get("state") in ("completed", "error"):
                total_tools += 1
                name = part.get("name", "unknown")
                tool_names[name] = tool_names.get(name, 0) + 1
                if part.get("state") == "error" or part.get("error"):
                    failed_tools += 1

    if total_tools == 0:
        return "Session completed: no tool calls recorded"

    success_rate = (total_tools - failed_tools) / total_tools * 100
    feedback = f"Session completed: {total_tools} tool calls, {success_rate:.0f}% success rate"
    if failed_tools > 0:
        feedback += f" ({failed_tools} failures)"

    # Top tools used
    top_tools = sorted(tool_names.items(), key=lambda x: x[1], reverse=True)[:5]
    if top_tools:
        tool_summary = ", ".join(f"{name}({count})" for name, count in top_tools)
        feedback += f". Tools: {tool_summary}"

    return feedback


def extract_last_user_prompt(messages: List[Dict[str, Any]]) -> str:
    """Extract the last non-system user text from messages.

    Args:
        messages: Raw messages from OpenCode API.

    Returns:
        Last user prompt text (truncated to 200 chars), or default string.
    """
    last_prompt = "OpenCode session"

    for msg in messages:
        info = msg.get("info", {})
        if info.get("role") != "user":
            continue

        for part in msg.get("parts", []):
            if part.get("type") != "text":
                continue
            content = part.get("content", "")
            if not isinstance(content, str):
                continue

            content = _strip_system_reminders(content)
            if content and not _contains_ace_content(content):
                last_prompt = content[:200]

    return last_prompt


def extract_session_stats(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract session statistics from messages.

    Returns:
        Dict with keys: message_count, tool_count, error_count, tool_names.
    """
    tool_count = 0
    error_count = 0
    tool_names: Dict[str, int] = {}
    text_parts = 0

    for msg in messages:
        for part in msg.get("parts", []):
            ptype = part.get("type", "")
            if ptype == "text":
                text_parts += 1
            elif ptype == "tool" and part.get("state") in ("completed", "error"):
                tool_count += 1
                name = part.get("name", "unknown")
                tool_names[name] = tool_names.get(name, 0) + 1
                if part.get("state") == "error" or part.get("error"):
                    error_count += 1

    return {
        "message_count": len(messages),
        "text_parts": text_parts,
        "tool_count": tool_count,
        "error_count": error_count,
        "tool_names": tool_names,
    }
