"""OpenCode REST API client.

HTTP client for talking to the OpenCode API running inside a sandbox.
Uses httpx for async-capable HTTP with sync wrappers.

Base URL defaults to http://localhost:3111 (OpenCode API inside sandbox),
configurable via OPENCODE_URL env var or constructor param.
"""

import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_OPENCODE_URL = "http://localhost:3111"


class OpenCodeClientError(Exception):
    """Raised when the OpenCode API returns an error."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


class OpenCodeClient:
    """HTTP client for the OpenCode REST API.

    Endpoints:
        GET /session             — list sessions
        GET /session/{id}        — session metadata
        GET /session/{id}/message — messages for a session
        GET /session/status      — health check
        GET /event               — SSE event stream
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = (
            base_url
            or os.environ.get("OPENCODE_URL")
            or DEFAULT_OPENCODE_URL
        ).rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Core endpoints
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Check if the OpenCode API is reachable."""
        try:
            resp = self._client.get("/session/status")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions, sorted by most recent first."""
        resp = self._client.get("/session")
        resp.raise_for_status()
        sessions = resp.json()
        # Sort by updatedAt descending
        sessions.sort(key=lambda s: s.get("updatedAt", ""), reverse=True)
        return sessions

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session metadata."""
        resp = self._client.get(f"/session/{session_id}")
        if resp.status_code == 404:
            raise OpenCodeClientError(
                f"Session not found: {session_id}", status_code=404
            )
        resp.raise_for_status()
        return resp.json()

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a session.

        Returns list of message objects, each with 'info' and 'parts' fields.
        """
        resp = self._client.get(f"/session/{session_id}/message")
        if resp.status_code == 404:
            raise OpenCodeClientError(
                f"Session not found: {session_id}", status_code=404
            )
        resp.raise_for_status()
        return resp.json()

    def get_latest_session(
        self, project_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get the most recently updated session.

        Args:
            project_id: If provided, filter to sessions for this project.

        Returns:
            Session dict, or None if no sessions exist.
        """
        sessions = self.list_sessions()
        if project_id:
            sessions = [s for s in sessions if s.get("projectID") == project_id]
        return sessions[0] if sessions else None

    # ------------------------------------------------------------------
    # SSE event stream
    # ------------------------------------------------------------------

    def subscribe_events(self) -> Iterator[Dict[str, Any]]:
        """Subscribe to the SSE event stream.

        Yields parsed event dicts from GET /event.
        """
        with httpx.stream(
            "GET",
            f"{self.base_url}/event",
            timeout=None,
        ) as response:
            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    event_text, buffer = buffer.split("\n\n", 1)
                    event = self._parse_sse_event(event_text)
                    if event:
                        yield event

    @staticmethod
    def _parse_sse_event(text: str) -> Optional[Dict[str, Any]]:
        """Parse a single SSE event block into a dict."""
        event_type = None
        data_lines = []

        for line in text.strip().split("\n"):
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif line.startswith("id:"):
                pass  # ignore event IDs

        if not data_lines:
            return None

        data_str = "\n".join(data_lines)
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            data = {"raw": data_str}

        result = {"data": data}
        if event_type:
            result["type"] = event_type
        return result
