"""Human action handling for safeguards."""

import logging
import threading
from collections.abc import Callable
from enum import Enum
from typing import Any

from ..notifications.channels import HumanInTheLoopChannel
from ..types import SafetyAlert


class ActionStatus(Enum):
    """Status of human actions."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    TIMED_OUT = "timed_out"


class HumanAction:
    """Represents an action that requires human approval or input."""

    def __init__(
        self,
        title: str,
        description: str,
        request_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize a human action.

        Args:
            title: Title of the action
            description: Description of the action
            request_id: Unique identifier for this action
            agent_id: ID of the agent requesting the action
            metadata: Additional information about the action
        """
        self.title = title
        self.description = description
        self.request_id = request_id
        self.agent_id = agent_id
        self.metadata = metadata or {}
        self.status = ActionStatus.PENDING
        self.comments = ""
        self.response_data: dict[str, Any] = {}

    def approve(self, comments: str = "") -> None:
        """Approve the action.

        Args:
            comments: Optional comments from the approver
        """
        self.status = ActionStatus.APPROVED
        self.comments = comments

    def reject(self, comments: str = "") -> None:
        """Reject the action.

        Args:
            comments: Optional comments from the rejector
        """
        self.status = ActionStatus.REJECTED
        self.comments = comments

    def modify(self, modifications: dict[str, Any], comments: str = "") -> None:
        """Modify the action with specific changes.

        Args:
            modifications: Dict of modifications to apply
            comments: Optional comments about the modifications
        """
        self.status = ActionStatus.MODIFIED
        self.comments = comments
        self.response_data = modifications


class HumanActionHandler:
    """Manages human actions and their responses."""

    def __init__(self, notification_channel: HumanInTheLoopChannel):
        """Initialize the human action handler.

        Args:
            notification_channel: Channel to send action requests through
        """
        self._channel = notification_channel
        self._pending_actions: dict[str, HumanAction] = {}
        self._callbacks: dict[str, list[Callable[[HumanAction], None]]] = {}
        self._action_timeout_seconds = 300  # Default timeout

        # For blocking operations
        self._action_events: dict[str, threading.Event] = {}

    def set_timeout(self, timeout_seconds: int) -> None:
        """Set the default timeout for actions.

        Args:
            timeout_seconds: Number of seconds to wait for human response
        """
        self._action_timeout_seconds = timeout_seconds

    def request_action(
        self,
        title: str,
        description: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        callbacks: list[Callable[[HumanAction], None]] | None = None,
    ) -> HumanAction:
        """Request a human action.

        Args:
            title: Title of the action
            description: Description of the action
            agent_id: ID of the agent requesting the action
            metadata: Additional information about the action
            callbacks: Functions to call when action is responded to

        Returns:
            Created human action object
        """
        # Create alert and action
        alert = SafetyAlert(
            title=title,
            description=description,
            severity="INFO",  # Use appropriate severity
            metadata=metadata or {},
        )

        # Send through channel
        success = self._channel.send_notification(alert)
        if not success:
            msg = "Failed to send human action request"
            raise RuntimeError(msg)

        # Extract request ID from alert
        request_id = alert.metadata.get("request_id", str(id(alert)))

        # Create action
        action = HumanAction(
            title=title,
            description=description,
            request_id=request_id,
            agent_id=agent_id,
            metadata=metadata,
        )

        # Store action
        self._pending_actions[request_id] = action

        # Set up callback
        self._channel.register_approval_callback(request_id, self._handle_response)

        # Store callbacks
        if callbacks:
            self._callbacks[request_id] = callbacks

        # Create event for blocking operations
        self._action_events[request_id] = threading.Event()

        return action

    def _handle_response(self, request_id: str, approved: bool, comments: str) -> None:
        """Handle a response from a human.

        Args:
            request_id: ID of the request
            approved: Whether the request was approved
            comments: Comments from the human
        """
        if request_id not in self._pending_actions:
            logging.warning(f"Received response for unknown action: {request_id}")
            return

        action = self._pending_actions[request_id]

        # Parse comments for modifications
        modifications = self._parse_modifications(comments)

        if approved and not modifications:
            action.approve(comments)
        elif approved and modifications:
            action.modify(modifications, comments)
        else:
            action.reject(comments)

        # Execute callbacks
        if request_id in self._callbacks:
            for callback in self._callbacks[request_id]:
                try:
                    callback(action)
                except Exception as e:
                    logging.error(f"Error in action callback: {e}")

        # Signal any waiting threads
        if request_id in self._action_events:
            self._action_events[request_id].set()

    def _parse_modifications(self, comments: str) -> dict[str, Any]:
        """Parse modification instructions from comments.

        This is a simple implementation that could be extended with more
        sophisticated parsing logic.

        Args:
            comments: Comments from the human

        Returns:
            Dict of modifications if any were specified
        """
        modifications = {}

        # Simple parsing - look for key:value pairs
        if comments:
            for line in comments.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if key and value:
                        modifications[key] = value

        return modifications

    def wait_for_action(
        self,
        action: HumanAction,
        timeout: int | None = None,
    ) -> ActionStatus:
        """Wait for an action to be responded to.

        Args:
            action: Action to wait for
            timeout: Timeout in seconds, or None to use default

        Returns:
            Final status of the action
        """
        if action.request_id not in self._action_events:
            return action.status

        timeout_seconds = timeout if timeout is not None else self._action_timeout_seconds

        # Wait for response
        event = self._action_events[action.request_id]
        if not event.wait(timeout=timeout_seconds):
            # Timed out
            action.status = ActionStatus.TIMED_OUT

            # Clean up
            self._cleanup_action(action.request_id)

        return action.status

    def _cleanup_action(self, request_id: str) -> None:
        """Clean up resources for an action.

        Args:
            request_id: ID of the action to clean up
        """
        if request_id in self._pending_actions:
            del self._pending_actions[request_id]

        if request_id in self._callbacks:
            del self._callbacks[request_id]

        if request_id in self._action_events:
            del self._action_events[request_id]
