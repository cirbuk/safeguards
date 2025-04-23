"""Notification channels for safeguards."""

import json
import logging
import queue
import smtplib
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import requests

from ..types import AlertSeverity, SafetyAlert


class NotificationChannel(ABC):
    """Base class for notification channels."""

    @abstractmethod
    def send_notification(self, alert: SafetyAlert) -> bool:
        """Send a notification through this channel.

        Args:
            alert: The alert to send

        Returns:
            True if notification was sent successfully, False otherwise
        """
        pass

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the notification channel.

        Args:
            config: Channel-specific configuration
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        pass


class LoggingChannel(NotificationChannel):
    """Notification channel that logs alerts to Python's logging system."""

    def __init__(self):
        """Initialize the logging channel."""
        self.logger = logging.getLogger("safeguards.notifications")

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the logging channel.

        Args:
            config: Configuration parameters
        """
        log_level = config.get("log_level", "INFO")
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO

        self.logger.setLevel(numeric_level)

        # Add handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def send_notification(self, alert: SafetyAlert) -> bool:
        """Log an alert.

        Args:
            alert: The alert to log

        Returns:
            True if logged successfully
        """
        level_map = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }

        level = level_map.get(alert.severity, logging.INFO)
        self.logger.log(
            level,
            f"{alert.title}: {alert.description}",
            extra={"metadata": alert.metadata},
        )
        return True

    def shutdown(self) -> None:
        """Clean up resources."""
        for handler in self.logger.handlers:
            handler.close()


class EmailChannel(NotificationChannel):
    """Notification channel that sends alerts via email."""

    def __init__(self):
        """Initialize the email channel."""
        self._initialized = False
        self._smtp_server = ""
        self._smtp_port = 587
        self._username = ""
        self._password = ""
        self._sender = ""
        self._recipients = []
        self._use_tls = True

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the email channel.

        Args:
            config: Email configuration including:
                - smtp_server: SMTP server address
                - smtp_port: SMTP server port
                - username: SMTP username
                - password: SMTP password
                - sender: Sender email address
                - recipients: List of recipient email addresses
                - use_tls: Whether to use TLS (default: True)
        """
        self._smtp_server = config.get("smtp_server", "")
        self._smtp_port = config.get("smtp_port", 587)
        self._username = config.get("username", "")
        self._password = config.get("password", "")
        self._sender = config.get("sender", "")
        self._recipients = config.get("recipients", [])
        self._use_tls = config.get("use_tls", True)

        # Validate configuration
        if not (self._smtp_server and self._sender and self._recipients):
            msg = "Missing required email configuration parameters"
            raise ValueError(msg)

        self._initialized = True

    def send_notification(self, alert: SafetyAlert) -> bool:
        """Send an alert via email.

        Args:
            alert: The alert to send

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._initialized:
            return False

        try:
            # Create email message
            msg = MIMEMultipart()
            msg["From"] = self._sender
            msg["To"] = ", ".join(self._recipients)

            # Set subject based on severity
            severity_prefix = {
                AlertSeverity.INFO: "[INFO]",
                AlertSeverity.WARNING: "[WARNING]",
                AlertSeverity.ERROR: "[ERROR]",
                AlertSeverity.CRITICAL: "[CRITICAL]",
            }.get(alert.severity, "[INFO]")

            msg["Subject"] = f"{severity_prefix} {alert.title}"

            # Build email body
            body = f"""
            {alert.description}

            Time: {alert.timestamp}
            Severity: {alert.severity.name}
            """

            # Add metadata if present
            if alert.metadata:
                body += "\nMetadata:\n"
                for key, value in alert.metadata.items():
                    body += f"  {key}: {value}\n"

            msg.attach(MIMEText(body, "plain"))

            # Connect to SMTP server and send email
            with smtplib.SMTP(self._smtp_server, self._smtp_port) as server:
                if self._use_tls:
                    server.starttls()

                # Login if credentials provided
                if self._username and self._password:
                    server.login(self._username, self._password)

                server.send_message(msg)

            return True
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
            return False

    def shutdown(self) -> None:
        """Clean up resources."""
        self._initialized = False


class HumanInTheLoopChannel(NotificationChannel):
    """Notification channel that supports human review and approval workflows."""

    def __init__(self):
        """Initialize the human-in-the-loop channel."""
        self._initialized = False
        self._pending_approvals: dict[str, dict[str, Any]] = {}
        self._approval_callbacks: dict[str, Callable[[str, bool, str], None]] = {}
        self._webhook_url = ""
        self._api_key = ""
        self._timeout_seconds = 300  # 5 minutes default
        self._poll_interval = 5  # 5 seconds

        # Start a background thread for checking responses
        self._response_queue: queue.Queue = queue.Queue()
        self._running = False
        self._worker_thread = None

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the human-in-the-loop channel.

        Args:
            config: Configuration including:
                - webhook_url: URL to send approval requests to
                - api_key: API key for authentication
                - timeout_seconds: Seconds to wait for human response
                - poll_interval: Seconds between polling for responses
        """
        self._webhook_url = config.get("webhook_url", "")
        self._api_key = config.get("api_key", "")
        self._timeout_seconds = config.get("timeout_seconds", 300)
        self._poll_interval = config.get("poll_interval", 5)

        # Validate configuration
        if not self._webhook_url:
            msg = "Missing required webhook_url parameter"
            raise ValueError(msg)

        # Start the background worker
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._check_responses,
            daemon=True,
        )
        self._worker_thread.start()

        self._initialized = True

    def send_notification(self, alert: SafetyAlert) -> bool:
        """Send an alert requiring human review.

        Args:
            alert: The alert to send

        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self._initialized:
            return False

        try:
            # Generate unique ID for this request
            request_id = str(uuid.uuid4())

            # Create approval request
            approval_request = {
                "id": request_id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.name,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata,
                "expires_at": (alert.timestamp.timestamp() + self._timeout_seconds) * 1000,
                "callback_url": f"{self._webhook_url}/callback",
            }

            # Store in pending approvals
            self._pending_approvals[request_id] = approval_request

            # Send to external system
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }

            response = requests.post(
                f"{self._webhook_url}/request-approval",
                headers=headers,
                json=approval_request,
            )

            if response.status_code not in (200, 201, 202):
                logging.error(f"Failed to send approval request: {response.text}")
                return False

            return True
        except Exception as e:
            logging.error(f"Failed to send human-in-the-loop alert: {e}")
            return False

    def register_approval_callback(
        self,
        request_id: str,
        callback: Callable[[str, bool, str], None],
    ) -> None:
        """Register a callback to be called when an approval is received.

        Args:
            request_id: The ID of the approval request
            callback: Function to call with (request_id, approved, comments)
        """
        self._approval_callbacks[request_id] = callback

    def process_response(self, request_id: str, approved: bool, comments: str) -> None:
        """Process a human response to an approval request.

        This can be called directly or via a webhook endpoint.

        Args:
            request_id: The ID of the approval request
            approved: Whether the request was approved
            comments: Any comments from the human reviewer
        """
        self._response_queue.put((request_id, approved, comments))

    def _check_responses(self) -> None:
        """Background worker to check for responses and trigger callbacks."""
        while self._running:
            try:
                # Process any responses in the queue
                while not self._response_queue.empty():
                    request_id, approved, comments = self._response_queue.get()

                    # Execute callback if registered
                    if request_id in self._approval_callbacks:
                        try:
                            self._approval_callbacks[request_id](
                                request_id,
                                approved,
                                comments,
                            )
                        except Exception as e:
                            logging.error(f"Error in approval callback: {e}")

                        # Remove callback after execution
                        del self._approval_callbacks[request_id]

                    # Remove from pending approvals
                    if request_id in self._pending_approvals:
                        del self._pending_approvals[request_id]

                # Check for expired approvals
                current_time = time.time() * 1000  # milliseconds
                expired_requests = []

                for request_id, request in self._pending_approvals.items():
                    if current_time > request["expires_at"]:
                        expired_requests.append(request_id)

                # Process expired requests
                for request_id in expired_requests:
                    if request_id in self._approval_callbacks:
                        try:
                            # Call with approved=False to indicate timeout
                            self._approval_callbacks[request_id](
                                request_id,
                                False,
                                "Request timed out",
                            )
                        except Exception as e:
                            logging.error(f"Error in timeout callback: {e}")

                        # Remove callback and request
                        del self._approval_callbacks[request_id]

                    if request_id in self._pending_approvals:
                        del self._pending_approvals[request_id]

                # Sleep before next check
                time.sleep(self._poll_interval)
            except Exception as e:
                logging.error(f"Error in response checker: {e}")
                time.sleep(1)  # Sleep briefly on error

    def shutdown(self) -> None:
        """Stop the background thread and clean up resources."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2)
        self._initialized = False
        self._pending_approvals.clear()
        self._approval_callbacks.clear()


class SlackChannel(NotificationChannel):
    """Notification channel that sends alerts to Slack."""

    def __init__(self):
        """Initialize the Slack channel."""
        self._initialized = False
        self._webhook_url = ""
        self._channel = ""
        self._username = "Safeguards Bot"
        self._icon_emoji = ":shield:"

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the Slack channel.

        Args:
            config: Slack configuration including:
                - webhook_url: Slack webhook URL
                - channel: Channel to post to (optional)
                - username: Bot username (optional)
                - icon_emoji: Bot icon emoji (optional)
        """
        self._webhook_url = config.get("webhook_url", "")
        self._channel = config.get("channel", "")
        self._username = config.get("username", "Safeguards Bot")
        self._icon_emoji = config.get("icon_emoji", ":shield:")

        # Validate configuration
        if not self._webhook_url:
            msg = "Missing required webhook_url parameter"
            raise ValueError(msg)

        self._initialized = True

    def send_notification(self, alert: SafetyAlert) -> bool:
        """Send an alert to Slack.

        Args:
            alert: The alert to send

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._initialized:
            return False

        try:
            # Map severity to color
            color_map = {
                AlertSeverity.INFO: "#36a64f",  # green
                AlertSeverity.WARNING: "#ffcc00",  # yellow
                AlertSeverity.ERROR: "#ff9900",  # orange
                AlertSeverity.CRITICAL: "#ff0000",  # red
            }
            color = color_map.get(alert.severity, "#36a64f")

            # Create message payload
            payload = {
                "username": self._username,
                "icon_emoji": self._icon_emoji,
                "attachments": [
                    {
                        "fallback": f"{alert.title}: {alert.description}",
                        "color": color,
                        "title": alert.title,
                        "text": alert.description,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.name,
                                "short": True,
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.isoformat(),
                                "short": True,
                            },
                        ],
                        "footer": "Safeguards Framework",
                        "ts": int(alert.timestamp.timestamp()),
                    },
                ],
            }

            # Add channel if specified
            if self._channel:
                payload["channel"] = self._channel

            # Add metadata as fields if present
            if alert.metadata:
                for key, value in alert.metadata.items():
                    if isinstance(value, str | int | float | bool):
                        payload["attachments"][0]["fields"].append(
                            {"title": key, "value": str(value), "short": True},
                        )

            # Send to Slack
            response = requests.post(
                self._webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                logging.error(f"Failed to send Slack notification: {response.text}")
                return False

            return True
        except Exception as e:
            logging.error(f"Failed to send Slack notification: {e}")
            return False

    def shutdown(self) -> None:
        """Clean up resources."""
        self._initialized = False
