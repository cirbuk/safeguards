"""Security validation system for input sanitization and output encoding.

This module provides:
- Input sanitization
- Output encoding
- Resource access validation
- Security policy enforcement
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern, Set, Union
from decimal import Decimal

from pydantic import BaseModel, Field, validator


class SecurityPolicy(BaseModel):
    """Security policy configuration."""

    # Input validation
    max_string_length: int = Field(default=10000, gt=0)
    max_number_value: float = Field(default=1e9)
    allowed_schemes: Set[str] = Field(default={"http", "https"})
    blocked_domains: Set[str] = Field(default=set())

    # Resource access
    max_file_size_bytes: int = Field(default=10 * 1024 * 1024)  # 10MB
    allowed_file_types: Set[str] = Field(default={".txt", ".log", ".json", ".yaml"})
    blocked_paths: Set[str] = Field(default={"/etc", "/var/log", "/root"})

    # Rate limiting
    max_requests_per_minute: int = Field(default=60, gt=0)
    max_concurrent_operations: int = Field(default=10, gt=0)


class InputValidator:
    """Validates and sanitizes input data."""

    def __init__(self, policy: SecurityPolicy):
        """Initialize validator with security policy.

        Args:
            policy: Security policy to enforce
        """
        self.policy = policy

        # Common validation patterns
        self._patterns = {
            "agent_id": re.compile(r"^[a-zA-Z0-9_-]{1,64}$"),
            "filename": re.compile(r"^[a-zA-Z0-9_.-]{1,255}$"),
            "path": re.compile(r"^[a-zA-Z0-9/_.-]{1,1024}$"),
            "url": re.compile(
                r"^https?://[a-zA-Z0-9-.]+(:[0-9]+)?(/[a-zA-Z0-9_./-]*)?(\?[a-zA-Z0-9=&_.-]*)?$"
            ),
        }

    def validate_string(
        self,
        value: str,
        pattern: Optional[Union[str, Pattern]] = None,
        max_length: Optional[int] = None,
    ) -> str:
        """Validate and sanitize a string value.

        Args:
            value: String to validate
            pattern: Optional regex pattern name or compiled pattern
            max_length: Optional maximum length override

        Returns:
            Sanitized string

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, str):
            raise ValueError("Value must be a string")

        # Check length
        max_len = max_length or self.policy.max_string_length
        if len(value) > max_len:
            raise ValueError(f"String exceeds maximum length of {max_len}")

        # Apply pattern matching
        if pattern:
            if isinstance(pattern, str):
                if pattern not in self._patterns:
                    raise ValueError(f"Unknown pattern: {pattern}")
                pattern = self._patterns[pattern]

            if not pattern.match(value):
                raise ValueError(f"String does not match required pattern")

        return value

    def validate_number(
        self,
        value: Union[int, float, Decimal],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> Union[int, float, Decimal]:
        """Validate a numeric value.

        Args:
            value: Number to validate
            min_value: Optional minimum value
            max_value: Optional maximum value override

        Returns:
            Validated number

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, (int, float, Decimal)):
            raise ValueError("Value must be a number")

        # Convert to float for comparison
        float_val = float(value)

        # Check bounds
        if min_value is not None and float_val < min_value:
            raise ValueError(f"Number below minimum value of {min_value}")

        max_val = max_value or self.policy.max_number_value
        if float_val > max_val:
            raise ValueError(f"Number exceeds maximum value of {max_val}")

        return value

    def validate_url(self, url: str) -> str:
        """Validate a URL string.

        Args:
            url: URL to validate

        Returns:
            Validated URL

        Raises:
            ValueError: If validation fails
        """
        # Basic pattern check
        if not self._patterns["url"].match(url):
            raise ValueError("Invalid URL format")

        # Check scheme
        scheme = url.split("://")[0]
        if scheme not in self.policy.allowed_schemes:
            raise ValueError(f"URL scheme {scheme} not allowed")

        # Check domain
        domain = url.split("://")[1].split("/")[0].split(":")[0]
        if domain in self.policy.blocked_domains:
            raise ValueError(f"Domain {domain} is blocked")

        return url

    def validate_file_access(
        self,
        path: str,
        check_size: bool = True,
        operation: str = "read",
    ) -> str:
        """Validate file access request.

        Args:
            path: File path to validate
            check_size: Whether to check file size
            operation: Type of access ("read" or "write")

        Returns:
            Validated path

        Raises:
            ValueError: If validation fails
        """
        # Check path pattern
        if not self._patterns["path"].match(path):
            raise ValueError("Invalid path format")

        # Check blocked paths
        if any(path.startswith(blocked) for blocked in self.policy.blocked_paths):
            raise ValueError("Access to path is blocked")

        # Check file extension
        ext = path.split(".")[-1] if "." in path else ""
        if f".{ext}" not in self.policy.allowed_file_types:
            raise ValueError(f"File type .{ext} not allowed")

        # Check file size for reads
        if check_size and operation == "read":
            try:
                import os

                size = os.path.getsize(path)
                if size > self.policy.max_file_size_bytes:
                    raise ValueError(
                        f"File size {size} exceeds maximum of "
                        f"{self.policy.max_file_size_bytes}"
                    )
            except OSError:
                pass  # Ignore size check if file doesn't exist

        return path


class OutputEncoder:
    """Encodes output data for security."""

    @staticmethod
    def encode_html(value: str) -> str:
        """HTML encode a string value.

        Args:
            value: String to encode

        Returns:
            HTML encoded string
        """
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

    @staticmethod
    def encode_json(value: Any) -> str:
        """JSON encode a value with additional escaping.

        Args:
            value: Value to encode

        Returns:
            JSON encoded string
        """
        import json

        return json.dumps(
            value,
            default=str,
            ensure_ascii=True,
            separators=(",", ":"),  # Use compact separators to remove spaces
        )

    @staticmethod
    def encode_shell(value: str) -> str:
        """Shell escape a string value.

        Args:
            value: String to escape

        Returns:
            Shell escaped string
        """
        import shlex

        return shlex.quote(value)

    @staticmethod
    def encode_path(value: str) -> str:
        """Encode a path string.

        Args:
            value: Path to encode

        Returns:
            Encoded path string
        """
        import urllib.parse

        return urllib.parse.quote(value, safe="/")


class SecurityValidator:
    """Main security validation interface."""

    def __init__(self, policy: Optional[SecurityPolicy] = None):
        """Initialize validator.

        Args:
            policy: Optional security policy override
        """
        self.policy = policy or SecurityPolicy()
        self.input_validator = InputValidator(self.policy)
        self.output_encoder = OutputEncoder()

    def validate_agent_input(
        self,
        agent_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate agent input data.

        Args:
            agent_id: ID of agent receiving input
            input_data: Input data to validate

        Returns:
            Validated input data

        Raises:
            ValueError: If validation fails
        """
        # Validate agent ID
        self.input_validator.validate_string(agent_id, "agent_id")

        # Validate input fields
        validated = {}
        for key, value in input_data.items():
            if isinstance(value, str):
                validated[key] = self.input_validator.validate_string(value)
            elif isinstance(value, (int, float, Decimal)):
                validated[key] = self.input_validator.validate_number(value)
            elif isinstance(value, dict):
                validated[key] = self.validate_agent_input(agent_id, value)
            elif isinstance(value, list):
                validated[key] = [
                    (
                        self.validate_agent_input(agent_id, item)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value
                ]
            else:
                validated[key] = value

        return validated

    def encode_agent_output(
        self,
        agent_id: str,
        output_data: Dict[str, Any],
        encoding: str = "json",
    ) -> str:
        """Encode agent output data.

        Args:
            agent_id: ID of agent producing output
            output_data: Output data to encode
            encoding: Type of encoding to apply

        Returns:
            Encoded output string

        Raises:
            ValueError: If encoding fails
        """
        # Validate agent ID
        self.input_validator.validate_string(agent_id, "agent_id")

        # Apply encoding
        if encoding == "json":
            return self.output_encoder.encode_json(output_data)
        elif encoding == "html":
            return self.output_encoder.encode_html(str(output_data))
        elif encoding == "shell":
            return self.output_encoder.encode_shell(str(output_data))
        else:
            raise ValueError(f"Unknown encoding type: {encoding}")

    def validate_resource_access(
        self,
        agent_id: str,
        resource_type: str,
        resource_id: str,
        operation: str,
    ) -> None:
        """Validate resource access request.

        Args:
            agent_id: ID of requesting agent
            resource_type: Type of resource
            resource_id: Resource identifier
            operation: Type of access

        Raises:
            ValueError: If validation fails
        """
        # Validate agent ID
        self.input_validator.validate_string(agent_id, "agent_id")

        # Validate resource type
        self.input_validator.validate_string(resource_type)

        # Validate resource ID
        if resource_type == "file":
            self.input_validator.validate_file_access(
                resource_id,
                operation=operation,
            )
        else:
            self.input_validator.validate_string(resource_id)
