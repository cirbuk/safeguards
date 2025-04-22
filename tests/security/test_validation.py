"""Unit tests for security validation system."""

import pytest
import re
from decimal import Decimal

from safeguards.security.validation import (
    SecurityPolicy,
    InputValidator,
    OutputEncoder,
    SecurityValidator,
)


def test_security_policy():
    """Test SecurityPolicy configuration."""
    policy = SecurityPolicy()

    # Check defaults
    assert policy.max_string_length == 10000
    assert policy.max_number_value == 1e9
    assert policy.allowed_schemes == {"http", "https"}
    assert policy.max_file_size_bytes == 10 * 1024 * 1024

    # Test custom values
    custom_policy = SecurityPolicy(
        max_string_length=5000,
        max_number_value=1e6,
        allowed_schemes={"https"},
        max_file_size_bytes=5 * 1024 * 1024,
    )
    assert custom_policy.max_string_length == 5000
    assert custom_policy.max_number_value == 1e6
    assert custom_policy.allowed_schemes == {"https"}
    assert custom_policy.max_file_size_bytes == 5 * 1024 * 1024


class TestInputValidator:
    """Test cases for InputValidator."""

    @pytest.fixture
    def validator(self):
        """Create InputValidator instance for testing."""
        return InputValidator(SecurityPolicy())

    def test_validate_string(self, validator):
        """Test string validation."""
        # Valid strings
        assert validator.validate_string("test") == "test"
        assert validator.validate_string("a" * 1000) == "a" * 1000

        # Invalid strings
        with pytest.raises(ValueError):
            validator.validate_string("a" * 20000)  # Too long

        with pytest.raises(ValueError):
            validator.validate_string(123)  # Not a string

        # Pattern matching
        assert (
            validator.validate_string(
                "test_agent_1",
                pattern="agent_id",
            )
            == "test_agent_1"
        )

        with pytest.raises(ValueError):
            validator.validate_string(
                "invalid@agent",
                pattern="agent_id",
            )

    def test_validate_number(self, validator):
        """Test number validation."""
        # Valid numbers
        assert validator.validate_number(100) == 100
        assert validator.validate_number(1.5) == 1.5
        assert validator.validate_number(Decimal("100.50")) == Decimal("100.50")

        # Invalid numbers
        with pytest.raises(ValueError):
            validator.validate_number("100")  # Not a number

        with pytest.raises(ValueError):
            validator.validate_number(1e10)  # Too large

        # Custom bounds
        assert validator.validate_number(5, min_value=0, max_value=10) == 5

        with pytest.raises(ValueError):
            validator.validate_number(-1, min_value=0)

    def test_validate_url(self, validator):
        """Test URL validation."""
        # Valid URLs
        assert validator.validate_url("https://example.com") == "https://example.com"
        assert (
            validator.validate_url("http://test.com/path?q=1")
            == "http://test.com/path?q=1"
        )

        # Invalid URLs
        with pytest.raises(ValueError):
            validator.validate_url("ftp://example.com")  # Invalid scheme

        with pytest.raises(ValueError):
            validator.validate_url("not_a_url")  # Invalid format

        # Blocked domains
        validator.policy.blocked_domains.add("blocked.com")
        with pytest.raises(ValueError):
            validator.validate_url("https://blocked.com")

    def test_validate_file_access(self, validator):
        """Test file access validation."""
        # Valid paths
        assert (
            validator.validate_file_access(
                "test.txt",
                check_size=False,
            )
            == "test.txt"
        )
        assert (
            validator.validate_file_access(
                "data/test.json",
                check_size=False,
            )
            == "data/test.json"
        )

        # Invalid paths
        with pytest.raises(ValueError):
            validator.validate_file_access("/etc/passwd")  # Blocked path

        with pytest.raises(ValueError):
            validator.validate_file_access("test.exe")  # Invalid extension


class TestOutputEncoder:
    """Test cases for OutputEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create OutputEncoder instance for testing."""
        return OutputEncoder()

    def test_encode_html(self, encoder):
        """Test HTML encoding."""
        assert (
            encoder.encode_html('<script>alert("test")</script>')
            == "&lt;script&gt;alert(&quot;test&quot;)&lt;/script&gt;"
        )

        assert encoder.encode_html("Test & Demo") == "Test &amp; Demo"

    def test_encode_json(self, encoder):
        """Test JSON encoding."""
        data = {
            "str": "test",
            "num": 123,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
        }
        encoded = encoder.encode_json(data)
        assert isinstance(encoded, str)
        assert '"str":"test"' in encoded
        assert '"num":123' in encoded

    def test_encode_shell(self, encoder):
        """Test shell command encoding."""
        assert encoder.encode_shell('echo "hello world"') == "'echo \"hello world\"'"

        assert encoder.encode_shell("rm -rf /") == "'rm -rf /'"

    def test_encode_path(self, encoder):
        """Test path encoding."""
        assert (
            encoder.encode_path("/test/path with spaces/")
            == "/test/path%20with%20spaces/"
        )

        assert encoder.encode_path("/test/special#chars?") == "/test/special%23chars%3F"


class TestSecurityValidator:
    """Test cases for SecurityValidator."""

    @pytest.fixture
    def validator(self):
        """Create SecurityValidator instance for testing."""
        return SecurityValidator()

    def test_validate_agent_input(self, validator):
        """Test agent input validation."""
        input_data = {
            "name": "test_agent",
            "value": 100,
            "nested": {
                "key": "value",
                "num": 50,
            },
            "list": ["a", "b", {"x": 1}],
        }

        validated = validator.validate_agent_input("test_agent", input_data)
        assert validated["name"] == "test_agent"
        assert validated["value"] == 100
        assert validated["nested"]["key"] == "value"
        assert validated["list"][2]["x"] == 1

        # Invalid input
        with pytest.raises(ValueError):
            validator.validate_agent_input(
                "invalid@agent",
                {"key": "value"},
            )

    def test_encode_agent_output(self, validator):
        """Test agent output encoding."""
        output_data = {
            "result": "test",
            "value": 100,
        }

        # JSON encoding
        json_encoded = validator.encode_agent_output(
            "test_agent",
            output_data,
            encoding="json",
        )
        assert isinstance(json_encoded, str)
        assert '"result":"test"' in json_encoded

        # HTML encoding
        html_encoded = validator.encode_agent_output(
            "test_agent",
            {"html": "<test>"},
            encoding="html",
        )
        assert "&lt;test&gt;" in html_encoded

        # Invalid encoding
        with pytest.raises(ValueError):
            validator.encode_agent_output(
                "test_agent",
                output_data,
                encoding="invalid",
            )

    def test_validate_resource_access(self, validator):
        """Test resource access validation."""
        # File access
        validator.validate_resource_access(
            "test_agent",
            "file",
            "test.txt",
            "read",
        )

        with pytest.raises(ValueError):
            validator.validate_resource_access(
                "test_agent",
                "file",
                "/etc/passwd",
                "read",
            )

        # Other resource types
        validator.validate_resource_access(
            "test_agent",
            "memory",
            "heap_1",
            "read",
        )
