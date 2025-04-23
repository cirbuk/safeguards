# Safeguards Tests

This directory contains the test suite for the Safeguards framework. The tests are organized by component to ensure comprehensive coverage of the codebase.

## Test Structure

- `unit/`: Unit tests for individual components
  - `core/`: Tests for core components
  - `api/`: Tests for API interfaces

- `integration/`: Integration tests that verify interactions between components

- `budget/`: Tests for budget management components
  - `coordinator/`: Tests for budget coordination

- `security/`: Tests for security and authentication components

- `monitoring/`: Tests for monitoring and metrics components

- `notifications/`: Tests for notification system

- `guardrails/`: Tests for safety guardrails

- `resource/`: Tests for resource management

- `swarm/`: Tests for swarm management

- `utils/`: Test utilities and helpers

## Running Tests

To run the entire test suite:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/path/to/test_file.py
```

To run tests with code coverage:

```bash
pytest --cov=safeguards
```

## Adding New Tests

When adding new tests:

1. Place them in the appropriate directory based on the component being tested
2. Follow the naming convention: `test_*.py` for test files, `test_*` for test functions
3. Use proper test fixtures and mocks to isolate unit tests
4. Include docstrings that explain what each test is verifying
5. Ensure each test has clear assertions that validate expected behavior
