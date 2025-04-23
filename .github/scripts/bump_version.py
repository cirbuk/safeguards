#!/usr/bin/env python3
"""
Script to bump the version in pyproject.toml
Usage: python bump_version.py [patch|minor|major]
"""

import sys

import toml


def bump_version(version: str, bump_type: str) -> str:
    """Bump the version number based on the bump type."""
    major, minor, patch = map(int, version.split("."))

    if bump_type == "patch":
        patch += 1
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        msg = f"Invalid bump type: {bump_type}"
        raise ValueError(msg)

    return f"{major}.{minor}.{patch}"


def main():
    """Main function."""
    if len(sys.argv) != 2 or sys.argv[1] not in ["patch", "minor", "major"]:
        print("Usage: python bump_version.py [patch|minor|major]")
        sys.exit(1)

    bump_type = sys.argv[1]

    # Read pyproject.toml
    try:
        pyproject = toml.load("pyproject.toml")
    except Exception as e:
        print(f"Error reading pyproject.toml: {e}")
        sys.exit(1)

    # Get current version
    try:
        current_version = pyproject["project"]["version"]
    except KeyError:
        print("Error: version field not found in pyproject.toml")
        sys.exit(1)

    # Bump version
    new_version = bump_version(current_version, bump_type)
    print(f"Bumping {bump_type} version: {current_version} -> {new_version}")

    # Update version in pyproject.toml
    pyproject["project"]["version"] = new_version

    # Write updated pyproject.toml
    try:
        with open("pyproject.toml", "w") as f:
            toml.dump(pyproject, f)
    except Exception as e:
        print(f"Error writing pyproject.toml: {e}")
        sys.exit(1)

    print(f"Updated pyproject.toml with new version: {new_version}")


if __name__ == "__main__":
    main()
