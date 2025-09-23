#!/usr/bin/env python3
"""Script to update version in Python files for semantic-release."""

import re
import sys
from pathlib import Path


def update_version_in_file(file_path: Path, new_version: str) -> bool:
    """Update version in a Python file.

    Args:
        file_path: Path to the file to update
        new_version: New version string (e.g., "1.2.3")

    Returns:
        True if version was updated, False otherwise
    """
    if not file_path.exists():
        return False

    content = file_path.read_text()

    # Pattern to match version assignments
    patterns = [
        r'__version__\s*=\s*["\']([^"\']+)["\']',
        r'version\s*=\s*["\']([^"\']+)["\']',
    ]

    updated = False
    for pattern in patterns:
        new_content = re.sub(
            pattern,
            f'__version__ = "{new_version}"'
            if "__version__" in pattern
            else f'version = "{new_version}"',
            content,
        )
        if new_content != content:
            content = new_content
            updated = True

    if updated:
        file_path.write_text(content)
        print(f"Updated version in {file_path} to {new_version}")

    return updated


def update_pyproject_toml(file_path: Path, new_version: str) -> bool:
    """Update version in pyproject.toml.

    Args:
        file_path: Path to pyproject.toml
        new_version: New version string

    Returns:
        True if version was updated, False otherwise
    """
    if not file_path.exists():
        return False

    content = file_path.read_text()

    # Update version in pyproject.toml
    new_content = re.sub(
        r'version\s*=\s*["\']([^"\']+)["\']', f'version = "{new_version}"', content
    )

    if new_content != content:
        file_path.write_text(new_content)
        print(f"Updated version in {file_path} to {new_version}")
        return True

    return False


def main() -> None:
    """Update version in all relevant files."""
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <new_version>")
        sys.exit(1)

    new_version = sys.argv[1]

    # Files to update
    files_to_update = [
        Path("gaussian_fit_parameter_tsa/__init__.py"),
        Path("pyproject.toml"),
    ]

    updated_files = []

    for file_path in files_to_update:
        if file_path.name == "pyproject.toml":
            if update_pyproject_toml(file_path, new_version):
                updated_files.append(str(file_path))
        else:
            if update_version_in_file(file_path, new_version):
                updated_files.append(str(file_path))

    if updated_files:
        print(f"Successfully updated version to {new_version} in:")
        for file in updated_files:
            print(f"  - {file}")
    else:
        print("No files were updated")


if __name__ == "__main__":
    main()
