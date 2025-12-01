# File: src/utils/paths.py
"""
Path utilities for the bias-drift project.

This module provides functions to compute important project paths relative to the
project root directory, ensuring consistency across different execution contexts.
"""

from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """
    Get the absolute path to the project root directory.

    The project root is defined as the directory containing the 'src' folder.
    This function traverses up from the current file location to find it.

    Returns:
        Path: Absolute path to the project root directory.

    Raises:
        FileNotFoundError: If the project root cannot be determined.
    """
    # Start from the current file's directory
    current_path = Path(__file__).resolve()

    # Traverse up to find the project root (directory containing 'src')
    # Current file is in src/utils/paths.py, so go up 2 levels
    project_root = current_path.parent.parent.parent

    # Verify that this is actually the project root by checking for 'src' directory
    if not (project_root / "src").exists():
        raise FileNotFoundError(
            f"Could not determine project root. Expected 'src' directory in {project_root}"
        )

    return project_root


def get_configs_dir() -> Path:
    """
    Get the absolute path to the configs directory.

    Returns:
        Path: Absolute path to the configs directory.
    """
    return get_project_root() / "configs"


def get_data_dir(subdir: Union[str, None] = None) -> Path:
    """
    Get the absolute path to the data directory or a specific subdirectory.

    Args:
        subdir: Optional subdirectory name (e.g., 'crows_pairs', 'winobias').

    Returns:
        Path: Absolute path to the data directory or specified subdirectory.

    Examples:
        >>> get_data_dir()
        PosixPath('/path/to/bias-drift/data')
        >>> get_data_dir('crows_pairs')
        PosixPath('/path/to/bias-drift/data/crows_pairs')
    """
    data_dir = get_project_root() / "data"

    if subdir is not None:
        return data_dir / subdir

    return data_dir


def get_results_dir(subdir: Union[str, None] = None) -> Path:
    """
    Get the absolute path to the results directory or a specific subdirectory.

    Args:
        subdir: Optional subdirectory name (e.g., 'raw', 'aggregated', 'figures').

    Returns:
        Path: Absolute path to the results directory or specified subdirectory.

    Examples:
        >>> get_results_dir()
        PosixPath('/path/to/bias-drift/results')
        >>> get_results_dir('raw')
        PosixPath('/path/to/bias-drift/results/raw')
    """
    results_dir = get_project_root() / "results"

    if subdir is not None:
        return results_dir / subdir

    return results_dir


def get_scripts_dir() -> Path:
    """
    Get the absolute path to the scripts directory.

    Returns:
        Path: Absolute path to the scripts directory.
    """
    return get_project_root() / "scripts"


def get_notebooks_dir() -> Path:
    """
    Get the absolute path to the notebooks directory.

    Returns:
        Path: Absolute path to the notebooks directory.
    """
    return get_project_root() / "notebooks"


def ensure_dir_exists(directory: Path) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory: Path to the directory to create.

    Returns:
        Path: The directory path (for chaining).

    Examples:
        >>> ensure_dir_exists(get_results_dir('custom_output'))
        PosixPath('/path/to/bias-drift/results/custom_output')
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory
