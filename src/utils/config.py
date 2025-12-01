# File: src/utils/config.py
"""
Configuration utilities for loading YAML config files.

This module provides functions to load and parse YAML configuration files
from the configs/ directory, with robust path resolution relative to the
project root.
"""

from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .paths import get_configs_dir, get_project_root


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        path: Path to the YAML file. Can be absolute or relative to project root.

    Returns:
        Dictionary containing the parsed YAML contents.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML file is malformed.

    Examples:
        >>> config = load_yaml('configs/models.yaml')
        >>> config = load_yaml('/absolute/path/to/config.yaml')
    """
    # Convert to Path object
    yaml_path = Path(path)

    # If path is not absolute, resolve it relative to project root
    if not yaml_path.is_absolute():
        yaml_path = get_project_root() / yaml_path

    # Check if file exists
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    # Load and parse YAML
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {yaml_path}: {e}")

    # Handle empty YAML files
    if config is None:
        config = {}

    return config


def load_models_config() -> Dict[str, Any]:
    """
    Load the models configuration file.

    Loads configs/models.yaml which contains model definitions, versions,
    and metadata for evaluation.

    Returns:
        Dictionary containing model configurations.

    Raises:
        FileNotFoundError: If configs/models.yaml does not exist.
        yaml.YAMLError: If the YAML file is malformed.

    Examples:
        >>> models = load_models_config()
        >>> print(models['gpt2']['versions'])
    """
    models_config_path = get_configs_dir() / "models.yaml"
    return load_yaml(models_config_path)


def load_benchmarks_config() -> Dict[str, Any]:
    """
    Load the benchmarks configuration file.

    Loads configs/benchmarks.yaml which contains benchmark settings,
    evaluation parameters, and dataset configurations.

    Returns:
        Dictionary containing benchmark configurations.

    Raises:
        FileNotFoundError: If configs/benchmarks.yaml does not exist.
        yaml.YAMLError: If the YAML file is malformed.

    Examples:
        >>> benchmarks = load_benchmarks_config()
        >>> print(benchmarks['crows_pairs']['categories'])
    """
    benchmarks_config_path = get_configs_dir() / "benchmarks.yaml"
    return load_yaml(benchmarks_config_path)


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save a dictionary to a YAML file.

    Args:
        data: Dictionary to save.
        path: Path to the output YAML file. Can be absolute or relative to project root.

    Examples:
        >>> config = {'model': 'gpt2', 'batch_size': 32}
        >>> save_yaml(config, 'configs/custom_config.yaml')
    """
    # Convert to Path object
    yaml_path = Path(path)

    # If path is not absolute, resolve it relative to project root
    if not yaml_path.is_absolute():
        yaml_path = get_project_root() / yaml_path

    # Ensure parent directory exists
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # Save YAML
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.

    Later dictionaries override values from earlier ones.

    Args:
        *configs: Variable number of configuration dictionaries to merge.

    Returns:
        Merged configuration dictionary.

    Examples:
        >>> default_config = {'batch_size': 32, 'learning_rate': 0.001}
        >>> user_config = {'batch_size': 64}
        >>> merged = merge_configs(default_config, user_config)
        >>> print(merged['batch_size'])
        64
    """
    result = {}
    for config in configs:
        result.update(config)
    return result
