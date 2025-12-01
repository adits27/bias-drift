# File: src/datasets/__init__.py
"""
Dataset loading and management for bias benchmarks.

This module provides unified access to different bias benchmark datasets,
converting them all into a common BiasExample schema for downstream processing.

## Supported Benchmarks

- **CrowS-Pairs**: Pairwise stereotype comparison across 9 bias dimensions
- **WinoBias**: Gender bias in coreference resolution

## Usage

```python
# Load specific benchmarks
from src.datasets import load_crows_pairs, load_winobias

crows_examples = load_crows_pairs()
wino_examples = load_winobias()

# Or use the unified loader
from src.datasets import load_benchmark

examples = load_benchmark("crows_pairs")

# Access the common schema
from src.datasets import BiasExample

for example in examples:
    print(f"ID: {example.id}, Bias: {example.bias_axis}")
```
"""

from .crows_pairs import load_crows_pairs
from .schema import BiasExample
from .winobias import load_winobias

__all__ = [
    "BiasExample",
    "load_crows_pairs",
    "load_winobias",
    "load_benchmark",
]


def load_benchmark(name: str, **kwargs) -> list[BiasExample]:
    """
    Load a bias benchmark dataset by name.

    This is a convenience function that dispatches to the appropriate loader
    based on the benchmark name. It provides a unified interface for loading
    different benchmarks.

    Args:
        name: Name of the benchmark to load. Supported values:
            - "crows_pairs" or "crows-pairs"
            - "winobias" or "wino-bias"
        **kwargs: Additional keyword arguments to pass to the specific loader.
            Common kwargs:
            - data_path (Path): Custom path to data file
            - verbose (bool): Whether to log loading statistics

    Returns:
        List of BiasExample objects from the specified benchmark.

    Raises:
        ValueError: If the benchmark name is not recognized.
        FileNotFoundError: If the data file for the benchmark is not found.

    Examples:
        >>> # Load CrowS-Pairs
        >>> examples = load_benchmark("crows_pairs")
        >>> print(f"Loaded {len(examples)} examples")
        >>>
        >>> # Load WinoBias with custom path
        >>> from pathlib import Path
        >>> examples = load_benchmark(
        ...     "winobias",
        ...     data_path=Path("custom/winobias.tsv")
        ... )
        >>>
        >>> # Load with verbose output disabled
        >>> examples = load_benchmark("crows_pairs", verbose=False)
    """
    # Normalize benchmark name
    name_normalized = name.lower().replace("-", "_").replace(" ", "_")

    # Dispatch to appropriate loader
    if name_normalized in ["crows_pairs", "crowspairs"]:
        return load_crows_pairs(**kwargs)
    elif name_normalized in ["winobias", "wino_bias"]:
        return load_winobias(**kwargs)
    else:
        raise ValueError(
            f"Unknown benchmark: '{name}'. "
            f"Supported benchmarks: 'crows_pairs', 'winobias'"
        )


def get_available_benchmarks() -> list[str]:
    """
    Get a list of available benchmark names.

    Returns:
        List of benchmark names that can be loaded.

    Examples:
        >>> benchmarks = get_available_benchmarks()
        >>> print(benchmarks)
        ['crows_pairs', 'winobias']
    """
    return ["crows_pairs", "winobias"]


def get_benchmark_info(name: str) -> dict:
    """
    Get metadata about a benchmark.

    Args:
        name: Name of the benchmark.

    Returns:
        Dictionary containing benchmark metadata (name, type, bias axes, etc.).

    Raises:
        ValueError: If the benchmark name is not recognized.

    Examples:
        >>> info = get_benchmark_info("crows_pairs")
        >>> print(info["type"])
        'pair'
        >>> print(info["bias_axes"])
        ['race-color', 'socioeconomic', 'gender', ...]
    """
    from src.utils.config import load_benchmarks_config

    # Normalize benchmark name
    name_normalized = name.lower().replace("-", "_")

    # Load config
    config = load_benchmarks_config()

    if name_normalized not in config:
        raise ValueError(
            f"Unknown benchmark: '{name}'. "
            f"Available: {list(config.keys())}"
        )

    return config[name_normalized]
