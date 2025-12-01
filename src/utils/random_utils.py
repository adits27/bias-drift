# File: src/utils/random_utils.py
"""
Reproducibility utilities for setting random seeds.

This module provides functions to ensure reproducibility across different
random number generators (Python, NumPy, PyTorch) used in the project.
"""

import random
import warnings
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    This function configures all major random number generators to use the same
    seed, enabling reproducible experiments. For PyTorch, it also configures
    CUDA operations for deterministic behavior.

    Args:
        seed: The random seed to use across all libraries.
        deterministic_cudnn: If True, sets cuDNN to deterministic mode.
            This may reduce performance but ensures reproducibility.
            Default is True.

    Notes:
        Performance Trade-offs:
        - Setting `torch.backends.cudnn.deterministic = True` can slow down
          training, especially for models with varying input sizes.
        - Setting `torch.backends.cudnn.benchmark = False` disables the
          autotuner that finds the best algorithms for your hardware.
        - For reproducibility in research, these trade-offs are usually acceptable.
        - For production/performance-critical applications, consider disabling
          deterministic mode after initial validation.

        Limitations:
        - Some PyTorch operations do not have deterministic implementations,
          particularly for GPU operations. Check PyTorch documentation for details.
        - Multi-threaded CPU operations may still have non-deterministic behavior.

    Examples:
        >>> # Standard usage for research experiments
        >>> set_global_seed(42)
        >>>
        >>> # For maximum performance (non-deterministic)
        >>> set_global_seed(42, deterministic_cudnn=False)

    References:
        PyTorch Reproducibility:
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed for CPU
    torch.manual_seed(seed)

    # Set PyTorch random seed for all GPUs (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Configure cuDNN for deterministic behavior
    if deterministic_cudnn:
        # Ensure that cuDNN uses deterministic algorithms
        # Note: This may reduce performance
        torch.backends.cudnn.deterministic = True

        # Disable cuDNN autotuner (which can introduce non-determinism)
        # The autotuner benchmarks multiple algorithms and picks the fastest
        torch.backends.cudnn.benchmark = False

        # Set environment variable for PyTorch < 1.8 compatibility
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # For older PyTorch versions that don't have this function
            warnings.warn(
                "torch.use_deterministic_algorithms() not available. "
                "Using cuDNN deterministic settings only."
            )
    else:
        # Enable cuDNN autotuner for better performance
        # Note: This may introduce non-deterministic behavior
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_rng_state() -> dict:
    """
    Get the current state of all random number generators.

    Returns:
        Dictionary containing the RNG states for Python, NumPy, and PyTorch.

    Examples:
        >>> # Save RNG state before some operation
        >>> rng_state = get_rng_state()
        >>> # ... do some random operations ...
        >>> # Restore RNG state
        >>> set_rng_state(rng_state)
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def set_rng_state(state: dict) -> None:
    """
    Restore the state of all random number generators.

    Args:
        state: Dictionary containing RNG states (from get_rng_state()).

    Examples:
        >>> # Save RNG state before some operation
        >>> rng_state = get_rng_state()
        >>> # ... do some random operations ...
        >>> # Restore RNG state
        >>> set_rng_state(rng_state)
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def seed_worker(worker_id: int) -> None:
    """
    Seed a PyTorch DataLoader worker for reproducible data loading.

    This function should be passed to the `worker_init_fn` parameter
    of torch.utils.data.DataLoader.

    Args:
        worker_id: Worker ID (automatically provided by DataLoader).

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     worker_init_fn=seed_worker,
        ...     generator=torch.Generator().manual_seed(42)
        ... )
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
