# File: src/models/registry.py
"""
Model registry and factory for creating model wrappers.

This module provides a centralized system for:
1. Creating model wrappers from configuration
2. Loading all configured models
3. Managing model instances

The factory pattern allows us to instantiate the correct wrapper type
(MaskedLM, GenerativeHF, or GenerativeAPI) based on model configuration,
providing a uniform interface for the evaluation pipeline.

## Usage

```python
from src.models.registry import create_model, load_all_models

# Create a single model from config
model_cfg = {
    "id": "bert-base-uncased",
    "type": "masked_lm",
    "hf_name": "bert-base-uncased"
}
model = create_model(model_cfg, device="cuda")

# Or load all models from configs/models.yaml
models = load_all_models(device="cuda")
bert = models["bert-base-uncased"]
```
"""

import logging
import os
from typing import Any, Dict, Mapping, Optional, Protocol

from src.utils.config import load_models_config

from .generative_api import GenerativeAPIWrapper
from .generative_hf import GenerativeHFWrapper
from .masked_lm import MaskedLMWrapper

logger = logging.getLogger(__name__)


# ============================================================================
# Base Protocol for Model Wrappers
# ============================================================================


class BaseModelWrapper(Protocol):
    """
    Protocol defining the common interface for all model wrappers.

    This protocol ensures that all wrappers (MaskedLM, GenerativeHF, GenerativeAPI)
    implement a minimal common interface, enabling polymorphic usage.

    Methods:
        to(device: str) -> None: Move model to specified device.
    """

    def to(self, device: str) -> None:
        """Move model to specified device."""
        ...


# ============================================================================
# Model Factory
# ============================================================================


def create_model(
    model_cfg: Mapping[str, Any],
    device: str = "cuda",
    api_client: Optional[Any] = None,
) -> BaseModelWrapper:
    """
    Create a model wrapper instance from configuration.

    This factory function inspects the model configuration and instantiates
    the appropriate wrapper class based on the model type.

    Args:
        model_cfg: Model configuration dictionary with keys:
            - type: "masked_lm", "generative_hf", or "generative_api"
            - hf_name: HuggingFace model ID (for HF models)
            - api_name: API model name (for API models)
        device: Device for local models ("cpu", "cuda", "mps", "auto").
        api_client: Initialized API client (required for API models).

    Returns:
        Instantiated model wrapper.

    Raises:
        ValueError: If model type is unknown or required config is missing.

    Examples:
        >>> # Create a masked LM
        >>> cfg = {"type": "masked_lm", "hf_name": "bert-base-uncased"}
        >>> model = create_model(cfg, device="cuda")
        >>>
        >>> # Create a generative HF model
        >>> cfg = {"type": "generative_hf", "hf_name": "gpt2"}
        >>> model = create_model(cfg, device="cuda")
        >>>
        >>> # Create an API model
        >>> from openai import OpenAI
        >>> client = OpenAI()
        >>> cfg = {"type": "generative_api", "api_name": "gpt-3.5-turbo"}
        >>> model = create_model(cfg, api_client=client)
    """
    model_type = model_cfg.get("type")

    if not model_type:
        raise ValueError(f"Model config missing 'type' field: {model_cfg}")

    # Masked Language Models (BERT, RoBERTa, etc.)
    if model_type == "masked_lm":
        hf_name = model_cfg.get("hf_name")
        if not hf_name:
            raise ValueError(
                f"Masked LM config missing 'hf_name' field: {model_cfg}"
            )

        logger.info(f"Creating MaskedLMWrapper for {hf_name}")
        return MaskedLMWrapper(
            model_name=hf_name,
            device=device,
        )

    # Generative HuggingFace Models (GPT-2, GPT-Neo, etc.)
    elif model_type == "generative_hf":
        hf_name = model_cfg.get("hf_name")
        if not hf_name:
            raise ValueError(
                f"Generative HF config missing 'hf_name' field: {model_cfg}"
            )

        logger.info(f"Creating GenerativeHFWrapper for {hf_name}")
        return GenerativeHFWrapper(
            model_name=hf_name,
            device=device,
        )

    # API-based Generative Models (GPT-3/4, Claude, etc.)
    elif model_type == "generative_api":
        api_name = model_cfg.get("api_name")
        if not api_name:
            raise ValueError(
                f"Generative API config missing 'api_name' field: {model_cfg}"
            )

        if api_client is None:
            raise ValueError(
                f"API model {api_name} requires 'api_client' parameter. "
                f"Example: create_model(cfg, api_client=OpenAI())"
            )

        logger.info(f"Creating GenerativeAPIWrapper for {api_name}")
        return GenerativeAPIWrapper(
            api_name=api_name,
            client=api_client,
        )

    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Supported types: 'masked_lm', 'generative_hf', 'generative_api'"
        )


def load_all_models(
    device: str = "cuda",
    only_types: Optional[list[str]] = None,
    only_ids: Optional[list[str]] = None,
    api_client: Optional[Any] = None,
) -> Dict[str, BaseModelWrapper]:
    """
    Load all models defined in configs/models.yaml.

    This function reads the model configuration file and instantiates
    wrappers for all defined models, returning them in a dictionary
    keyed by model ID.

    Args:
        device: Device for local models ("cpu", "cuda", "mps", "auto").
        only_types: If specified, only load models of these types.
            Example: ["masked_lm", "generative_hf"]
        only_ids: If specified, only load models with these IDs.
            Example: ["bert-base-uncased", "gpt2"]
        api_client: Initialized API client (required if loading API models).

    Returns:
        Dictionary mapping model ID to wrapper instance.

    Raises:
        ValueError: If configuration is invalid or API client is missing for API models.

    Examples:
        >>> # Load all local models
        >>> models = load_all_models(device="cuda", only_types=["masked_lm", "generative_hf"])
        >>> bert = models["bert-base-uncased"]
        >>> gpt2 = models["gpt2"]
        >>>
        >>> # Load specific models
        >>> models = load_all_models(only_ids=["bert-base-uncased", "gpt2"])
        >>>
        >>> # Load API models
        >>> from openai import OpenAI
        >>> client = OpenAI()
        >>> models = load_all_models(
        ...     only_types=["generative_api"],
        ...     api_client=client
        ... )
    """
    # Load configuration
    config = load_models_config()
    models_list = config.get("models", [])

    if not models_list:
        logger.warning("No models found in configs/models.yaml")
        return {}

    # Filter models if requested
    if only_types:
        models_list = [m for m in models_list if m.get("type") in only_types]

    if only_ids:
        models_list = [m for m in models_list if m.get("id") in only_ids]

    logger.info(f"Loading {len(models_list)} models from config")

    # Load each model
    loaded_models = {}

    for model_cfg in models_list:
        model_id = model_cfg.get("id")

        if not model_id:
            logger.warning(f"Skipping model with missing 'id': {model_cfg}")
            continue

        try:
            # Create model wrapper
            model = create_model(
                model_cfg,
                device=device,
                api_client=api_client,
            )

            loaded_models[model_id] = model
            logger.info(f"✓ Loaded {model_id}")

        except Exception as e:
            logger.error(f"✗ Failed to load {model_id}: {e}")
            # Continue loading other models even if one fails

    logger.info(f"Successfully loaded {len(loaded_models)}/{len(models_list)} models")

    return loaded_models


def get_model_config(model_id: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model by ID.

    Args:
        model_id: Model identifier (e.g., "bert-base-uncased").

    Returns:
        Model configuration dictionary.

    Raises:
        ValueError: If model ID is not found in configuration.

    Examples:
        >>> cfg = get_model_config("bert-base-uncased")
        >>> print(cfg["type"])
        'masked_lm'
    """
    config = load_models_config()
    models_list = config.get("models", [])

    for model_cfg in models_list:
        if model_cfg.get("id") == model_id:
            return model_cfg

    raise ValueError(
        f"Model '{model_id}' not found in configs/models.yaml. "
        f"Available models: {[m.get('id') for m in models_list]}"
    )


def list_available_models(model_type: Optional[str] = None) -> list[str]:
    """
    List all available model IDs from configuration.

    Args:
        model_type: Optional filter by type ("masked_lm", "generative_hf", "generative_api").

    Returns:
        List of model IDs.

    Examples:
        >>> # List all models
        >>> all_models = list_available_models()
        >>> print(all_models)
        ['bert-base-uncased', 'gpt2', 'gpt-3.5-turbo', ...]
        >>>
        >>> # List only masked LMs
        >>> masked_lms = list_available_models(model_type="masked_lm")
        >>> print(masked_lms)
        ['bert-base-uncased', 'bert-large-uncased', ...]
    """
    config = load_models_config()
    models_list = config.get("models", [])

    if model_type:
        models_list = [m for m in models_list if m.get("type") == model_type]

    return [m.get("id") for m in models_list if m.get("id")]


def main():
    """Test the model registry."""
    print("=" * 70)
    print("Model Registry Test")
    print("=" * 70)

    # List available models
    print("\nAvailable models:")
    print("-" * 70)

    all_models = list_available_models()
    print(f"Total: {len(all_models)} models")

    print("\nBy type:")
    for model_type in ["masked_lm", "generative_hf", "generative_api"]:
        models = list_available_models(model_type=model_type)
        print(f"  {model_type:20s}: {len(models)} models - {models}")

    # Test get_model_config
    print("\n" + "-" * 70)
    print("Model Configuration Test")
    print("-" * 70)

    try:
        cfg = get_model_config("bert-base-uncased")
        print(f"\nConfiguration for 'bert-base-uncased':")
        for key, value in cfg.items():
            print(f"  {key:20s}: {value}")
    except ValueError as e:
        print(f"Error: {e}")

    # Test create_model with individual models
    print("\n" + "-" * 70)
    print("Creating Individual Models")
    print("-" * 70)

    # Create BERT (masked LM)
    try:
        print("\nCreating bert-base-uncased (masked LM)...")
        cfg = get_model_config("bert-base-uncased")
        bert = create_model(cfg, device="cpu")
        print(f"✓ Created: {bert}")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Create GPT-2 (generative HF)
    try:
        print("\nCreating gpt2 (generative HF)...")
        cfg = get_model_config("gpt2")
        gpt2 = create_model(cfg, device="cpu")
        print(f"✓ Created: {gpt2}")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test API model (without actual client)
    print("\nCreating gpt-3.5-turbo (generative API)...")
    print("  (Skipped - requires API client)")

    # Test load_all_models (only local models)
    print("\n" + "-" * 70)
    print("Loading All Local Models")
    print("-" * 70)

    print("\nNote: This will download models if not cached. May take time...")
    print("Loading only masked_lm and generative_hf types...")

    try:
        models = load_all_models(
            device="cpu",
            only_types=["masked_lm", "generative_hf"],
        )

        print(f"\n✓ Loaded {len(models)} models:")
        for model_id, model_wrapper in models.items():
            print(f"  {model_id:25s}: {type(model_wrapper).__name__}")

    except Exception as e:
        print(f"✗ Failed to load models: {e}")

    print("\n" + "=" * 70)
    print("Registry test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
