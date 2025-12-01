# File: src/models/__init__.py
"""
Model wrappers and registry for bias drift evaluation.

This module provides unified interfaces for working with different types of
language models:

- **Masked LMs**: BERT, RoBERTa (for pseudo-log-likelihood scoring)
- **Generative HF**: GPT-2, GPT-Neo (for generation and probability scoring)
- **Generative API**: GPT-3/4, Claude (for generation via API)

The registry system enables loading models from configuration files and
instantiating the appropriate wrapper automatically.

## Basic Usage

```python
# Import model wrappers
from src.models import MaskedLMWrapper, GenerativeHFWrapper, GenerativeAPIWrapper

# Use specific wrappers directly
bert = MaskedLMWrapper("bert-base-uncased", device="cuda")
gpt2 = GenerativeHFWrapper("gpt2", device="cuda")

# Or use the registry to load from config
from src.models import create_model, load_all_models

# Load a single model
cfg = {"type": "masked_lm", "hf_name": "bert-base-uncased"}
model = create_model(cfg, device="cuda")

# Load all models from configs/models.yaml
models = load_all_models(device="cuda")
```

## Model Registry

The registry provides a factory pattern for creating models from configuration:

```python
from src.models import create_model, load_all_models, list_available_models

# List available models
available = list_available_models()
print(available)  # ['bert-base-uncased', 'gpt2', 'gpt-3.5-turbo', ...]

# Load specific models
models = load_all_models(only_ids=["bert-base-uncased", "gpt2"])

# Load only certain types
local_models = load_all_models(only_types=["masked_lm", "generative_hf"])
```
"""

from .generative_api import GenerativeAPIWrapper
from .generative_hf import GenerativeHFWrapper
from .masked_lm import MaskedLMWrapper
from .registry import (
    BaseModelWrapper,
    create_model,
    get_model_config,
    list_available_models,
    load_all_models,
)

__all__ = [
    # Model wrappers
    "MaskedLMWrapper",
    "GenerativeHFWrapper",
    "GenerativeAPIWrapper",
    # Registry functions
    "BaseModelWrapper",
    "create_model",
    "load_all_models",
    "get_model_config",
    "list_available_models",
]
