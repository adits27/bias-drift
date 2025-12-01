# File: src/scoring/__init__.py
"""
Bias scoring and evaluation modules.

This package provides the core evaluation infrastructure for measuring bias
in language models using different benchmarks.

## Components

- **Schema**: `ScoredExample` dataclass for storing evaluation results
- **Scorers**: Benchmark-specific scoring implementations
  - `score_crows_example`: CrowS-Pairs pairwise comparison scoring
  - `score_winobias_example`: WinoBias coreference resolution scoring
- **Evaluator**: Generic evaluation orchestrator for running benchmarks

## Usage

```python
# Import scoring functions
from src.scoring import (
    ScoredExample,
    score_crows_example,
    score_winobias_example,
    evaluate_model_on_benchmark,
)

# Load model and examples
from src.models import MaskedLMWrapper
from src.datasets import load_crows_pairs

model = MaskedLMWrapper("bert-base-uncased", device="cuda")
model_cfg = {"id": "bert-base-uncased", "family": "bert", "version": 1, "type": "masked_lm"}
examples = load_crows_pairs()

# Evaluate
results = evaluate_model_on_benchmark(
    model_wrapper=model,
    model_cfg=model_cfg,
    benchmark_name="crows_pairs",
    examples=examples
)

# Process results
for result in results:
    if result.bias_direction == 1:
        print(f"Example {result.example_id}: stereotypical preference")
```

## Output Format

All scoring functions return `ScoredExample` objects with:
- Model metadata (id, family, version)
- Example metadata (benchmark, id, bias_axis)
- Bias measurements (score, direction, preferred_variant)
- Original text and raw outputs

Convert to dict for JSON serialization:
```python
result_dict = result.to_dict()
import json
json.dumps(result_dict)
```
"""

from .crows_pairs import score_crows_example
from .evaluator import evaluate_model_on_benchmark, evaluate_multiple_models
from .schema import ScoredExample
from .winobias import score_winobias_example

__all__ = [
    # Schema
    "ScoredExample",
    # Scorers
    "score_crows_example",
    "score_winobias_example",
    # Evaluator
    "evaluate_model_on_benchmark",
    "evaluate_multiple_models",
]
