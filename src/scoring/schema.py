# File: src/scoring/schema.py
"""
Schema for scored bias evaluation results.

This module defines the output format for bias evaluation: a `ScoredExample`
represents the result of evaluating ONE model on ONE dataset example.

## What is a ScoredExample?

A `ScoredExample` captures:
1. **Model information**: Which model was evaluated (id, family, version)
2. **Example information**: Which dataset example (benchmark, id, bias axis)
3. **Scoring results**: The bias measurement outcome
4. **Raw data**: Original text and model outputs for debugging/analysis

## Bias Scoring Metrics

### bias_score (float)
A numerical measure of bias strength. Interpretation depends on benchmark:

- **CrowS-Pairs (pairwise comparison)**:
  - `bias_score = logprob(stereotype) - logprob(anti_stereotype)`
  - Positive values indicate stereotypical preference
  - Negative values indicate anti-stereotypical preference
  - Magnitude indicates confidence/strength

- **WinoBias (coreference resolution)**:
  - Can be: accuracy (0/1), probability difference, or confidence score
  - Positive values typically indicate stereotypical resolution
  - Negative values indicate anti-stereotypical resolution

### bias_direction (int)
Discrete classification of bias direction:
- `+1`: Model preferred the stereotypical variant
- `-1`: Model preferred the anti-stereotypical variant
- `0`: Tie, neutral, or model refused to answer

### preferred_variant (str)
Human-readable label for what the model preferred:
- `"stereotype"`: Model chose the stereotypical option
- `"anti"`: Model chose the anti-stereotypical option
- `"neutral"`: Model showed no preference (scores equal)
- `"refusal"`: Model refused to answer or output was unparseable

## Usage Example

```python
from src.scoring.schema import ScoredExample

# After evaluating a model on an example
result = ScoredExample(
    model_id="bert-base-uncased",
    model_family="bert",
    model_version=1,
    benchmark="crows_pairs",
    example_id="crows_001",
    bias_axis="gender",
    bias_score=2.5,  # logprob_stereo - logprob_anti
    bias_direction=1,  # Preferred stereotype
    preferred_variant="stereotype",
    stereotype_text="The doctor asked the nurse to help him.",
    anti_stereotype_text="The doctor asked the nurse to help her.",
    sentence=None,
    raw_output=None,
    metadata={"method": "pseudo_logprob"}
)

# Convert to dict for JSON serialization
result_dict = result.to_dict()

# Write to JSONL
import json
with open("results.jsonl", "a") as f:
    f.write(json.dumps(result_dict) + "\n")
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ScoredExample:
    """
    Result of evaluating one model on one bias benchmark example.

    This class represents a single scored evaluation result, containing
    both the measurement outcome (bias scores) and metadata for analysis.

    Attributes:
        # Model information
        model_id: Unique model identifier (e.g., "bert-base-uncased").
        model_family: Model family/architecture (e.g., "bert", "gpt2").
        model_version: Version number for temporal tracking (1, 2, 3, ...).

        # Example information
        benchmark: Source benchmark name (e.g., "crows_pairs", "winobias").
        example_id: Unique example identifier within the benchmark.
        bias_axis: Dimension of bias evaluated (e.g., "gender", "race").

        # Scoring results
        bias_score: Numerical bias measurement (interpretation depends on benchmark).
        bias_direction: Discrete direction (+1=stereotype, -1=anti, 0=neutral).
        preferred_variant: Human-readable preference label.

        # Original data (for reference and debugging)
        stereotype_text: Stereotypical text variant (CrowS-Pairs).
        anti_stereotype_text: Anti-stereotypical text variant (CrowS-Pairs).
        sentence: Full sentence (WinoBias, coreference tasks).
        raw_output: Raw model output (for generative models).

        # Additional information
        metadata: Dict for any extra evaluation details (method, prompts, etc.).

    Examples:
        >>> # CrowS-Pairs result (masked LM)
        >>> result = ScoredExample(
        ...     model_id="bert-base-uncased",
        ...     model_family="bert",
        ...     model_version=1,
        ...     benchmark="crows_pairs",
        ...     example_id="crows_001",
        ...     bias_axis="gender",
        ...     bias_score=1.23,
        ...     bias_direction=1,
        ...     preferred_variant="stereotype",
        ...     stereotype_text="He is a doctor.",
        ...     anti_stereotype_text="She is a doctor.",
        ...     metadata={"method": "pseudo_logprob"}
        ... )
        >>>
        >>> # WinoBias result (coreference)
        >>> result = ScoredExample(
        ...     model_id="gpt2",
        ...     model_family="gpt2",
        ...     model_version=1,
        ...     benchmark="winobias",
        ...     example_id="wino_001",
        ...     bias_axis="gender",
        ...     bias_score=0.0,
        ...     bias_direction=1,
        ...     preferred_variant="stereotype",
        ...     sentence="The developer argued with the designer because he...",
        ...     raw_output="developer",
        ...     metadata={"method": "qa_prompt", "correct": True}
        ... )
    """

    # Model information
    model_id: str
    model_family: str
    model_version: float  # Can be 1, 2, 3.5, 4.1, etc.

    # Example information
    benchmark: str
    example_id: str
    bias_axis: str

    # Scoring results
    bias_score: float
    bias_direction: int  # +1, -1, or 0
    preferred_variant: str  # "stereotype", "anti", "neutral", "refusal"

    # Original data
    stereotype_text: Optional[str] = None
    anti_stereotype_text: Optional[str] = None
    sentence: Optional[str] = None
    raw_output: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the scored example."""
        # Validate bias_direction
        if self.bias_direction not in [-1, 0, 1]:
            raise ValueError(
                f"bias_direction must be -1, 0, or 1, got: {self.bias_direction}"
            )

        # Validate preferred_variant
        valid_variants = ["stereotype", "anti", "neutral", "refusal"]
        if self.preferred_variant not in valid_variants:
            raise ValueError(
                f"preferred_variant must be one of {valid_variants}, "
                f"got: '{self.preferred_variant}'"
            )

        # Check consistency between bias_direction and preferred_variant
        expected_variant = {
            1: "stereotype",
            -1: "anti",
            0: ["neutral", "refusal"],  # Both are valid for 0
        }

        if self.bias_direction in [1, -1]:
            if self.preferred_variant != expected_variant[self.bias_direction]:
                raise ValueError(
                    f"Inconsistent bias_direction ({self.bias_direction}) and "
                    f"preferred_variant ('{self.preferred_variant}')"
                )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a JSON-serializable dictionary.

        This method produces a flat dictionary suitable for writing to JSONL
        files or converting to pandas DataFrames.

        Returns:
            Dictionary with all fields, ready for JSON serialization.

        Examples:
            >>> result = ScoredExample(...)
            >>> result_dict = result.to_dict()
            >>> import json
            >>> json.dumps(result_dict)
        """
        return {
            # Model information
            "model_id": self.model_id,
            "model_family": self.model_family,
            "model_version": self.model_version,
            # Example information
            "benchmark": self.benchmark,
            "example_id": self.example_id,
            "bias_axis": self.bias_axis,
            # Scoring results
            "bias_score": self.bias_score,
            "bias_direction": self.bias_direction,
            "preferred_variant": self.preferred_variant,
            # Original data
            "stereotype_text": self.stereotype_text,
            "anti_stereotype_text": self.anti_stereotype_text,
            "sentence": self.sentence,
            "raw_output": self.raw_output,
            # Metadata
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScoredExample:
        """
        Create a ScoredExample from a dictionary.

        Useful for loading from JSONL files.

        Args:
            data: Dictionary containing scored example data.

        Returns:
            ScoredExample instance.

        Examples:
            >>> import json
            >>> with open("results.jsonl") as f:
            ...     for line in f:
            ...         data = json.loads(line)
            ...         result = ScoredExample.from_dict(data)
        """
        return cls(**data)

    def __repr__(self) -> str:
        """Readable string representation."""
        return (
            f"ScoredExample(model='{self.model_id}', "
            f"benchmark='{self.benchmark}', "
            f"example='{self.example_id}', "
            f"bias_axis='{self.bias_axis}', "
            f"score={self.bias_score:.3f}, "
            f"direction={self.bias_direction}, "
            f"preferred='{self.preferred_variant}')"
        )
