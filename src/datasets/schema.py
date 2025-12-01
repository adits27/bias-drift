# File: src/datasets/schema.py
"""
Unified schema for bias benchmark examples.

This module defines a common data structure (`BiasExample`) that represents
examples from different bias benchmarks in a consistent format. This allows
downstream code (scoring, analysis, drift detection) to handle different
benchmarks uniformly.

## Design Philosophy

Different bias benchmarks have different structures:

1. **CrowS-Pairs (Pairwise Comparison)**:
   - Each example contains TWO sentences: one more stereotypical, one less.
   - The task is to measure which sentence a model prefers.
   - Maps to BiasExample as:
     * `stereotype_text`: sent_more (more stereotypical sentence)
     * `anti_stereotype_text`: sent_less (less stereotypical sentence)
     * `sentence`, `answer_options`, `correct_option`: all None
     * `type`: "pair"

2. **WinoBias (Coreference Resolution)**:
   - Each example contains a sentence with a pronoun and multiple candidate referents.
   - The task is to identify the correct referent (measuring gender bias).
   - Maps to BiasExample as:
     * `sentence`: the full sentence with pronoun
     * `answer_options`: list of candidate referents [answer_a, answer_b]
     * `correct_option`: the correct referent
     * `stereotype_text`, `anti_stereotype_text`: None (or optionally populated
       if the raw data distinguishes pro- vs anti-stereotypical cases)
     * `type`: "coref"

## Usage Example

```python
from src.datasets.schema import BiasExample
from src.datasets import load_crows_pairs, load_winobias

# Load different benchmarks into unified format
crows_examples = load_crows_pairs()
wino_examples = load_winobias()

# Both return List[BiasExample], so they can be processed uniformly
all_examples = crows_examples + wino_examples

# Downstream code can handle both types uniformly
for example in all_examples:
    if example.type == "pair":
        # Process pairwise comparison
        score = compare_sentences(example.stereotype_text, example.anti_stereotype_text)
    elif example.type == "coref":
        # Process coreference resolution
        score = evaluate_coref(example.sentence, example.answer_options)
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BiasExample:
    """
    Unified representation of a bias benchmark example.

    This dataclass provides a common schema for different bias benchmarks,
    enabling uniform processing and analysis across diverse evaluation tasks.

    Attributes:
        id: Unique identifier within the benchmark (e.g., "crows_123", "wino_456").
        benchmark: Name of the source benchmark (e.g., "crows_pairs", "winobias").
        bias_axis: The dimension of bias being evaluated (e.g., "gender", "race",
            "religion", "socioeconomic"). Also called "bias type" or "category".
        type: The task type (e.g., "pair" for pairwise comparison, "coref" for
            coreference resolution).

        # Fields for pairwise comparison tasks (CrowS-Pairs)
        stereotype_text: Sentence exhibiting stereotypical bias (CrowS-Pairs: sent_more).
            None for non-pairwise tasks.
        anti_stereotype_text: Sentence exhibiting reduced/opposite bias (CrowS-Pairs: sent_less).
            None for non-pairwise tasks.

        # Fields for coreference resolution tasks (WinoBias)
        sentence: Full sentence containing the task (WinoBias: sentence with pronoun).
            None for pairwise tasks.
        answer_options: List of candidate answers (WinoBias: [answer_a, answer_b]).
            None for pairwise tasks.
        correct_option: The correct/gold answer (WinoBias: correct referent).
            None for pairwise tasks.

        # Metadata
        meta: Dictionary containing additional benchmark-specific fields not
            captured in the main schema (e.g., annotations, difficulty, profession,
            subtype). Use this for preserving raw data that might be useful for
            fine-grained analysis.

    Examples:
        >>> # CrowS-Pairs example
        >>> crows_ex = BiasExample(
        ...     id="crows_001",
        ...     benchmark="crows_pairs",
        ...     bias_axis="gender",
        ...     type="pair",
        ...     stereotype_text="The doctor asked the nurse to help him.",
        ...     anti_stereotype_text="The doctor asked the nurse to help her.",
        ...     sentence=None,
        ...     answer_options=None,
        ...     correct_option=None,
        ...     meta={"stereo_antistereo": "stereo"}
        ... )
        >>>
        >>> # WinoBias example
        >>> wino_ex = BiasExample(
        ...     id="wino_001",
        ...     benchmark="winobias",
        ...     bias_axis="gender",
        ...     type="coref",
        ...     stereotype_text=None,
        ...     anti_stereotype_text=None,
        ...     sentence="The developer argued with the designer because he did not like the design.",
        ...     answer_options=["developer", "designer"],
        ...     correct_option="developer",
        ...     meta={"subtype": "pro_stereotype", "profession": "developer"}
        ... )
    """

    # Core identifiers
    id: str
    benchmark: str
    bias_axis: str
    type: str

    # Pairwise comparison fields (CrowS-Pairs)
    stereotype_text: Optional[str] = None
    anti_stereotype_text: Optional[str] = None

    # Coreference resolution fields (WinoBias)
    sentence: Optional[str] = None
    answer_options: Optional[list[str]] = None
    correct_option: Optional[str] = None

    # Additional metadata
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate that the example has consistent fields for its type."""
        if self.type == "pair":
            # Pairwise tasks should have stereotype/anti-stereotype texts
            if self.stereotype_text is None or self.anti_stereotype_text is None:
                raise ValueError(
                    f"BiasExample of type 'pair' must have both stereotype_text "
                    f"and anti_stereotype_text (id: {self.id})"
                )
        elif self.type == "coref":
            # Coreference tasks should have sentence and options
            if self.sentence is None or self.answer_options is None:
                raise ValueError(
                    f"BiasExample of type 'coref' must have both sentence "
                    f"and answer_options (id: {self.id})"
                )

    def __repr__(self) -> str:
        """Provide a readable representation of the example."""
        if self.type == "pair":
            return (
                f"BiasExample(id='{self.id}', benchmark='{self.benchmark}', "
                f"bias_axis='{self.bias_axis}', type='{self.type}', "
                f"stereotype='{self.stereotype_text[:50]}...', "
                f"anti_stereotype='{self.anti_stereotype_text[:50]}...')"
            )
        elif self.type == "coref":
            return (
                f"BiasExample(id='{self.id}', benchmark='{self.benchmark}', "
                f"bias_axis='{self.bias_axis}', type='{self.type}', "
                f"sentence='{self.sentence[:50] if self.sentence else None}...', "
                f"options={self.answer_options}, correct='{self.correct_option}')"
            )
        else:
            return (
                f"BiasExample(id='{self.id}', benchmark='{self.benchmark}', "
                f"bias_axis='{self.bias_axis}', type='{self.type}')"
            )
