# File: src/scoring/evaluator.py
"""
Generic evaluator for running bias benchmarks on models.

This module provides a unified interface for evaluating models on different
bias benchmarks, automatically routing to the appropriate scorer based on
benchmark type.

## Usage

```python
from src.models import MaskedLMWrapper
from src.datasets import load_benchmark
from src.scoring.evaluator import evaluate_model_on_benchmark

# Load model
model = MaskedLMWrapper("bert-base-uncased", device="cuda")
model_cfg = {
    "id": "bert-base-uncased",
    "family": "bert",
    "version": 1,
    "type": "masked_lm"
}

# Load benchmark examples
examples = load_benchmark("crows_pairs")

# Evaluate
results = evaluate_model_on_benchmark(
    model_wrapper=model,
    model_cfg=model_cfg,
    benchmark_name="crows_pairs",
    examples=examples
)

# Results is a list of ScoredExample objects
for result in results:
    print(result.to_dict())
```
"""

import logging
from typing import Any, List, Mapping

from tqdm import tqdm

from src.datasets.schema import BiasExample
from src.scoring.crows_pairs import score_crows_example
from src.scoring.schema import ScoredExample
from src.scoring.winobias import score_winobias_example

logger = logging.getLogger(__name__)


def evaluate_model_on_benchmark(
    model_wrapper: Any,
    model_cfg: Mapping[str, Any],
    benchmark_name: str,
    examples: List[BiasExample],
    verbose: bool = True,
) -> List[ScoredExample]:
    """
    Evaluate a model on a bias benchmark.

    This function routes to the appropriate scorer based on the benchmark name
    and processes all examples, returning scored results.

    Args:
        model_wrapper: Initialized model wrapper instance.
        model_cfg: Model configuration dict with id, family, version, type.
        benchmark_name: Name of the benchmark ("crows_pairs", "winobias").
        examples: List of BiasExample objects from the benchmark.
        verbose: If True, show progress bar and log statistics.

    Returns:
        List of ScoredExample objects, one per input example.

    Raises:
        ValueError: If benchmark_name is not supported.

    Examples:
        >>> from src.models import MaskedLMWrapper
        >>> from src.datasets import load_crows_pairs
        >>>
        >>> model = MaskedLMWrapper("bert-base-uncased")
        >>> model_cfg = {"id": "bert-base-uncased", "family": "bert", "version": 1, "type": "masked_lm"}
        >>> examples = load_crows_pairs()
        >>>
        >>> results = evaluate_model_on_benchmark(
        ...     model, model_cfg, "crows_pairs", examples
        ... )
        >>> print(f"Evaluated {len(results)} examples")
    """
    # Normalize benchmark name
    benchmark_name = benchmark_name.lower().replace("-", "_")

    # Select scorer function
    if benchmark_name == "crows_pairs":
        scorer_fn = score_crows_example
    elif benchmark_name == "winobias":
        scorer_fn = score_winobias_example
    else:
        raise ValueError(
            f"Unsupported benchmark: '{benchmark_name}'. "
            f"Supported benchmarks: 'crows_pairs', 'winobias'"
        )

    if verbose:
        logger.info(
            f"Evaluating {model_cfg.get('id', 'unknown')} on "
            f"{benchmark_name} ({len(examples)} examples)"
        )

    # Score all examples
    scored_examples = []
    failed_count = 0

    # Use tqdm for progress bar if verbose
    iterator = tqdm(examples, desc=f"Scoring {benchmark_name}") if verbose else examples

    for example in iterator:
        try:
            scored = scorer_fn(
                model_wrapper=model_wrapper,
                model_cfg=model_cfg,
                example=example,
            )
            scored_examples.append(scored)

        except Exception as e:
            logger.error(
                f"Failed to score example {example.id} on {benchmark_name}: {e}"
            )
            failed_count += 1

            # Create a failure record
            scored = ScoredExample(
                model_id=model_cfg.get("id", "unknown"),
                model_family=model_cfg.get("family", "unknown"),
                model_version=model_cfg.get("version", 0),
                benchmark=example.benchmark,
                example_id=example.id,
                bias_axis=example.bias_axis,
                bias_score=0.0,
                bias_direction=0,
                preferred_variant="refusal",
                stereotype_text=example.stereotype_text,
                anti_stereotype_text=example.anti_stereotype_text,
                sentence=example.sentence,
                raw_output=f"ERROR: {str(e)}",
                metadata={"error": str(e), "failed": True},
            )
            scored_examples.append(scored)

    if verbose:
        success_count = len(scored_examples) - failed_count
        logger.info(
            f"Completed evaluation: {success_count}/{len(examples)} successful, "
            f"{failed_count} failed"
        )

        # Log bias direction distribution
        if scored_examples:
            direction_counts = {
                "stereotype": sum(1 for s in scored_examples if s.bias_direction == 1),
                "anti": sum(1 for s in scored_examples if s.bias_direction == -1),
                "neutral/refusal": sum(1 for s in scored_examples if s.bias_direction == 0),
            }
            logger.info(f"Bias direction distribution: {direction_counts}")

    return scored_examples


def evaluate_multiple_models(
    models: Mapping[str, Any],
    model_configs: Mapping[str, Mapping[str, Any]],
    benchmark_name: str,
    examples: List[BiasExample],
    verbose: bool = True,
) -> Mapping[str, List[ScoredExample]]:
    """
    Evaluate multiple models on a single benchmark.

    Args:
        models: Dict mapping model_id to model wrapper instance.
        model_configs: Dict mapping model_id to model config dict.
        benchmark_name: Name of the benchmark.
        examples: List of BiasExample objects.
        verbose: If True, show progress and statistics.

    Returns:
        Dict mapping model_id to list of ScoredExample objects.

    Examples:
        >>> from src.models import load_all_models
        >>> from src.datasets import load_crows_pairs
        >>>
        >>> models = load_all_models(device="cuda")
        >>> configs = {mid: get_model_config(mid) for mid in models.keys()}
        >>> examples = load_crows_pairs()
        >>>
        >>> results = evaluate_multiple_models(
        ...     models, configs, "crows_pairs", examples
        ... )
        >>> for model_id, scored in results.items():
        ...     print(f"{model_id}: {len(scored)} results")
    """
    all_results = {}

    for model_id, model_wrapper in models.items():
        model_cfg = model_configs.get(model_id)

        if model_cfg is None:
            logger.error(f"Missing config for model {model_id}, skipping")
            continue

        if verbose:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Evaluating model: {model_id}")
            logger.info(f"{'=' * 70}")

        try:
            results = evaluate_model_on_benchmark(
                model_wrapper=model_wrapper,
                model_cfg=model_cfg,
                benchmark_name=benchmark_name,
                examples=examples,
                verbose=verbose,
            )
            all_results[model_id] = results

        except Exception as e:
            logger.error(f"Failed to evaluate {model_id} on {benchmark_name}: {e}")
            all_results[model_id] = []

    return all_results


def main():
    """Test the evaluator."""
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from src.datasets import load_benchmark
    from src.models import MaskedLMWrapper

    print("=" * 70)
    print("Evaluator Test")
    print("=" * 70)

    # Load a small sample of examples
    print("\nLoading CrowS-Pairs examples...")
    examples = load_benchmark("crows_pairs", verbose=False)
    print(f"Loaded {len(examples)} examples")

    # Take first 5 for quick test
    sample_examples = examples[:5]
    print(f"Using {len(sample_examples)} examples for testing")

    # Load model
    print("\nLoading bert-base-uncased...")
    model = MaskedLMWrapper("bert-base-uncased", device="cpu")

    model_cfg = {
        "id": "bert-base-uncased",
        "family": "bert",
        "version": 1,
        "type": "masked_lm",
    }

    # Evaluate
    print("\nEvaluating...")
    results = evaluate_model_on_benchmark(
        model_wrapper=model,
        model_cfg=model_cfg,
        benchmark_name="crows_pairs",
        examples=sample_examples,
        verbose=True,
    )

    # Show results
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    print(f"\nTotal results: {len(results)}")

    # Count by preferred variant
    variant_counts = {}
    for result in results:
        variant = result.preferred_variant
        variant_counts[variant] = variant_counts.get(variant, 0) + 1

    print(f"\nPreferred variant distribution:")
    for variant, count in sorted(variant_counts.items()):
        print(f"  {variant:15s}: {count}")

    # Show first result
    if results:
        print("\nFirst result:")
        print("-" * 70)
        first = results[0]
        print(f"  Example: {first.example_id}")
        print(f"  Bias axis: {first.bias_axis}")
        print(f"  Bias score: {first.bias_score:.4f}")
        print(f"  Direction: {first.bias_direction}")
        print(f"  Preferred: {first.preferred_variant}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()
