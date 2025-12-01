#!/usr/bin/env python3
# File: scripts/run_eval.py
"""
Driver script for running bias evaluations on language models.

This script orchestrates the complete evaluation pipeline:
1. Load model configurations and select models to evaluate
2. Load benchmark datasets
3. Run each model on each benchmark
4. Write results to JSONL files in results/raw/

## Usage

```bash
# Evaluate all models on all benchmarks
python scripts/run_eval.py

# Evaluate specific models
python scripts/run_eval.py --models bert-base-uncased,gpt2

# Evaluate specific benchmarks
python scripts/run_eval.py --benchmarks crows_pairs

# Specify device
python scripts/run_eval.py --device cuda

# Evaluate only local models (skip API models)
python scripts/run_eval.py --skip-api

# Dry run (load models and examples but don't evaluate)
python scripts/run_eval.py --dry-run
```

## Output

Results are written to: `results/raw/{model_id}__{benchmark}.jsonl`

Each line is a JSON object representing a ScoredExample:
```json
{
  "model_id": "bert-base-uncased",
  "model_family": "bert",
  "model_version": 1,
  "benchmark": "crows_pairs",
  "example_id": "crows_001",
  "bias_axis": "gender",
  "bias_score": 1.23,
  "bias_direction": 1,
  "preferred_variant": "stereotype",
  ...
}
```
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.datasets import load_benchmark
from src.models import load_all_models
from src.models.registry import get_model_config, list_available_models
from src.scoring.evaluator import evaluate_model_on_benchmark
from src.utils.config import load_benchmarks_config, load_models_config
from src.utils.paths import ensure_dir_exists, get_results_dir
from src.utils.random_utils import set_global_seed

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate language models on bias benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model IDs to evaluate. If not specified, evaluates all models.",
    )

    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip API-based models (useful when no API key is available)",
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Comma-separated list of benchmarks to run. If not specified, runs all benchmarks.",
    )

    # Device and performance
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device to run models on (default: cuda)",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write results (default: results/raw/)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result files",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Debugging
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load models and datasets but don't run evaluation",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def select_models(args) -> list[str]:
    """Select which models to evaluate based on arguments."""
    all_models = list_available_models()

    if args.models:
        # Use specified models
        model_ids = [m.strip() for m in args.models.split(",")]

        # Validate that all specified models exist
        invalid = [m for m in model_ids if m not in all_models]
        if invalid:
            logger.error(f"Unknown models: {invalid}")
            logger.error(f"Available models: {all_models}")
            sys.exit(1)

        return model_ids

    else:
        # Use all models
        selected = all_models

        # Filter out API models if requested
        if args.skip_api:
            model_configs = load_models_config()
            selected = [
                m["id"]
                for m in model_configs["models"]
                if m["id"] in selected and m["type"] != "generative_api"
            ]

        return selected


def select_benchmarks(args) -> list[str]:
    """Select which benchmarks to run based on arguments."""
    all_benchmarks = ["crows_pairs", "winobias"]

    if args.benchmarks:
        # Use specified benchmarks
        benchmark_names = [b.strip() for b in args.benchmarks.split(",")]

        # Validate
        invalid = [b for b in benchmark_names if b not in all_benchmarks]
        if invalid:
            logger.error(f"Unknown benchmarks: {invalid}")
            logger.error(f"Available benchmarks: {all_benchmarks}")
            sys.exit(1)

        return benchmark_names

    else:
        return all_benchmarks


def load_benchmark_examples(benchmark_name: str) -> list:
    """Load examples for a benchmark."""
    logger.info(f"Loading {benchmark_name} examples...")
    examples = load_benchmark(benchmark_name, verbose=False)
    logger.info(f"Loaded {len(examples)} examples from {benchmark_name}")
    return examples


def write_results_to_jsonl(
    results: list,
    output_path: Path,
    overwrite: bool = False,
):
    """Write scored results to JSONL file."""
    # Check if file exists
    if output_path.exists() and not overwrite:
        logger.warning(
            f"Output file already exists: {output_path}. "
            f"Use --overwrite to replace it. Skipping."
        )
        return

    # Ensure output directory exists
    ensure_dir_exists(output_path.parent)

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            json_line = json.dumps(result.to_dict())
            f.write(json_line + "\n")

    logger.info(f"Wrote {len(results)} results to {output_path}")


def main():
    """Main evaluation pipeline."""
    args = parse_args()
    setup_logging(args.verbose)

    logger.info("=" * 70)
    logger.info("Bias Drift Evaluation Pipeline")
    logger.info("=" * 70)

    # Set random seed
    set_global_seed(args.seed)
    logger.info(f"Set random seed: {args.seed}")

    # Select models and benchmarks
    model_ids = select_models(args)
    benchmark_names = select_benchmarks(args)

    logger.info(f"\nModels to evaluate ({len(model_ids)}): {model_ids}")
    logger.info(f"Benchmarks to run ({len(benchmark_names)}): {benchmark_names}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_results_dir("raw")

    logger.info(f"Output directory: {output_dir}")

    # Dry run mode
    if args.dry_run:
        logger.info("\nDRY RUN MODE - Loading models and datasets only\n")

    # Load models
    logger.info("\n" + "=" * 70)
    logger.info("Loading Models")
    logger.info("=" * 70)

    try:
        # Determine if we need API client
        model_configs = load_models_config()
        model_types = {
            m["id"]: m["type"]
            for m in model_configs["models"]
            if m["id"] in model_ids
        }

        needs_api = any(t == "generative_api" for t in model_types.values())

        api_client = None
        if needs_api:
            logger.info("API-based models detected, initializing OpenAI client...")
            try:
                from openai import OpenAI
                import os

                api_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.error(
                    "openai package not installed. Install with: pip install openai"
                )
                logger.error("Skipping API-based models")
                model_ids = [
                    mid for mid, mtype in model_types.items()
                    if mtype != "generative_api"
                ]
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                logger.error("Skipping API-based models")
                model_ids = [
                    mid for mid, mtype in model_types.items()
                    if mtype != "generative_api"
                ]

        # Load selected models
        models = load_all_models(
            device=args.device,
            only_ids=model_ids,
            api_client=api_client,
        )

        logger.info(f"\nSuccessfully loaded {len(models)}/{len(model_ids)} models")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        sys.exit(1)

    # Get model configs
    model_configs_dict = {}
    for model_id in models.keys():
        try:
            cfg = get_model_config(model_id)
            model_configs_dict[model_id] = cfg
        except Exception as e:
            logger.error(f"Failed to get config for {model_id}: {e}")

    if args.dry_run:
        logger.info("\nDry run complete. Exiting.")
        return

    # Run evaluation for each benchmark
    for benchmark_name in benchmark_names:
        logger.info("\n" + "=" * 70)
        logger.info(f"Benchmark: {benchmark_name}")
        logger.info("=" * 70)

        # Load benchmark examples
        try:
            examples = load_benchmark_examples(benchmark_name)
        except Exception as e:
            logger.error(f"Failed to load {benchmark_name}: {e}")
            continue

        # Evaluate each model
        for model_id, model_wrapper in models.items():
            logger.info(f"\n{'─' * 70}")
            logger.info(f"Evaluating: {model_id} on {benchmark_name}")
            logger.info(f"{'─' * 70}")

            model_cfg = model_configs_dict.get(model_id)
            if model_cfg is None:
                logger.error(f"Missing config for {model_id}, skipping")
                continue

            try:
                # Run evaluation
                results = evaluate_model_on_benchmark(
                    model_wrapper=model_wrapper,
                    model_cfg=model_cfg,
                    benchmark_name=benchmark_name,
                    examples=examples,
                    verbose=True,
                )

                # Write results to JSONL
                output_filename = f"{model_id}__{benchmark_name}.jsonl"
                output_path = output_dir / output_filename

                write_results_to_jsonl(
                    results=results,
                    output_path=output_path,
                    overwrite=args.overwrite,
                )

            except Exception as e:
                logger.error(
                    f"Failed to evaluate {model_id} on {benchmark_name}: {e}",
                    exc_info=True,
                )

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("Evaluation Complete!")
    logger.info("=" * 70)
    logger.info(f"Results written to: {output_dir}")
    logger.info(f"Models evaluated: {len(models)}")
    logger.info(f"Benchmarks run: {len(benchmark_names)}")


if __name__ == "__main__":
    main()
