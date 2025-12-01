# File: src/datasets/crows_pairs.py
"""
CrowS-Pairs dataset loader.

CrowS-Pairs (Crowd-Sourced Stereotype Pairs) is a benchmark for measuring
stereotypical biases in masked language models. It consists of sentence pairs
where one sentence is more stereotypical and the other is less stereotypical
or anti-stereotypical.

## Expected Data Format

The loader expects a CSV file at: `data/crows_pairs/raw/crows_pairs.csv`

### Required Columns:
- `id`: Unique identifier for each example (integer or string)
- `sent_more`: The more stereotypical sentence
- `sent_less`: The less stereotypical/anti-stereotypical sentence
- `bias_type`: The type/axis of bias (e.g., "gender", "race", "religion")

### Optional Columns (preserved in `meta`):
- `stereo_antistereo`: Direction indicator (which sentence is stereotypical)
- `annotations`: Annotator agreement/confidence information
- Any other columns will be stored in the `meta` dictionary

## Data Source

If you don't have the data yet, download from:
- Original paper: https://aclanthology.org/2020.emnlp-main.154/
- GitHub: https://github.com/nyu-mll/crows-pairs

## Usage

```python
from src.datasets.crows_pairs import load_crows_pairs

# Load all examples
examples = load_crows_pairs()
print(f"Loaded {len(examples)} CrowS-Pairs examples")

# Filter by bias axis
gender_examples = [ex for ex in examples if ex.bias_axis == "gender"]
```
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import load_benchmarks_config
from src.utils.paths import get_project_root

from .schema import BiasExample

logger = logging.getLogger(__name__)


def load_crows_pairs(
    data_path: Optional[Path] = None,
    verbose: bool = True,
) -> list[BiasExample]:
    """
    Load CrowS-Pairs dataset into unified BiasExample format.

    Args:
        data_path: Optional custom path to the CSV file. If None, uses the path
            from configs/benchmarks.yaml.
        verbose: If True, logs loading statistics and warnings.

    Returns:
        List of BiasExample objects, one per row in the CSV.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If required columns are missing from the CSV.
        KeyError: If expected columns are not found in the CSV.

    Examples:
        >>> # Load using default config path
        >>> examples = load_crows_pairs()
        >>> print(f"Loaded {len(examples)} examples")
        >>>
        >>> # Load from custom path
        >>> from pathlib import Path
        >>> examples = load_crows_pairs(Path("custom/path/crows_pairs.csv"))
    """
    # Determine data path
    if data_path is None:
        config = load_benchmarks_config()
        crows_config = config.get("crows_pairs", {})
        raw_file = crows_config.get("raw_file", "data/crows_pairs/raw/crows_pairs.csv")
        data_path = get_project_root() / raw_file
    else:
        data_path = Path(data_path)

    # Check file existence
    if not data_path.exists():
        raise FileNotFoundError(
            f"CrowS-Pairs data file not found: {data_path}\n"
            f"Please download the dataset from:\n"
            f"  https://github.com/nyu-mll/crows-pairs\n"
            f"and place it at: {data_path}"
        )

    if verbose:
        logger.info(f"Loading CrowS-Pairs from: {data_path}")

    # Load CSV
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise IOError(f"Error reading CSV file {data_path}: {e}")

    # Validate required columns
    required_columns = ["sent_more", "sent_less", "bias_type"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"CrowS-Pairs CSV is missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Expected columns: {required_columns} (and optionally 'id')"
        )

    # Check for ID column, create if missing
    if "id" not in df.columns:
        if verbose:
            logger.warning("No 'id' column found. Generating IDs from row indices.")
        df["id"] = [f"crows_{i}" for i in range(len(df))]

    # Convert to BiasExample objects
    examples = []
    seen_ids = set()

    for idx, row in df.iterrows():
        # Get or generate ID
        example_id = str(row["id"])

        # Check for duplicate IDs
        if example_id in seen_ids:
            raise ValueError(
                f"Duplicate ID found in CrowS-Pairs data: {example_id} (row {idx})"
            )
        seen_ids.add(example_id)

        # Extract core fields
        bias_axis = str(row["bias_type"]).strip()
        stereotype_text = str(row["sent_more"]).strip()
        anti_stereotype_text = str(row["sent_less"]).strip()

        # Collect metadata from other columns
        meta = {}
        for col in df.columns:
            if col not in ["id", "sent_more", "sent_less", "bias_type"]:
                # Store additional columns in meta
                meta[col] = row[col]

        # Create BiasExample
        example = BiasExample(
            id=example_id,
            benchmark="crows_pairs",
            bias_axis=bias_axis,
            type="pair",
            stereotype_text=stereotype_text,
            anti_stereotype_text=anti_stereotype_text,
            sentence=None,
            answer_options=None,
            correct_option=None,
            meta=meta,
        )

        examples.append(example)

    # Log statistics
    if verbose:
        logger.info(f"Successfully loaded {len(examples)} CrowS-Pairs examples")

        # Count examples by bias axis
        bias_distribution = Counter(ex.bias_axis for ex in examples)
        logger.info("Bias axis distribution:")
        for bias_axis, count in sorted(bias_distribution.items()):
            logger.info(f"  {bias_axis}: {count}")

    return examples


def main():
    """Command-line interface for testing the loader."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 70)
    print("CrowS-Pairs Dataset Loader")
    print("=" * 70)

    try:
        examples = load_crows_pairs(verbose=True)

        print(f"\n✓ Successfully loaded {len(examples)} examples")

        # Show first example
        if examples:
            print("\nFirst example:")
            print("-" * 70)
            first = examples[0]
            print(f"ID: {first.id}")
            print(f"Benchmark: {first.benchmark}")
            print(f"Bias Axis: {first.bias_axis}")
            print(f"Type: {first.type}")
            print(f"\nStereotype Text:\n  {first.stereotype_text}")
            print(f"\nAnti-Stereotype Text:\n  {first.anti_stereotype_text}")
            print(f"\nMetadata: {first.meta}")
            print("-" * 70)

        # Show distribution
        bias_counts = Counter(ex.bias_axis for ex in examples)
        print(f"\nBias axis distribution:")
        for bias_axis, count in sorted(bias_counts.items(), key=lambda x: -x[1]):
            print(f"  {bias_axis:20s}: {count:4d}")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nTo fix this:")
        print("  1. Download CrowS-Pairs from: https://github.com/nyu-mll/crows-pairs")
        print("  2. Create the directory: mkdir -p data/crows_pairs/raw/")
        print("  3. Place the CSV file at: data/crows_pairs/raw/crows_pairs.csv")
    except Exception as e:
        print(f"\n✗ Error loading CrowS-Pairs: {e}")
        raise


if __name__ == "__main__":
    main()
