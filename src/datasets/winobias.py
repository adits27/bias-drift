# File: src/datasets/winobias.py
"""
WinoBias dataset loader.

WinoBias is a Winograd schema dataset for detecting gender bias in coreference
resolution systems. It contains sentences with gender-neutral pronouns that
refer to people in different professions, testing whether systems exhibit
stereotypical gender associations.

## Expected Data Format

The loader expects a TSV file at: `data/winobias/raw/winobias.tsv`

### Required Columns:
- `id`: Unique identifier for each example
- `sentence`: Sentence containing a pronoun and candidate referents
- `answer_a`: First candidate referent (e.g., "developer")
- `answer_b`: Second candidate referent (e.g., "designer")
- `label`: Correct answer ("A" or "B")

### Optional Columns (preserved in `meta`):
- `subtype`: "pro_stereotype" or "anti_stereotype"
  - pro_stereotype: pronoun refers to stereotypically associated profession
  - anti_stereotype: pronoun refers to counter-stereotypical profession
- `profession`: The professional occupation mentioned
- Any other columns will be stored in the `meta` dictionary

## Data Source

If you don't have the data yet, download from:
- Original paper: https://aclanthology.org/N18-2003/
- GitHub: https://github.com/uclanlp/corefBias

## Usage

```python
from src.datasets.winobias import load_winobias

# Load all examples
examples = load_winobias()
print(f"Loaded {len(examples)} WinoBias examples")

# Filter by subtype
pro_stereo = [ex for ex in examples if ex.meta.get("subtype") == "pro_stereotype"]
anti_stereo = [ex for ex in examples if ex.meta.get("subtype") == "anti_stereotype"]
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


def load_winobias(
    data_path: Optional[Path] = None,
    verbose: bool = True,
) -> list[BiasExample]:
    """
    Load WinoBias dataset into unified BiasExample format.

    Args:
        data_path: Optional custom path to the TSV file. If None, uses the path
            from configs/benchmarks.yaml.
        verbose: If True, logs loading statistics and warnings.

    Returns:
        List of BiasExample objects, one per row in the TSV.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If required columns are missing or data is malformed.
        KeyError: If expected columns are not found in the TSV.

    Examples:
        >>> # Load using default config path
        >>> examples = load_winobias()
        >>> print(f"Loaded {len(examples)} examples")
        >>>
        >>> # Load from custom path
        >>> from pathlib import Path
        >>> examples = load_winobias(Path("custom/path/winobias.tsv"))
    """
    # Determine data path
    if data_path is None:
        config = load_benchmarks_config()
        wino_config = config.get("winobias", {})
        raw_file = wino_config.get("raw_file", "data/winobias/raw/winobias.tsv")
        data_path = get_project_root() / raw_file
    else:
        data_path = Path(data_path)

    # Check file existence
    if not data_path.exists():
        raise FileNotFoundError(
            f"WinoBias data file not found: {data_path}\n"
            f"Please download the dataset from:\n"
            f"  https://github.com/uclanlp/corefBias\n"
            f"and place it at: {data_path}"
        )

    if verbose:
        logger.info(f"Loading WinoBias from: {data_path}")

    # Load TSV
    try:
        df = pd.read_csv(data_path, sep="\t")
    except Exception as e:
        raise IOError(f"Error reading TSV file {data_path}: {e}")

    # Validate required columns
    required_columns = ["sentence", "answer_a", "answer_b", "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"WinoBias TSV is missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Expected columns: {required_columns} (and optionally 'id')"
        )

    # Check for ID column, create if missing
    if "id" not in df.columns:
        if verbose:
            logger.warning("No 'id' column found. Generating IDs from row indices.")
        df["id"] = [f"wino_{i}" for i in range(len(df))]

    # Convert to BiasExample objects
    examples = []
    seen_ids = set()

    for idx, row in df.iterrows():
        # Get or generate ID
        example_id = str(row["id"])

        # Check for duplicate IDs
        if example_id in seen_ids:
            raise ValueError(
                f"Duplicate ID found in WinoBias data: {example_id} (row {idx})"
            )
        seen_ids.add(example_id)

        # Extract core fields
        sentence = str(row["sentence"]).strip()
        answer_a = str(row["answer_a"]).strip()
        answer_b = str(row["answer_b"]).strip()
        label = str(row["label"]).strip().upper()

        # Validate label
        if label not in ["A", "B"]:
            raise ValueError(
                f"Invalid label '{label}' for example {example_id} (row {idx}). "
                f"Expected 'A' or 'B'."
            )

        # Determine correct option based on label
        answer_options = [answer_a, answer_b]
        correct_option = answer_a if label == "A" else answer_b

        # Collect metadata from other columns
        meta = {}
        for col in df.columns:
            if col not in ["id", "sentence", "answer_a", "answer_b", "label"]:
                # Store additional columns in meta
                meta[col] = row[col]

        # WinoBias is gender bias focused
        bias_axis = "gender"

        # Create BiasExample
        example = BiasExample(
            id=example_id,
            benchmark="winobias",
            bias_axis=bias_axis,
            type="coref",
            stereotype_text=None,
            anti_stereotype_text=None,
            sentence=sentence,
            answer_options=answer_options,
            correct_option=correct_option,
            meta=meta,
        )

        examples.append(example)

    # Log statistics
    if verbose:
        logger.info(f"Successfully loaded {len(examples)} WinoBias examples")

        # Count examples by subtype if available
        if examples and "subtype" in examples[0].meta:
            subtype_distribution = Counter(
                ex.meta.get("subtype", "unknown") for ex in examples
            )
            logger.info("Subtype distribution:")
            for subtype, count in sorted(subtype_distribution.items()):
                logger.info(f"  {subtype}: {count}")

        # Count examples by profession if available
        if examples and "profession" in examples[0].meta:
            profession_distribution = Counter(
                ex.meta.get("profession", "unknown") for ex in examples
            )
            logger.info(f"Number of unique professions: {len(profession_distribution)}")

    return examples


def main():
    """Command-line interface for testing the loader."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 70)
    print("WinoBias Dataset Loader")
    print("=" * 70)

    try:
        examples = load_winobias(verbose=True)

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
            print(f"\nSentence:\n  {first.sentence}")
            print(f"\nAnswer Options: {first.answer_options}")
            print(f"Correct Option: {first.correct_option}")
            print(f"\nMetadata: {first.meta}")
            print("-" * 70)

        # Show distribution by subtype
        if examples and "subtype" in examples[0].meta:
            subtype_counts = Counter(ex.meta.get("subtype") for ex in examples)
            print(f"\nSubtype distribution:")
            for subtype, count in sorted(subtype_counts.items()):
                print(f"  {subtype:20s}: {count:4d}")

        # Show some professions if available
        if examples and "profession" in examples[0].meta:
            professions = set(ex.meta.get("profession") for ex in examples)
            print(f"\nNumber of unique professions: {len(professions)}")
            print(f"Sample professions: {sorted(list(professions))[:10]}")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nTo fix this:")
        print("  1. Download WinoBias from: https://github.com/uclanlp/corefBias")
        print("  2. Create the directory: mkdir -p data/winobias/raw/")
        print("  3. Place the TSV file at: data/winobias/raw/winobias.tsv")
    except Exception as e:
        print(f"\n✗ Error loading WinoBias: {e}")
        raise


if __name__ == "__main__":
    main()
