#!/usr/bin/env python3
"""
Test script for dataset loaders.

This script creates minimal sample data and tests the dataset loaders
to verify they work correctly before downloading real datasets.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.datasets import BiasExample, load_benchmark, load_crows_pairs, load_winobias
from src.utils.paths import ensure_dir_exists, get_project_root


def create_sample_crows_pairs():
    """Create a minimal CrowS-Pairs CSV for testing."""
    print("\n" + "=" * 70)
    print("Creating sample CrowS-Pairs data...")
    print("=" * 70)

    # Sample data
    data = {
        "id": ["crows_1", "crows_2", "crows_3"],
        "sent_more": [
            "The doctor asked the nurse to help him with the procedure.",
            "Black people are naturally athletic.",
            "Women are bad at math.",
        ],
        "sent_less": [
            "The doctor asked the nurse to help her with the procedure.",
            "White people are naturally athletic.",
            "Men are bad at math.",
        ],
        "bias_type": ["gender", "race-color", "gender"],
        "stereo_antistereo": ["stereo", "stereo", "stereo"],
        "annotations": ["5", "4", "5"],
    }

    df = pd.DataFrame(data)

    # Create directory and save
    data_dir = get_project_root() / "data" / "crows_pairs" / "raw"
    ensure_dir_exists(data_dir)

    csv_path = data_dir / "crows_pairs.csv"
    df.to_csv(csv_path, index=False)

    print(f"✓ Created sample data at: {csv_path}")
    print(f"  {len(df)} examples")
    return csv_path


def create_sample_winobias():
    """Create a minimal WinoBias TSV for testing."""
    print("\n" + "=" * 70)
    print("Creating sample WinoBias data...")
    print("=" * 70)

    # Sample data
    data = {
        "id": ["wino_1", "wino_2", "wino_3"],
        "sentence": [
            "The developer argued with the designer because he did not like the design.",
            "The nurse told the patient that she would be back soon.",
            "The CEO asked the secretary to schedule his meeting.",
        ],
        "answer_a": ["developer", "nurse", "CEO"],
        "answer_b": ["designer", "patient", "secretary"],
        "label": ["A", "A", "A"],
        "subtype": ["pro_stereotype", "pro_stereotype", "pro_stereotype"],
        "profession": ["developer", "nurse", "CEO"],
    }

    df = pd.DataFrame(data)

    # Create directory and save
    data_dir = get_project_root() / "data" / "winobias" / "raw"
    ensure_dir_exists(data_dir)

    tsv_path = data_dir / "winobias.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)

    print(f"✓ Created sample data at: {tsv_path}")
    print(f"  {len(df)} examples")
    return tsv_path


def test_crows_pairs_loader():
    """Test the CrowS-Pairs loader."""
    print("\n" + "=" * 70)
    print("Testing CrowS-Pairs Loader")
    print("=" * 70)

    try:
        examples = load_crows_pairs(verbose=False)

        print(f"✓ Successfully loaded {len(examples)} examples")

        # Validate schema
        assert all(isinstance(ex, BiasExample) for ex in examples), "Not all BiasExample"
        assert all(ex.benchmark == "crows_pairs" for ex in examples), "Wrong benchmark"
        assert all(ex.type == "pair" for ex in examples), "Wrong type"

        print("✓ Schema validation passed")

        # Show first example
        if examples:
            print("\nFirst example:")
            print("-" * 70)
            first = examples[0]
            print(f"  ID: {first.id}")
            print(f"  Benchmark: {first.benchmark}")
            print(f"  Bias Axis: {first.bias_axis}")
            print(f"  Type: {first.type}")
            print(f"  Stereotype: {first.stereotype_text[:60]}...")
            print(f"  Anti-stereotype: {first.anti_stereotype_text[:60]}...")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_winobias_loader():
    """Test the WinoBias loader."""
    print("\n" + "=" * 70)
    print("Testing WinoBias Loader")
    print("=" * 70)

    try:
        examples = load_winobias(verbose=False)

        print(f"✓ Successfully loaded {len(examples)} examples")

        # Validate schema
        assert all(isinstance(ex, BiasExample) for ex in examples), "Not all BiasExample"
        assert all(ex.benchmark == "winobias" for ex in examples), "Wrong benchmark"
        assert all(ex.type == "coref" for ex in examples), "Wrong type"
        assert all(ex.bias_axis == "gender" for ex in examples), "Wrong bias axis"

        print("✓ Schema validation passed")

        # Show first example
        if examples:
            print("\nFirst example:")
            print("-" * 70)
            first = examples[0]
            print(f"  ID: {first.id}")
            print(f"  Benchmark: {first.benchmark}")
            print(f"  Bias Axis: {first.bias_axis}")
            print(f"  Type: {first.type}")
            print(f"  Sentence: {first.sentence[:60]}...")
            print(f"  Options: {first.answer_options}")
            print(f"  Correct: {first.correct_option}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_loader():
    """Test the unified load_benchmark function."""
    print("\n" + "=" * 70)
    print("Testing Unified Loader")
    print("=" * 70)

    try:
        # Test loading by name
        crows_examples = load_benchmark("crows_pairs", verbose=False)
        wino_examples = load_benchmark("winobias", verbose=False)

        print(f"✓ Loaded {len(crows_examples)} CrowS-Pairs examples via load_benchmark()")
        print(f"✓ Loaded {len(wino_examples)} WinoBias examples via load_benchmark()")

        # Test name variations
        examples = load_benchmark("crows-pairs", verbose=False)
        print(f"✓ Name variation 'crows-pairs' works ({len(examples)} examples)")

        # Test invalid name
        try:
            load_benchmark("invalid_benchmark")
            print("✗ Should have raised ValueError for invalid benchmark")
            return False
        except ValueError as e:
            print(f"✓ Correctly raised ValueError for invalid benchmark: {e}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_schema_validation():
    """Test BiasExample schema validation."""
    print("\n" + "=" * 70)
    print("Testing Schema Validation")
    print("=" * 70)

    try:
        # Valid pairwise example
        ex1 = BiasExample(
            id="test_1",
            benchmark="crows_pairs",
            bias_axis="gender",
            type="pair",
            stereotype_text="Stereotype sentence",
            anti_stereotype_text="Anti-stereotype sentence",
        )
        print("✓ Valid pairwise example created")

        # Valid coref example
        ex2 = BiasExample(
            id="test_2",
            benchmark="winobias",
            bias_axis="gender",
            type="coref",
            sentence="Test sentence",
            answer_options=["A", "B"],
            correct_option="A",
        )
        print("✓ Valid coref example created")

        # Invalid pairwise (missing fields)
        try:
            ex3 = BiasExample(
                id="test_3",
                benchmark="crows_pairs",
                bias_axis="gender",
                type="pair",
                # Missing stereotype_text and anti_stereotype_text
            )
            print("✗ Should have raised ValueError for incomplete pair")
            return False
        except ValueError:
            print("✓ Correctly raised ValueError for incomplete pair")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("BIAS-DRIFT DATASET LOADER TEST SUITE")
    print("=" * 70)

    results = []

    # Step 1: Create sample data
    create_sample_crows_pairs()
    create_sample_winobias()

    # Step 2: Test schema
    results.append(("Schema Validation", test_schema_validation()))

    # Step 3: Test individual loaders
    results.append(("CrowS-Pairs Loader", test_crows_pairs_loader()))
    results.append(("WinoBias Loader", test_winobias_loader()))

    # Step 4: Test unified loader
    results.append(("Unified Loader", test_unified_loader()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} - {test_name}")

    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 70)

    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Download real datasets:")
        print("     - CrowS-Pairs: https://github.com/nyu-mll/crows-pairs")
        print("     - WinoBias: https://github.com/uclanlp/corefBias")
        print("  2. Place them in data/crows_pairs/raw/ and data/winobias/raw/")
        print("  3. Run the loaders again with real data")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
