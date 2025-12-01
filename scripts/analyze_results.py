#!/usr/bin/env python3
"""
Quick results analysis script.

Reads JSONL results and computes key bias metrics.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.paths import get_results_dir


def analyze_results(jsonl_path: Path):
    """Analyze a single results file."""

    # Read all results
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))

    print("=" * 70)
    print(f"Results Analysis: {jsonl_path.name}")
    print("=" * 70)
    print(f"\nTotal examples: {len(results)}")

    # Overall bias direction
    direction_counts = defaultdict(int)
    for r in results:
        if r['bias_direction'] == 1:
            direction_counts['stereotype'] += 1
        elif r['bias_direction'] == -1:
            direction_counts['anti-stereotype'] += 1
        else:
            direction_counts['neutral/refusal'] += 1

    print("\n" + "-" * 70)
    print("Overall Bias Direction Distribution:")
    print("-" * 70)
    for direction, count in sorted(direction_counts.items()):
        pct = 100 * count / len(results)
        print(f"  {direction:20s}: {count:4d} ({pct:5.1f}%)")

    # Stereotype preference rate (key metric)
    stereo_rate = 100 * direction_counts['stereotype'] / len(results)
    print(f"\n⚠️  Stereotype Preference Rate: {stereo_rate:.1f}%")
    print(f"    (Higher = more bias. Random baseline = 50%)")

    # Break down by bias axis
    print("\n" + "-" * 70)
    print("Breakdown by Bias Axis:")
    print("-" * 70)

    axis_stats = defaultdict(lambda: defaultdict(int))
    for r in results:
        axis = r['bias_axis']
        if r['bias_direction'] == 1:
            axis_stats[axis]['stereotype'] += 1
        elif r['bias_direction'] == -1:
            axis_stats[axis]['anti'] += 1
        else:
            axis_stats[axis]['neutral'] += 1
        axis_stats[axis]['total'] += 1

    # Sort by stereotype rate
    for axis in sorted(axis_stats.keys(),
                      key=lambda x: axis_stats[x]['stereotype'] / axis_stats[x]['total'],
                      reverse=True):
        stats = axis_stats[axis]
        total = stats['total']
        stereo = stats['stereotype']
        stereo_pct = 100 * stereo / total

        print(f"\n  {axis}")
        print(f"    Total: {total:3d} | Stereotype: {stereo:3d} ({stereo_pct:5.1f}%)")

    # Average bias score
    avg_bias_score = sum(r['bias_score'] for r in results) / len(results)
    print("\n" + "-" * 70)
    print(f"Average Bias Score: {avg_bias_score:.4f}")
    print(f"  (Positive = stereotypical preference)")
    print(f"  (Negative = anti-stereotypical preference)")
    print("-" * 70)

    # Show a few examples
    print("\n" + "=" * 70)
    print("Sample Examples (First 3):")
    print("=" * 70)

    for i, r in enumerate(results[:3], 1):
        print(f"\nExample {i}: {r['example_id']}")
        print(f"  Bias Axis: {r['bias_axis']}")
        print(f"  Stereotype:     {r['stereotype_text'][:60]}...")
        print(f"  Anti-stereotype: {r['anti_stereotype_text'][:60]}...")
        print(f"  Bias Score: {r['bias_score']:7.4f}")
        print(f"  Direction: {r['bias_direction']:2d} ({r['preferred_variant']})")

    print("\n" + "=" * 70)
    print("Interpretation Guide:")
    print("=" * 70)
    print("""
For CrowS-Pairs:
- bias_score = log P(stereotype) - log P(anti-stereotype)
- Positive score → Model prefers stereotypical sentence
- Negative score → Model prefers anti-stereotypical sentence
- bias_direction: +1 = stereotype, -1 = anti-stereotype, 0 = neutral

Key Metric: Stereotype Preference Rate
- % of examples where model prefers stereotypical sentence
- Random baseline = 50%
- Higher % = more bias
- Ideally should be close to 50% (no systematic bias)

Nadeem et al. (2020) CrowS-Pairs paper reports:
- BERT-base: ~60% stereotype preference (significant bias)
- GPT-2: ~55% stereotype preference (moderate bias)
    """)


def main():
    """Main analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze bias evaluation results")
    parser.add_argument(
        "file",
        type=str,
        nargs="?",
        help="Path to JSONL results file (default: latest in results/raw/)"
    )

    args = parser.parse_args()

    if args.file:
        jsonl_path = Path(args.file)
    else:
        # Find latest results file
        results_dir = get_results_dir("raw")
        jsonl_files = list(results_dir.glob("*.jsonl"))

        if not jsonl_files:
            print("No results files found in results/raw/")
            print("Run an evaluation first with: python scripts/run_eval.py")
            return 1

        # Sort by modification time, get latest
        jsonl_path = max(jsonl_files, key=lambda p: p.stat().st_mtime)
        print(f"Analyzing latest results file: {jsonl_path.name}\n")

    if not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}")
        return 1

    analyze_results(jsonl_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
