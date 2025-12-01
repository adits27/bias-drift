#!/usr/bin/env python3
"""
Statistical analysis of bias evaluation results.

Computes confidence intervals, significance tests, and effect sizes.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import math

import numpy as np
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.paths import get_results_dir


def load_results(jsonl_path):
    """Load results from JSONL file."""
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_confidence_interval(proportion, n, confidence=0.95):
    """
    Compute Wilson score confidence interval for a proportion.

    Args:
        proportion: Sample proportion (0-1)
        n: Sample size
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound) as proportions
    """
    z = stats.norm.ppf((1 + confidence) / 2)

    denominator = 1 + z**2/n
    centre_adjusted_probability = proportion + z*z / (2*n)
    adjusted_standard_deviation = math.sqrt((proportion*(1 - proportion) + z*z/(4*n)) / n)

    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator

    return lower_bound, upper_bound


def chi_square_test(observed_stereo, total, expected_proportion=0.5):
    """
    Perform chi-square goodness-of-fit test.

    Tests whether observed stereotype preference differs from expected (default 50%).

    Args:
        observed_stereo: Number of stereotypical preferences
        total: Total number of examples
        expected_proportion: Expected proportion under null hypothesis

    Returns:
        (chi2_statistic, p_value, is_significant)
    """
    observed_anti = total - observed_stereo
    expected_stereo = total * expected_proportion
    expected_anti = total * (1 - expected_proportion)

    chi2 = ((observed_stereo - expected_stereo)**2 / expected_stereo +
            (observed_anti - expected_anti)**2 / expected_anti)

    # Degrees of freedom = 1 for binary outcome
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    is_significant = p_value < 0.05

    return chi2, p_value, is_significant


def cohens_h(p1, p2):
    """
    Compute Cohen's h effect size for difference between two proportions.

    Effect size interpretation:
    - Small: h = 0.2
    - Medium: h = 0.5
    - Large: h = 0.8

    Args:
        p1, p2: Two proportions to compare

    Returns:
        Cohen's h effect size
    """
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    return phi1 - phi2


def two_proportion_ztest(count1, nobs1, count2, nobs2):
    """
    Two-proportion z-test for comparing stereotype rates between models.

    Args:
        count1: Number of stereotypical preferences in sample 1
        nobs1: Total observations in sample 1
        count2: Number of stereotypical preferences in sample 2
        nobs2: Total observations in sample 2

    Returns:
        (z_statistic, p_value, is_significant)
    """
    p1 = count1 / nobs1
    p2 = count2 / nobs2

    # Pooled proportion
    p_pooled = (count1 + count2) / (nobs1 + nobs2)

    # Standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/nobs1 + 1/nobs2))

    # Z-statistic
    if se == 0:
        z = 0
    else:
        z = (p1 - p2) / se

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    is_significant = p_value < 0.05

    return z, p_value, is_significant


def analyze_crows_statistics(results, model_id):
    """Compute statistics for CrowS-Pairs results."""
    total = len(results)
    stereo_count = sum(1 for r in results if r['bias_direction'] == 1)
    stereo_rate = stereo_count / total

    # Confidence interval
    ci_lower, ci_upper = compute_confidence_interval(stereo_rate, total)

    # Chi-square test against 50%
    chi2, p_val, is_sig = chi_square_test(stereo_count, total)

    # Bias scores
    bias_scores = [r['bias_score'] for r in results]
    mean_bias = np.mean(bias_scores)
    std_bias = np.std(bias_scores)

    return {
        'model_id': model_id,
        'total': total,
        'stereo_count': stereo_count,
        'stereo_rate': stereo_rate,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'chi2_statistic': chi2,
        'p_value': p_val,
        'significant': is_sig,
        'mean_bias_score': mean_bias,
        'std_bias_score': std_bias,
    }


def analyze_winobias_statistics(results, model_id):
    """Compute statistics for WinoBias results."""
    total = len(results)

    # Overall
    stereo_count = sum(1 for r in results if r['bias_direction'] == 1)
    stereo_rate = stereo_count / total

    # By subtype
    pro_examples = [r for r in results if r.get('metadata', {}).get('subtype') == 'pro_stereotype']
    anti_examples = [r for r in results if r.get('metadata', {}).get('subtype') == 'anti_stereotype']

    pro_correct = sum(1 for r in pro_examples if r.get('metadata', {}).get('is_correct', False))
    anti_correct = sum(1 for r in anti_examples if r.get('metadata', {}).get('is_correct', False))

    pro_acc = pro_correct / len(pro_examples) if pro_examples else 0
    anti_acc = anti_correct / len(anti_examples) if anti_examples else 0
    acc_gap = pro_acc - anti_acc

    # CI for accuracy gap
    ci_lower, ci_upper = compute_confidence_interval(stereo_rate, total)

    # Test if accuracy gap is significant
    if pro_examples and anti_examples:
        z, p_val, is_sig = two_proportion_ztest(
            pro_correct, len(pro_examples),
            anti_correct, len(anti_examples)
        )
    else:
        z, p_val, is_sig = 0, 1.0, False

    return {
        'model_id': model_id,
        'total': total,
        'pro_accuracy': pro_acc,
        'anti_accuracy': anti_acc,
        'accuracy_gap': acc_gap,
        'gap_z_statistic': z,
        'gap_p_value': p_val,
        'gap_significant': is_sig,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
    }


def compare_models(stats1, stats2, benchmark='crows'):
    """Compare two models statistically."""
    model1 = stats1['model_id']
    model2 = stats2['model_id']

    print(f"\n{'=' * 70}")
    print(f"Pairwise Comparison: {model1} vs {model2}")
    print(f"{'=' * 70}")

    if benchmark == 'crows':
        rate1 = stats1['stereo_rate']
        rate2 = stats2['stereo_rate']

        print(f"\nStereotype Preference Rates:")
        print(f"  {model1:30s}: {100*rate1:5.1f}% [{100*stats1['ci_95_lower']:5.1f}%, {100*stats1['ci_95_upper']:5.1f}%]")
        print(f"  {model2:30s}: {100*rate2:5.1f}% [{100*stats2['ci_95_lower']:5.1f}%, {100*stats2['ci_95_upper']:5.1f}%]")
        print(f"  Difference: {100*(rate2 - rate1):+.1f}pp")

        # Two-proportion z-test
        z, p_val, is_sig = two_proportion_ztest(
            stats1['stereo_count'], stats1['total'],
            stats2['stereo_count'], stats2['total']
        )

        print(f"\nStatistical Test:")
        print(f"  Z-statistic: {z:.4f}")
        print(f"  P-value: {p_val:.4f}")
        print(f"  Significant at α=0.05: {'YES' if is_sig else 'NO'}")

        # Effect size
        h = cohens_h(rate1, rate2)
        print(f"  Cohen's h: {h:.4f}", end="")
        if abs(h) < 0.2:
            print(" (negligible)")
        elif abs(h) < 0.5:
            print(" (small)")
        elif abs(h) < 0.8:
            print(" (medium)")
        else:
            print(" (large)")

    elif benchmark == 'wino':
        gap1 = stats1['accuracy_gap']
        gap2 = stats2['accuracy_gap']

        print(f"\nAccuracy Gaps:")
        print(f"  {model1:30s}: {100*gap1:5.1f}pp")
        print(f"  {model2:30s}: {100*gap2:5.1f}pp")
        print(f"  Difference: {100*(gap2 - gap1):+.1f}pp")


def main():
    """Main statistical analysis."""
    print("=" * 70)
    print("STATISTICAL ANALYSIS OF BIAS EVALUATION RESULTS")
    print("=" * 70)

    results_dir = get_results_dir("raw")

    # Load all available results
    results_dict = {}

    for jsonl_file in results_dir.glob("*.jsonl"):
        parts = jsonl_file.stem.split("__")
        if len(parts) != 2:
            continue

        model_id, benchmark = parts

        if model_id not in results_dict:
            results_dict[model_id] = {}

        if benchmark == 'crows_pairs':
            results_dict[model_id]['crows'] = load_results(jsonl_file)
        elif benchmark == 'winobias':
            results_dict[model_id]['wino'] = load_results(jsonl_file)

    print(f"\nFound results for {len(results_dict)} models")

    # Compute statistics for each model
    crows_stats = {}
    wino_stats = {}

    print("\n" + "=" * 70)
    print("PART 1: PER-MODEL STATISTICS")
    print("=" * 70)

    for model_id, data in sorted(results_dict.items()):
        if 'crows' in data:
            stats = analyze_crows_statistics(data['crows'], model_id)
            crows_stats[model_id] = stats

            print(f"\n{'-' * 70}")
            print(f"Model: {model_id} - CrowS-Pairs")
            print(f"{'-' * 70}")
            print(f"Sample size: {stats['total']}")
            print(f"Stereotype preference rate: {100*stats['stereo_rate']:.1f}%")
            print(f"95% CI: [{100*stats['ci_95_lower']:.1f}%, {100*stats['ci_95_upper']:.1f}%]")
            print(f"Test vs 50% baseline:")
            print(f"  χ² = {stats['chi2_statistic']:.2f}, p = {stats['p_value']:.4f} {'***' if stats['significant'] else '(n.s.)'}")
            print(f"Mean bias score: {stats['mean_bias_score']:.4f} ± {stats['std_bias_score']:.4f}")

        if 'wino' in data:
            stats = analyze_winobias_statistics(data['wino'], model_id)
            wino_stats[model_id] = stats

            print(f"\n{'-' * 70}")
            print(f"Model: {model_id} - WinoBias")
            print(f"{'-' * 70}")
            print(f"Sample size: {stats['total']}")
            print(f"Pro-stereotypical accuracy: {100*stats['pro_accuracy']:.1f}%")
            print(f"Anti-stereotypical accuracy: {100*stats['anti_accuracy']:.1f}%")
            print(f"Accuracy gap: {100*stats['accuracy_gap']:.1f}pp")
            print(f"Test for gap significance:")
            print(f"  z = {stats['gap_z_statistic']:.2f}, p = {stats['gap_p_value']:.4f} {'***' if stats['gap_significant'] else '(n.s.)'}")

    # Pairwise comparisons
    if len(crows_stats) >= 2:
        print("\n" + "=" * 70)
        print("PART 2: PAIRWISE MODEL COMPARISONS - CrowS-Pairs")
        print("=" * 70)

        model_ids = sorted(crows_stats.keys())
        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                compare_models(crows_stats[model_ids[i]], crows_stats[model_ids[j]], 'crows')

    if len(wino_stats) >= 2:
        print("\n" + "=" * 70)
        print("PART 3: PAIRWISE MODEL COMPARISONS - WinoBias")
        print("=" * 70)

        model_ids = sorted(wino_stats.keys())
        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                compare_models(wino_stats[model_ids[i]], wino_stats[model_ids[j]], 'wino')

    # Summary
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
Significance Levels:
  *** p < 0.001 (highly significant)
  **  p < 0.01  (very significant)
  *   p < 0.05  (significant)
  n.s. p ≥ 0.05 (not significant)

Effect Sizes (Cohen's h):
  < 0.2  Negligible
  0.2-0.5  Small
  0.5-0.8  Medium
  > 0.8  Large

Confidence Intervals:
  95% CI gives range where true value likely falls
  Non-overlapping CIs suggest significant difference
  CI excluding 50% indicates significant bias

Recommendations for Reporting:
  1. Always report sample sizes
  2. Include confidence intervals
  3. Report both statistical and practical significance
  4. Use effect sizes for interpretation
  5. Consider multiple comparison corrections if doing many tests
    """)

    print("=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
