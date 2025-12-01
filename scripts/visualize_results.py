#!/usr/bin/env python3
"""
Create visualizations for bias evaluation results.

Generates publication-ready charts and plots for analysis.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.paths import get_results_dir, ensure_dir_exists

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_results(jsonl_path):
    """Load results from JSONL file."""
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def plot_model_comparison(results_dict, output_dir):
    """Create bar chart comparing models on stereotype preference."""
    models = []
    crows_rates = []
    wino_rates = []

    for model_id in sorted(results_dict.keys()):
        if 'crows' in results_dict[model_id]:
            crows_data = results_dict[model_id]['crows']
            stereo_rate = 100 * sum(1 for r in crows_data if r['bias_direction'] == 1) / len(crows_data)

            models.append(model_id)
            crows_rates.append(stereo_rate)

        if 'wino' in results_dict[model_id]:
            wino_data = results_dict[model_id]['wino']
            stereo_rate = 100 * sum(1 for r in wino_data if r['bias_direction'] == 1) / len(wino_data)

            if model_id not in models:
                models.append(model_id)
                crows_rates.append(0)
            wino_rates.append(stereo_rate)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, crows_rates, width, label='CrowS-Pairs', alpha=0.8)
    bars2 = ax.bar(x + width/2, wino_rates, width, label='WinoBias', alpha=0.8)

    # Add baseline line at 50%
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random Baseline (50%)', alpha=0.7)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stereotype Preference Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: Stereotype Preference Rates', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_bias_by_dimension(results_dict, output_dir):
    """Create heatmap showing bias across social dimensions for each model."""
    # Collect data
    models = []
    axes_data = defaultdict(lambda: defaultdict(float))

    for model_id in sorted(results_dict.keys()):
        if 'crows' not in results_dict[model_id]:
            continue

        models.append(model_id)
        crows_data = results_dict[model_id]['crows']

        # Group by bias axis
        axis_stats = defaultdict(lambda: {'stereo': 0, 'total': 0})
        for r in crows_data:
            axis = r['bias_axis']
            axis_stats[axis]['total'] += 1
            if r['bias_direction'] == 1:
                axis_stats[axis]['stereo'] += 1

        # Calculate rates
        for axis, stats in axis_stats.items():
            rate = 100 * stats['stereo'] / stats['total']
            axes_data[model_id][axis] = rate

    # Create matrix for heatmap
    all_axes = sorted(set(axis for model_data in axes_data.values() for axis in model_data.keys()))
    matrix = []
    for model_id in models:
        row = [axes_data[model_id].get(axis, 0) for axis in all_axes]
        matrix.append(row)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, max(6, len(models) * 0.5)))

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',
        center=50,
        vmin=0,
        vmax=100,
        xticklabels=all_axes,
        yticklabels=models,
        cbar_kws={'label': 'Stereotype Preference Rate (%)'},
        ax=ax
    )

    ax.set_title('Bias Across Social Dimensions', fontsize=14, fontweight='bold')
    ax.set_xlabel('Bias Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()

    output_path = output_dir / 'bias_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_accuracy_gap(results_dict, output_dir):
    """Plot accuracy gap for WinoBias (pro vs anti stereotypical)."""
    models = []
    pro_acc = []
    anti_acc = []
    gaps = []

    for model_id in sorted(results_dict.keys()):
        if 'wino' not in results_dict[model_id]:
            continue

        wino_data = results_dict[model_id]['wino']

        # Separate pro and anti
        pro_examples = [r for r in wino_data if r.get('metadata', {}).get('subtype') == 'pro_stereotype']
        anti_examples = [r for r in wino_data if r.get('metadata', {}).get('subtype') == 'anti_stereotype']

        if not pro_examples or not anti_examples:
            continue

        pro_correct = sum(1 for r in pro_examples if r.get('metadata', {}).get('is_correct', False))
        anti_correct = sum(1 for r in anti_examples if r.get('metadata', {}).get('is_correct', False))

        pro_rate = 100 * pro_correct / len(pro_examples)
        anti_rate = 100 * anti_correct / len(anti_examples)
        gap = pro_rate - anti_rate

        models.append(model_id)
        pro_acc.append(pro_rate)
        anti_acc.append(anti_rate)
        gaps.append(gap)

    # Create grouped bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Pro vs Anti accuracy
    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, pro_acc, width, label='Pro-stereotypical', alpha=0.8, color='salmon')
    ax1.bar(x + width/2, anti_acc, width, label='Anti-stereotypical', alpha=0.8, color='skyblue')

    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('WinoBias: Pro vs Anti-Stereotypical Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 100)

    # Add value labels
    for i, (p, a) in enumerate(zip(pro_acc, anti_acc)):
        ax1.text(i - width/2, p + 1, f'{p:.1f}%', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, a + 1, f'{a:.1f}%', ha='center', va='bottom', fontsize=9)

    # Right plot: Accuracy gap
    colors = ['red' if g > 0 else 'green' for g in gaps]
    bars = ax2.bar(x, gaps, alpha=0.8, color=colors)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy Gap (pp)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Gap: Pro-Stereotypical - Anti-Stereotypical', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')

    # Add value labels
    for i, (bar, gap) in enumerate(zip(bars, gaps)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -1),
                f'{gap:.1f}pp',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'accuracy_gap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_temporal_drift(results_dict, model_metadata, output_dir):
    """Plot bias drift over time (release dates)."""
    # Group by model family
    families = defaultdict(list)

    for model_id in results_dict.keys():
        if 'crows' not in results_dict[model_id]:
            continue

        # Extract family and version from model_id
        if 'bert' in model_id.lower():
            family = 'BERT'
        elif 'gpt2' in model_id.lower() or 'gpt-2' in model_id.lower():
            family = 'GPT-2'
        elif 'roberta' in model_id.lower():
            family = 'RoBERTa'
        elif 'gpt-neo' in model_id.lower():
            family = 'GPT-Neo'
        else:
            family = 'Other'

        crows_data = results_dict[model_id]['crows']
        stereo_rate = 100 * sum(1 for r in crows_data if r['bias_direction'] == 1) / len(crows_data)

        # Get version number
        version = model_metadata.get(model_id, {}).get('version', 0)

        families[family].append({
            'model_id': model_id,
            'version': version,
            'stereo_rate': stereo_rate
        })

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'BERT': 'blue', 'GPT-2': 'orange', 'RoBERTa': 'green', 'GPT-Neo': 'red', 'Other': 'gray'}

    for family, data in families.items():
        if not data:
            continue

        # Sort by version
        data = sorted(data, key=lambda x: x['version'])

        versions = [d['version'] for d in data]
        rates = [d['stereo_rate'] for d in data]

        ax.plot(versions, rates, marker='o', label=family, linewidth=2,
               markersize=8, color=colors.get(family, 'gray'))

        # Add labels
        for d in data:
            ax.annotate(d['model_id'],
                       (d['version'], d['stereo_rate']),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center',
                       fontsize=8,
                       rotation=45)

    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random Baseline', alpha=0.7)

    ax.set_xlabel('Model Version', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stereotype Preference Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Temporal Bias Drift: CrowS-Pairs Stereotype Preference Over Model Versions',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(40, 80)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'temporal_drift.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_distribution(results_dict, output_dir):
    """Plot distribution of bias scores."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    row = 0
    col = 0

    for model_id in sorted(results_dict.keys()):
        if 'crows' not in results_dict[model_id]:
            continue

        crows_data = results_dict[model_id]['crows']
        bias_scores = [r['bias_score'] for r in crows_data]

        ax = axes[row, col]

        # Histogram
        ax.hist(bias_scores, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral (0)')
        ax.axvline(x=np.mean(bias_scores), color='green', linestyle='--', linewidth=2,
                  label=f'Mean ({np.mean(bias_scores):.2f})')

        ax.set_xlabel('Bias Score', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{model_id} - CrowS-Pairs Bias Score Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        col += 1
        if col >= 2:
            col = 0
            row += 1

        if row >= 2:
            break

    # Hide unused subplots
    for i in range(row * 2 + col, 4):
        r = i // 2
        c = i % 2
        if r < 2:
            axes[r, c].axis('off')

    plt.tight_layout()
    output_path = output_dir / 'bias_score_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("Generating Bias Evaluation Visualizations")
    print("=" * 70)

    results_dir = get_results_dir("raw")
    output_dir = get_results_dir("visualizations")
    ensure_dir_exists(output_dir)

    # Load all available results
    results_dict = {}

    # Check for results files
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
    for model_id, data in results_dict.items():
        benchmarks = ', '.join(data.keys())
        print(f"  - {model_id}: {benchmarks}")

    if not results_dict:
        print("\nNo results found! Run evaluations first.")
        return 1

    print(f"\nOutput directory: {output_dir}")
    print("\nGenerating visualizations...")
    print("-" * 70)

    # Create model metadata (from configs)
    model_metadata = {}
    try:
        from src.utils.config import load_models_config
        config = load_models_config()
        for model in config.get('models', []):
            model_metadata[model['id']] = model
    except:
        pass

    # Generate plots
    plot_model_comparison(results_dict, output_dir)
    plot_bias_by_dimension(results_dict, output_dir)
    plot_accuracy_gap(results_dict, output_dir)
    plot_temporal_drift(results_dict, model_metadata, output_dir)
    plot_distribution(results_dict, output_dir)

    print("-" * 70)
    print(f"\n✓ All visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    for viz_file in sorted(output_dir.glob("*.png")):
        print(f"  - {viz_file.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
