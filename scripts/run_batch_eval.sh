#!/bin/bash
# Batch evaluation script for running multiple models efficiently
#
# Usage:
#   ./scripts/run_batch_eval.sh [quick|medium|full]
#
# Modes:
#   quick  - Test 2-3 key models (DistilBERT, RoBERTa) - ~10 mins
#   medium - Test 5-6 models including GPT variants - ~30 mins
#   full   - Test all models (expensive, GPT-Neo 1.3B is slow) - ~2 hours

MODE="${1:-quick}"

echo "========================================================================"
echo "Bias Drift Batch Evaluation"
echo "========================================================================"
echo "Mode: $MODE"
echo ""

case $MODE in
  quick)
    echo "Quick mode: Testing DistilBERT and RoBERTa on both benchmarks"
    echo "Estimated time: ~10 minutes"
    echo ""

    # DistilBERT
    python3 scripts/run_eval.py --models distilbert-base-uncased --benchmarks crows_pairs --device cpu --verbose
    python3 scripts/run_eval.py --models distilbert-base-uncased --benchmarks winobias --device cpu --verbose

    # RoBERTa
    python3 scripts/run_eval.py --models roberta-base --benchmarks crows_pairs --device cpu --verbose
    python3 scripts/run_eval.py --models roberta-base --benchmarks winobias --device cpu --verbose
    ;;

  medium)
    echo "Medium mode: Testing 6 models (3 masked LM + 3 generative)"
    echo "Estimated time: ~30-40 minutes"
    echo ""

    # Masked LMs
    python3 scripts/run_eval.py --models distilbert-base-uncased,roberta-base,bert-large-uncased --benchmarks crows_pairs,winobias --device cpu --verbose

    # Generative models
    python3 scripts/run_eval.py --models gpt2-medium,gpt2-large,gpt-neo-125m --benchmarks crows_pairs,winobias --device cpu --verbose
    ;;

  full)
    echo "Full mode: ALL models (WARNING: GPT-Neo 1.3B is very slow on CPU)"
    echo "Estimated time: ~2 hours"
    echo ""
    read -p "Continue with full evaluation? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Cancelled."
      exit 0
    fi

    # Run all models
    python3 scripts/run_eval.py --skip-api --benchmarks crows_pairs,winobias --device cpu --verbose
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [quick|medium|full]"
    exit 1
    ;;
esac

echo ""
echo "========================================================================"
echo "Batch evaluation complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Analyze results:     python3 scripts/compare_results.py"
echo "  2. Generate plots:      python3 scripts/visualize_results.py"
echo "  3. View visualizations: open results/visualizations/"
echo ""
