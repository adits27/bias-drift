#!/bin/bash
# Regenerate all analyses and visualizations after new evaluations complete

echo "========================================================================"
echo "Regenerating All Analyses"
echo "========================================================================"
echo ""

# Check how many result files we have
RESULT_COUNT=$(ls -1 results/raw/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
echo "Found $RESULT_COUNT evaluation result files"
echo ""

if [ "$RESULT_COUNT" -eq 0 ]; then
    echo "Error: No result files found!"
    exit 1
fi

# List what we have
echo "Results available for:"
ls -1 results/raw/*.jsonl | xargs -n1 basename | sed 's/\.jsonl$//' | sed 's/^/  - /'
echo ""

# Step 1: Comparison Analysis
echo "========================================================================"
echo "Step 1: Running Comparison Analysis"
echo "========================================================================"
python3 scripts/compare_results.py
echo ""

# Step 2: Statistical Analysis
echo "========================================================================"
echo "Step 2: Running Statistical Analysis"
echo "========================================================================"
python3 scripts/statistical_analysis.py
echo ""

# Step 3: Regenerate Visualizations
echo "========================================================================"
echo "Step 3: Regenerating Visualizations"
echo "========================================================================"
python3 scripts/visualize_results.py
echo ""

# Summary
echo "========================================================================"
echo "✓ All Analyses Complete!"
echo "========================================================================"
echo ""
echo "Results:"
echo "  - Comparison:      (printed above)"
echo "  - Statistics:      (printed above)"
echo "  - Visualizations:  results/visualizations/*.png"
echo ""
echo "View visualizations:"
echo "  open results/visualizations/"
echo ""
