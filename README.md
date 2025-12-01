# Bias Drift: Measuring Bias Evolution in Language Models

## Overview

This repository implements a research framework for measuring **bias drift** in language models over time. The project evaluates how biases in language models change across different model versions and releases using established bias benchmarks.

### Benchmarks

- **CrowS-Pairs**: Measures stereotypical biases across nine categories (race, gender, religion, etc.)
- **WinoBias**: Evaluates gender bias in coreference resolution tasks

### Key Features

- Multi-model evaluation across different versions and checkpoints
- Standardized bias scoring metrics
- Change point detection for identifying significant bias shifts
- Statistical analysis and visualization tools
- Reproducible experimental pipeline

## Project Structure

```
bias-drift/
├─ data/                    # Benchmark datasets
│   ├─ crows_pairs/
│   └─ winobias/
├─ configs/                 # YAML configuration files
│   ├─ models.yaml         # Model definitions and versions
│   └─ benchmarks.yaml     # Benchmark configurations
├─ src/                     # Source code
│   ├─ datasets/           # Dataset loading and preprocessing
│   ├─ models/             # Model loading and inference
│   ├─ scoring/            # Bias scoring implementations
│   ├─ bdi/                # Bias Drift Index calculations
│   ├─ stats/              # Statistical analysis tools
│   └─ utils/              # Utility functions (config, paths, random seed)
├─ scripts/                 # Execution scripts
├─ results/                 # Experiment outputs
│   ├─ raw/                # Raw scores and predictions
│   ├─ aggregated/         # Aggregated results
│   └─ figures/            # Plots and visualizations
└─ notebooks/              # Jupyter notebooks for analysis
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster inference)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd bias-drift
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify Installation

```python
# Test imports
from src.utils.paths import get_project_root
from src.utils.config import load_models_config
from src.utils.random_utils import set_global_seed

# Set reproducibility
set_global_seed(42)

# Load configs
models_config = load_models_config()
print("Setup complete!")
```

## Usage

### Configuration

Edit `configs/models.yaml` to specify which models and versions to evaluate.
Edit `configs/benchmarks.yaml` to configure benchmark settings.

### Running Experiments

```bash
# Example: Evaluate all models on CrowS-Pairs
python scripts/run_evaluation.py --benchmark crows_pairs

# Example: Compare specific model versions
python scripts/compare_versions.py --model gpt2 --versions v1,v2,v3
```

### Analysis

Use Jupyter notebooks in the `notebooks/` directory for exploratory analysis and visualization:

```bash
jupyter notebook notebooks/
```

## Reproducibility

All experiments use deterministic random seeds set via `src.utils.random_utils.set_global_seed()`. Default seed is 42, but can be configured in benchmark configs.

## License

[Specify your license here]

## Citation

If you use this codebase in your research, please cite:

```bibtex
[Add citation when available]
```

## Contact

[Add contact information]
