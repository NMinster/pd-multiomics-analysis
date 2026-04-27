# Parkinson's Disease Multi-Omics Analysis Pipeline

## Project Structure

```
pd-multiomics-analysis/
│
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup file
├── config.yaml               # Configuration file for paths and parameters
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── data/                 # Data loading and processing
│   │   ├── __init__.py
│   │   ├── loader.py         # Data loading utilities
│   │   └── preprocessor.py   # Data preprocessing functions
│   │
│   ├── models/               # Machine learning models
│   │   ├── __init__.py
│   │   ├── pipeline.py       # ML pipeline definitions
│   │   └── evaluation.py     # Model evaluation utilities
│   │
│   ├── analysis/             # Statistical analysis
│   │   ├── __init__.py
│   │   ├── psi.py            # PSI calculation and analysis
│   │   └── statistics.py     # Statistical utilities
│   │
│   └── visualization/        # Plotting functions
│       ├── __init__.py
│       └── plots.py          # Visualization utilities
│
├── scripts/                  # Executable scripts
│   └── run_analysis.py       # Main analysis script
│
├── notebooks/                # Jupyter notebooks (optional)
│   └── exploratory.ipynb
│
├── tests/                    # Unit tests
│   ├── __init__.py
│   └── test_pipeline.py
│
└── outputs/                  # Output directory (generated)
    ├── figures/
    ├── models/
    └── results/
```

### Description

This repository contains the analysis pipeline for integrating proteomics and RNA-seq data to predict Parkinson's Disease status and assess disease progression using the AMP-PD dataset.

### Key Features

*   **Multi-omics Integration**: Combines proteomics (Olink) and RNA-seq data.
*   **Machine Learning Pipeline**: Implements multiple ML algorithms with Latin Hypercube Sampling for hyperparameter optimization.
*   **Disease Progression Modeling**: Calculates Proteomic Severity Index (PSI) with confidence intervals.
*   **Statistical Analysis**: Mixed-effects models for longitudinal data analysis.
*   **Comprehensive Evaluation**: Bootstrap confidence intervals for all metrics.

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pd-multiomics-analysis.git
cd pd-multiomics-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Configuration

Edit `config.yaml` to specify your data paths and analysis parameters.

### Usage

```bash
# Run the complete analysis pipeline
python scripts/run_analysis.py --config config.yaml

# Run specific components
python scripts/run_analysis.py --config config.yaml --component models
python scripts/run_analysis.py --config config.yaml --component psi
```

### Citation

If you use this code in your research, please cite:

Minster, N., Jafri, S. Plasma proteomics for Parkinson’s disease classification: cross-cohort benchmarking of proteomic, transcriptomic, and multimodal models. npj Parkinsons Dis. (2026). https://doi.org/10.1038/s41531-026-01344-5

### License

MIT

### Contact

nminster@gmu.edu
