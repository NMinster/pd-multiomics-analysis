#!/usr/bin/env python
"""
Main analysis script for PD multi-omics pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold, RandomizedSearchCV
from scipy.stats import wilcoxon
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.models.pipeline import ModelPipeline
from src.models.evaluation import ModelEvaluator
from src.analysis.psi import PSIAnalyzer
from src.visualization.plots import Visualizer

warnings.filterwarnings('ignore')


def setup_logging(config: dict) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler('analysis.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml
