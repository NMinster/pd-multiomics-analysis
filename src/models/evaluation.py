"""
Model evaluation utilities including metrics and confidence intervals.
"""

import logging
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, 
    classification_report, accuracy_score, 
    precision_score, recall_score, f1_score
)
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles model evaluation and metrics calculation."""
    
    def __init__(self, config: Dict):
        """
        Initialize ModelEvaluator with configuration.
        
        Parameters
        ----------
        config : Dict
            Configuration dictionary
        """
        self.config = config
        self.n_bootstraps = config['parameters']['n_bootstraps']
        self.confidence_level = config['parameters']['confidence_level']
        self.random_state = config['parameters']['random_state']
        
    @staticmethod
    def bootstrap_ci(vector: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Calculate nonparametric percentile confidence interval.
        
        Parameters
        ----------
        vector : np.ndarray
            Bootstrap statistics
        alpha : float
            Significance level
            
        Returns
        -------
        Tuple[float, float]
            Lower and upper CI bounds
        """
        lo = np.percentile(vector, 100 * (alpha / 2.0))
        hi = np.percentile(vector, 100 * (1 - alpha / 2.0))
        return float(lo), float(hi)
    
    def bootstrap_metric_cis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
        n_boot: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals for classification metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_proba : np.ndarray
            Predicted probabilities
        threshold : float
            Classification threshold
        n_boot : Optional[int]
            Number of bootstrap iterations
        seed : Optional[int]
            Random seed
            
        Returns
        -------
        Dict[str, Tuple[float, float]]
            Confidence intervals for each metric
        """
        if n_boot is None:
            n_boot = self.n_bootstraps
        if seed is None:
            seed = self.random_state
            
        rng = np.random.RandomState(seed)
        accs, precs, recs, f1s, aucs = [], [], [], [], []
        n = len(y_true)
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            yb = y_true[idx]
            pb = y_
