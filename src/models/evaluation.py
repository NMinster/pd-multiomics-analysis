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
            pb = y_proba[idx]
            
            # Check for class presence
            if len(np.unique(yb)) < 2:
                continue
                
            aucs.append(roc_auc_score(yb, pb))
            pred = (pb >= threshold).astype(int)
            
            # Calculate metrics with zero-division safety
            accs.append(accuracy_score(yb, pred))
            precs.append(precision_score(yb, pred, zero_division=0))
            recs.append(recall_score(yb, pred, zero_division=0))
            f1s.append(f1_score(yb, pred, zero_division=0))
        
        alpha = 1 - self.confidence_level
        cis = {
            'auc_ci': self.bootstrap_ci(np.array(aucs), alpha),
            'acc_ci': self.bootstrap_ci(np.array(accs), alpha),
            'prec_ci': self.bootstrap_ci(np.array(precs), alpha),
            'rec_ci': self.bootstrap_ci(np.array(recs), alpha),
            'f1_ci': self.bootstrap_ci(np.array(f1s), alpha),
        }
        
        return cis
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset_name: str = "Test"
    ) -> Dict[str, any]:
        """
        Comprehensive model evaluation with metrics and confidence intervals.
        
        Parameters
        ----------
        model : sklearn estimator
            Trained model
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        dataset_name : str
            Name of dataset for logging
            
        Returns
        -------
        Dict[str, any]
            Evaluation results
        """
        logger.info(f"Evaluating model on {dataset_name} set...")
        
        # Get predictions
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            # Fallback for non-proba models
            dec = model.decision_function(X_test)
            y_proba = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)
        
        y_pred = (y_proba >= 0.5).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        
        # Get confidence intervals
        metric_cis = self.bootstrap_metric_cis(y_test, y_proba)
        
        # Bootstrap AUC CI
        rng = np.random.RandomState(self.random_state)
        boot_aucs = []
        for _ in range(self.n_bootstraps):
            idxs = rng.randint(0, len(y_test), len(y_test))
            if len(np.unique(y_test[idxs])) < 2:
                continue
            boot_aucs.append(roc_auc_score(y_test[idxs], y_proba[idxs]))
        
        auc_ci = self.bootstrap_ci(np.array(boot_aucs), 1 - self.confidence_level)
        
        results = {
            'auc': auc,
            'auc_ci': auc_ci,
            'confusion_matrix': cm,
            'roc_curve': (fpr, tpr),
            'metric_cis': metric_cis,
            'y_true': y_test,
            'y_proba': y_proba,
            'y_pred': y_pred,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Log results
        logger.info(f"{dataset_name} AUC: {auc:.4f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
        logger.info(f"Accuracy CI: [{metric_cis['acc_ci'][0]:.3f}, {metric_cis['acc_ci'][1]:.3f}]")
        logger.info(f"Precision CI: [{metric_cis['prec_ci'][0]:.3f}, {metric_cis['prec_ci'][1]:.3f}]")
        logger.info(f"Recall CI: [{metric_cis['rec_ci'][0]:.3f}, {metric_cis['rec_ci'][1]:.3f}]")
        logger.info(f"F1 CI: [{metric_cis['f1_ci'][0]:.3f}, {metric_cis['f1_ci'][1]:.3f}]")
        
        return results
    
    def calculate_shap_values(
        self,
        pipeline,
        X_val: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Calculate SHAP values for tree-based models.
        
        Parameters
        ----------
        pipeline : sklearn Pipeline
            Trained pipeline
        X_val : pd.DataFrame
            Validation features
        output_path : Optional[str]
            Path to save SHAP plot
            
        Returns
        -------
        Optional[Dict]
            SHAP analysis results or None if not tree-based
        """
        from ..models.pipeline import ModelPipeline
        
        model = pipeline.named_steps["model"]
        
        if not ModelPipeline.is_tree_based(model):
            logger.info("Model is not tree-based, skipping SHAP analysis")
            return None
        
        logger.info("Calculating SHAP values...")
        
        # Transform features
        X_val_transformed = pipeline[:-1].transform(X_val)
        
        # Get feature names after transformation
        feature_mask = pipeline.named_steps['variance_threshold'].get_support()
        feature_names_filtered = X_val.columns[feature_mask]
        k_best_mask = pipeline.named_steps['feature_selection'].get_support()
        final_feature_names = feature_names_filtered[k_best_mask]
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_transformed)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 for binary
        
        if output_path:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values, 
                X_val_transformed, 
                feature_names=final_feature_names, 
                show=False
            )
            
            # Fix alpha issue
            for collection in plt.gca().collections:
                collection.set_alpha(1.0)
            
            plt.tight_layout()
            plt.savefig(output_path, format='eps', dpi=300)
            plt.close()
            logger.info(f"SHAP plot saved to {output_path}")
        
        return {
            'shap_values': shap_values,
            'feature_names': final_feature_names,
            'expected_value': explainer.expected_value
        }
