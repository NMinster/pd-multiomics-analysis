"""
Visualization utilities for model results and statistical analyses.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


class Visualizer:
    """Handles all visualization tasks."""
    
    def __init__(self, config: Dict):
        """
        Initialize Visualizer with configuration.
        
        Parameters
        ----------
        config : Dict
            Configuration dictionary
        """
        self.config = config
        self.viz_config = config['visualization']
        self.output_dir = Path(config['output']['figures_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_roc_curves(
        self,
        roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
        results: Dict[str, Dict]
    ) -> None:
        """
        Plot ROC curves for all models.
        
        Parameters
        ----------
        roc_curves : Dict
            ROC curve data for each model
        results : Dict
            Model results including AUC values
        """
        plt.figure(figsize=(8, 6))
        
        for name, (fpr, tpr) in roc_curves.items():
            auc = results[name]['val_auc']
            ci_lower = results[name]['auc_ci_lower']
            ci_upper = results[name]['auc_ci_upper']
            
            plt.plot(
                fpr, tpr,
                label=f'{name} (AUC={auc:.3f} [{ci_lower:.3f}-{ci_upper:.3f}])',
                linewidth=2
            )
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / 'roc_curves_comparison.eps'
        plt.savefig(output_path, format='eps', dpi=self.viz_config['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved to {output_path}")
    
    def plot_psi_scatter(
        self,
        baseline_data: pd.DataFrame,
        model_info: Dict
    ) -> None:
        """
        Plot PSI vs UPDRS scatter with prediction bands.
        
        Parameters
        ----------
        baseline_data : pd.DataFrame
            Baseline data with PSI values
        model_info : Dict
            PSI model information
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot
        ax.scatter(
            baseline_data["PSI"],
            baseline_data["total_updrs"],
            alpha=self.viz_config['alpha'],
            s=50,
            edgecolors='black',
            linewidth=0.5,
            label="Baseline participants"
        )
        
        # Fit line with prediction intervals
        psi_vec = baseline_data["PSI"].values.reshape(-1, 1)
        psi_sm = sm.add_constant(psi_vec)
        ols_simple = sm.OLS(baseline_data["total_updrs"].values, psi_sm).fit()
        
        xs = np.linspace(baseline_data["PSI"].min(), baseline_data["PSI"].max(), 200)
        Xpred = sm.add_constant(xs)
        pred = ols_simple.get_prediction(Xpred).summary_frame(alpha=0.05)
        
        ax.plot(xs, pred["mean"], 'r-', label="OLS fit", linewidth=2)
        ax.fill_between(
            xs, pred["mean_ci_lower"], pred["mean_ci_upper"],
            alpha=0.2, color='red', label="95% CI (mean)"
        )
        ax.fill_between(
            xs, pred["obs_ci_lower"], pred["obs_ci_upper"],
            alpha=0.1, color='gray', label="95% Prediction interval"
        )
        
        ax.set_xlabel("Proteomic Severity Index (PSI)", fontsize=12)
        ax.set_ylabel("Total UPDRS Score", fontsize=12)
        ax.set_title(
            f"PSI vs UPDRS at Baseline (CV R² = {model_info['mean_r2']:.2f})",
            fontsize=14
        )
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'psi_updrs_scatter.eps'
        plt.savefig(output_path, format='eps', dpi=self.viz_config['dpi'])
        plt.close()
        
        logger.info(f"PSI scatter plot saved to {output_path}")
    
    def plot_residuals(
        self,
        baseline_data: pd.DataFrame
    ) -> None:
        """
        Plot residual diagnostics for PSI model.
        
        Parameters
        ----------
        baseline_data : pd.DataFrame
            Baseline data with PSI values
        """
        # Refit simple model for residuals
        psi_vec = baseline_data["PSI"].values.reshape(-1, 1)
        psi_sm = sm.add_constant(psi_vec)
        ols_simple = sm.OLS(baseline_data["total_updrs"].values, psi_sm).fit()
        
        fitted = ols_simple.fittedvalues
        resid = ols_simple.resid
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residuals vs Fitted
        axes[0].scatter(fitted, resid, alpha=self.viz_config['alpha'])
        axes[0].axhline(0, linestyle="--", color='red', alpha=0.7)
        axes[0].set_xlabel("Fitted Values")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Fitted")
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        sm.qqplot(resid, line='45', ax=axes[1])
        axes[1].set_title("Q-Q Plot")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'psi_residuals.eps'
        plt.savefig(output_path, format='eps', dpi=self.viz_config['dpi'])
        plt.close()
        
        logger.info(f"Residual plots saved to {output_path}")
    
    def plot_volcano(
        self,
        baseline_data: pd.DataFrame,
        selected_features: List[str]
    ) -> None:
        """
        Create volcano plot for protein-UPDRS correlations.
        
        Parameters
        ----------
        baseline_data : pd.DataFrame
            Baseline data
        selected_features : List[str]
            List of protein features
        """
        results_corr = []
        
        for prot in selected_features:
            if prot in baseline_data.columns:
                rho, pval = spearmanr(baseline_data[prot], baseline_data["total_updrs"])
                results_corr.append({"feature": prot, "rho": rho, "pval": pval})
        
        df_corr = pd.DataFrame(results_corr)
        df_corr["qval"] = multipletests(df_corr["pval"], method="fdr_bh")[1]
        df_corr["–log10(q)"] = -np.log10(df_corr["qval"].clip(lower=1e-10))
        
        plt.figure(figsize=(8, 6))
        
        # Significance line
        plt.axhline(y=-np.log10(0.05), color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        plt.axvline(x=0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        
        # Separate positive and negative correlations
        pos = df_corr[df_corr["rho"] > 0]
        neg = df_corr[df_corr["rho"] < 0]
        
        plt.scatter(
            pos["rho"], pos["–log10(q)"],
            color="#1f77b4", alpha=0.8, s=50,
            edgecolors='black', linewidth=0.5
        )
        plt.scatter(
            neg["rho"], neg["–log10(q)"],
            color="#d62728", alpha=0.8, s=50,
            edgecolors='black', linewidth=0.5
        )
        
        # Annotate top hits
        top_hits = df_corr.nsmallest(5, "qval")
        for _, row in top_hits.iterrows():
            plt.text(
                row["rho"], row["–log10(q)"],
                row["feature"], fontsize=8,
                ha="center", va="bottom"
            )
        
        plt.xlabel("Spearman Correlation (ρ)", fontsize=12)
        plt.ylabel("–log₁₀(FDR-adjusted p-value)", fontsize=12)
        plt.title("Protein–UPDRS Correlations", fontsize=14)
        
        # Custom legend
        pos_handle = mlines.Line2D(
            [], [], color='#1f77b4', marker='o',
            linestyle='None', label='Positive correlation'
        )
        neg_handle = mlines.Line2D(
            [], [], color='#d62728', marker='o',
            linestyle='None', label='Negative correlation'
        )
        sig_handle = mlines.Line2D(
            [], [], color='red', linestyle='--',
            label='FDR = 0.05'
        )
        
        plt.legend(
            handles=[pos_handle, neg_handle, sig_handle],
            loc="best", frameon=True
        )
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / 'volcano_plot.eps'
        plt.savefig(output_path, format='eps', dpi=self.viz_config['dpi'])
        plt.close()
        
        logger.info(f"Volcano plot saved to {output_path}")
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        title: str = "Feature Importance"
    ) -> None:
        """
        Plot feature importance bar chart.
        
        Parameters
        ----------
        importance_df : pd.DataFrame
            DataFrame with feature names and importance scores
        title : str
            Plot title
        """
        plt.figure(figsize=(10, 6))
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=True).tail(20)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.tight_layout()
        
        output_path = self.output_dir / 'feature_importance.eps'
        plt.savefig(output_path, format='eps', dpi=self.viz_config['dpi'])
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_path}")
