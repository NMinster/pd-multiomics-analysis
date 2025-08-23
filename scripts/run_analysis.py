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
        config = yaml.safe_load(f)
    return config


def run_classification_pipeline(config: dict) -> dict:
    """
    Run the classification pipeline for PD prediction.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
        
    Returns
    -------
    dict
        Results from all models
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting classification pipeline...")
    
    # Initialize components
    data_loader = DataLoader(config)
    model_pipeline = ModelPipeline(config)
    evaluator = ModelEvaluator(config)
    visualizer = Visualizer(config)
    
    # Load data
    logger.info("Loading data...")
    proteomics_df = data_loader.load_proteomics_data()
    proteomics_wide = data_loader.pivot_proteomics_to_wide(proteomics_df)
    rnaseq_data = data_loader.load_rnaseq_data()
    deg_list = data_loader.load_deg_list()
    
    # Merge data
    merged_data = data_loader.merge_multiomics_data(
        rnaseq_data, proteomics_wide, deg_list
    )
    
    # Prepare datasets
    datasets = {}
    
    # Combined dataset
    columns_rna = [col for col in merged_data.columns if col in deg_list]
    proteomics_cols = [
        col for col in merged_data.columns
        if col not in ["PATNO", "pd"] and col not in deg_list
    ]
    datasets["Combined"] = merged_data[
        ["PATNO", "pd"] + columns_rna + proteomics_cols
    ].dropna()
    
    # Proteomics only
    proteomics_cols_only = [
        col for col in merged_data.columns
        if col in proteomics_wide.columns and col not in ["PATNO", "pd"]
    ]
    datasets["Proteomics only"] = merged_data[
        ["PATNO", "pd"] + proteomics_cols_only
    ].dropna()
    
    # RNA-seq only
    datasets["RNA-seq only"] = merged_data[
        ["PATNO", "pd"] + columns_rna
    ].dropna()
    
    # Model training
    results = {}
    roc_curves = {}
    val_auc_dict = {}
    y_val_dict = {}
    
    # Get hyperparameter distributions
    lhs_samples = model_pipeline.generate_lhs_samples()
    param_distributions = model_pipeline.get_param_distributions(lhs_samples)
    base_pipeline = model_pipeline.create_base_pipeline()
    
    group_kfold = GroupKFold(n_splits=config['parameters']['cv']['n_splits'])
    
    for name, df in datasets.items():
        logger.info(f"\n{'='*20} {name} {'='*20}")
        
        # Split data: PD- for training, PP- for validation
        train_test_data = df[df["PATNO"].astype(str).str.startswith("PD-")].copy()
        val_data = df[df["PATNO"].astype(str).str.startswith("PP-")].copy()
        
        X_tt = train_test_data.drop(columns=["PATNO", "pd"])
        y_tt = train_test_data["pd"].astype(int)
        X_val = val_data.drop(columns=["PATNO", "pd"])
        y_val = val_data["pd"].astype(int)
        
        # Hold-out split within PD- group
        X_train_res, X_test, y_train_res, y_test, groups_train, groups_test = train_test_split(
            X_tt, y_tt, train_test_data["PATNO"],
            test_size=config['parameters']['test_size'],
            random_state=config['parameters']['random_state'],
            stratify=y_tt
        )
        
        # Hyperparameter search
        logger.info("Running hyperparameter search...")
        rand_search = RandomizedSearchCV(
            estimator=base_pipeline,
            param_distributions=param_distributions,
            n_iter=50,
            cv=group_kfold,
            scoring=config['parameters']['cv']['scoring'],
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            random_state=config['parameters']['random_state']
        )
        
        rand_search.fit(X_train_res, y_train_res, groups=groups_train)
        
        best_cv_auc = float(rand_search.best_score_)
        logger.info(f"Best CV AUC: {best_cv_auc:.4f}")
        
        # Evaluate on validation set
        best_pipe = rand_search.best_estimator_
        val_results = evaluator.evaluate_model(
            best_pipe, X_val, y_val, dataset_name=f"{name} Validation"
        )
        
        # Store results
        model_name = best_pipe.named_steps['model'].__class__.__name__
        results[name] = {
            'best_cv_auc': best_cv_auc,
            'val_auc': val_results['auc'],
            'auc_ci_lower': val_results['auc_ci'][0],
            'auc_ci_upper': val_results['auc_ci'][1],
            'metric_cis': val_results['metric_cis'],
            'cm': val_results['confusion_matrix'],
            'best_model': model_name,
            'best_params': rand_search.best_params_
        }
        
        roc_curves[name] = val_results['roc_curve']
        val_auc_dict[name] = val_results['y_proba']
        y_val_dict[name] = y_val.values
        
        # SHAP analysis for tree-based models
        shap_path = Path(config['output']['figures_dir']) / f"{name}_shap_beeswarm.eps"
        evaluator.calculate_shap_values(best_pipe, X_val, output_path=str(shap_path))
    
    # Statistical comparison
    logger.info("\n=== Statistical Comparison (Proteomics vs Combined) ===")
    if all(name in y_val_dict for name in ["Combined", "Proteomics only"]):
        assert np.array_equal(y_val_dict["Combined"], y_val_dict["Proteomics only"]), \
            "Mismatch in validation labels"
        
        n_bootstraps = config['parameters']['n_bootstraps']
        rng = np.random.RandomState(config['parameters']['random_state'])
        bootstrap_combined = []
        bootstrap_proteomics = []
        y_ref = y_val_dict["Combined"]
        
        for _ in range(n_bootstraps):
            idxs = rng.randint(0, len(y_ref), len(y_ref))
            if len(np.unique(y_ref[idxs])) < 2:
                continue
            
            from sklearn.metrics import roc_auc_score
            auc_c = roc_auc_score(y_ref[idxs], val_auc_dict["Combined"][idxs])
            auc_p = roc_auc_score(y_ref[idxs], val_auc_dict["Proteomics only"][idxs])
            bootstrap_combined.append(auc_c)
            bootstrap_proteomics.append(auc_p)
        
        stat, p_value = wilcoxon(bootstrap_proteomics, bootstrap_combined)
        logger.info(f"Wilcoxon signed-rank p-value: {p_value:.5f}")
        
        if p_value < 0.05:
            logger.info("Statistically significant: Proteomics outperforms Combined.")
        else:
            logger.info("No significant difference between Proteomics and Combined.")
    
    # Create visualizations
    visualizer.plot_roc_curves(roc_curves, results)
    
    # Save summary
    summary_df = pd.DataFrame({
        k: {
            'best_cv_auc': v['best_cv_auc'],
            'val_auc': v['val_auc'],
            'ci_lower': v['auc_ci_lower'],
            'ci_upper': v['auc_ci_upper'],
            'best_model': v['best_model']
        } for k, v in results.items()
    }).T
    
    output_path = Path(config['output']['results_dir']) / 'model_summary.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path)
    logger.info(f"Summary saved to {output_path}")
    
    return results


def run_psi_analysis(config: dict) -> dict:
    """
    Run PSI (Proteomic Severity Index) analysis.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
        
    Returns
    -------
    dict
        PSI analysis results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting PSI analysis...")
    
    # Initialize components
    data_loader = DataLoader(config)
    psi_analyzer = PSIAnalyzer(config)
    visualizer = Visualizer(config)
    
    # Load proteomics data for longitudinal analysis
    logger.info("Loading proteomics data for PSI...")
    
    # Load all proteomics panels
    panels = ['oncology', 'neurology', 'inflammation', 'cardiometabolic']
    prot_dfs = []
    
    for panel in panels:
        path = config['data']['proteomics'][panel]
        df = pd.read_csv(path)
        df['panel'] = panel
        prot_dfs.append(df)
    
    prot_long = pd.concat(prot_dfs, ignore_index=True)
    prot_long.columns = prot_long.columns.str.strip()
    prot_long = prot_long.rename(columns={"participant_id": "PATNO"})
    prot_long["PATNO"] = prot_long["PATNO"].astype(str)
    
    # Filter for selected features
    selected_features = config['psi']['selected_features']
    prot_long = prot_long[prot_long["UniProt"].isin(selected_features)]
    
    # Pivot to wide format
    wide_long = (
        prot_long.pivot_table(
            index=["PATNO", "visit_month"],
            columns="UniProt",
            values="NPX",
            aggfunc="mean"
        ).reset_index()
    )
    
    # Load UPDRS data
    updrs_data = data_loader.load_updrs_data()
    
    # Prepare baseline data
    baseline_prot = wide_long[wide_long['visit_month'] == 0].drop(columns='visit_month')
    baseline_updrs = updrs_data[updrs_data['visit_month'] == 0][['PATNO', 'total_updrs']]
    
    baseline_data = baseline_prot.merge(baseline_updrs, on='PATNO', how='inner')
    baseline_data = baseline_data.dropna()
    
    # Calculate PSI
    baseline_data, psi_model_info = psi_analyzer.calculate_psi(baseline_data)
    
    # Fit mixed-effects models
    longitudinal_data = psi_analyzer.prepare_longitudinal_data(prot_long, updrs_data)
    mixed_models = psi_analyzer.fit_mixed_effects_models(longitudinal_data, psi_model_info)
    
    # Create visualizations
    visualizer.plot_psi_scatter(baseline_data, psi_model_info)
    visualizer.plot_residuals(baseline_data)
    visualizer.plot_volcano(baseline_data, selected_features)
    
    # Save results
    output_path = Path(config['output']['results_dir']) / 'baseline_PSI_with_CIs.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    baseline_out = baseline_data[
        ["PATNO", "total_updrs", "PSI", "PSI_CI_lower", "PSI_CI_upper"] + selected_features
    ]
    baseline_out.to_csv(output_path, index=False)
    logger.info(f"PSI results saved to {output_path}")
    
    return {
        'baseline_data': baseline_data,
        'psi_model_info': psi_model_info,
        'mixed_models': mixed_models
    }


def main():
    """Main entry point for the analysis."""
    parser = argparse.ArgumentParser(description='PD Multi-omics Analysis Pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--component',
        type=str,
        choices=['all', 'models', 'psi'],
        default='all',
        help='Which component to run'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting PD Multi-omics Analysis Pipeline")
    logger.info(f"Configuration loaded from {args.config}")
    
    # Set random seeds
    np.random.seed(config['parameters']['random_state'])
    
    try:
        if args.component in ['all', 'models']:
            logger.info("Running classification models...")
            classification_results = run_classification_pipeline(config)
            logger.info("Classification pipeline completed successfully")
        
        if args.component in ['all', 'psi']:
            logger.info("Running PSI analysis...")
            psi_results = run_psi_analysis(config)
            logger.info("PSI analysis completed successfully")
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
