"""
Machine learning pipeline definitions and model configurations.
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
from scipy.stats import qmc
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from imblearn.pipeline import Pipeline as IMBPipeline
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class ModelPipeline:
    """Handles creation and configuration of ML pipelines."""
    
    def __init__(self, config: Dict):
        """
        Initialize ModelPipeline with configuration.
        
        Parameters
        ----------
        config : Dict
            Configuration dictionary
        """
        self.config = config
        self.random_state = config['parameters']['random_state']
        self.lhs_config = config['parameters']['lhs']
        self.model_config = config['parameters']['models']
        
    def create_base_pipeline(self) -> IMBPipeline:
        """
        Create base preprocessing and modeling pipeline.
        
        Returns
        -------
        IMBPipeline
            Base pipeline with preprocessing steps
        """
        variance_threshold = self.config['parameters']['feature_selection']['variance_threshold']
        k_best = self.config['parameters']['feature_selection']['k_best']
        
        pipeline = IMBPipeline([
            ('variance_threshold', VarianceThreshold(threshold=variance_threshold)),
            ('feature_selection', SelectKBest(score_func=f_classif, k=k_best)),
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier())  # Placeholder
        ])
        
        return pipeline
    
    def generate_lhs_samples(self) -> np.ndarray:
        """
        Generate Latin Hypercube Samples for hyperparameter search.
        
        Returns
        -------
        np.ndarray
            LHS samples array
        """
        n_iter = self.lhs_config['n_iter']
        d = self.lhs_config['dimensions']
        
        sampler = qmc.LatinHypercube(d=d, seed=self.random_state)
        lhs_samples = sampler.random(n_iter)
        
        logger.info(f"Generated {n_iter} LHS samples with {d} dimensions")
        return lhs_samples
    
    def get_param_distributions(self, lhs_samples: np.ndarray) -> List[Dict]:
        """
        Generate parameter distributions for RandomizedSearchCV.
        
        Parameters
        ----------
        lhs_samples : np.ndarray
            Latin Hypercube Samples
            
        Returns
        -------
        List[Dict]
            List of parameter distributions for different models
        """
        rf_config = self.model_config['random_forest']
        xgb_config = self.model_config['xgboost']
        gb_config = self.model_config['gradient_boosting']
        svm_config = self.model_config['svm']
        lr_config = self.model_config['logistic_regression']
        et_config = self.model_config['extra_trees']
        
        param_distributions = [
            # Random Forest (LHS-based)
            {
                'model': [RandomForestClassifier(random_state=self.random_state)],
                'model__n_estimators': [
                    int(r[0] * (rf_config['n_estimators_range'][1] - rf_config['n_estimators_range'][0]) 
                        + rf_config['n_estimators_range'][0]) 
                    for r in lhs_samples
                ],
                'model__max_depth': [
                    None if r[1] < 1/11 else 
                    int((r[1] - 1/11)/(1 - 1/11) * (rf_config['max_depth_range'][1] - 1) + 1) 
                    for r in lhs_samples
                ],
                'model__min_samples_split': [
                    int(r[2] * (rf_config['min_samples_split_range'][1] - rf_config['min_samples_split_range'][0]) 
                        + rf_config['min_samples_split_range'][0]) 
                    for r in lhs_samples
                ],
                'model__min_samples_leaf': [
                    int(r[3] * (rf_config['min_samples_leaf_range'][1] - rf_config['min_samples_leaf_range'][0]) 
                        + rf_config['min_samples_leaf_range'][0]) 
                    for r in lhs_samples
                ],
                'model__class_weight': rf_config['class_weight'],
                'feature_selection__k': [
                    int(r[5] * (51 - 30) + 30) for r in lhs_samples
                ],
            },
            # XGBoost
            {
                'model': [XGBClassifier(
                    use_label_encoder=False, 
                    eval_metric='logloss', 
                    random_state=self.random_state
                )],
                'model__n_estimators': xgb_config['n_estimators'],
                'model__max_depth': xgb_config['max_depth'],
                'model__learning_rate': xgb_config['learning_rate'],
                'model__subsample': xgb_config['subsample'],
                'feature_selection__k': self.config['parameters']['feature_selection']['k_range'],
            },
            # Gradient Boosting
            {
                'model': [GradientBoostingClassifier(random_state=self.random_state)],
                'model__n_estimators': gb_config['n_estimators'],
                'model__learning_rate': gb_config['learning_rate'],
                'model__max_depth': gb_config['max_depth'],
                'feature_selection__k': self.config['parameters']['feature_selection']['k_range'],
            },
            # SVM
            {
                'model': [SVC(probability=True, random_state=self.random_state)],
                'model__C': svm_config['C'],
                'model__kernel': svm_config['kernel'],
                'model__gamma': svm_config['gamma'],
                'feature_selection__k': self.config['parameters']['feature_selection']['k_range'],
            },
            # Logistic Regression
            {
                'model': [LogisticRegression(
                    max_iter=lr_config['max_iter'], 
                    random_state=self.random_state
                )],
                'model__C': lr_config['C'],
                'model__penalty': lr_config['penalty'],
                'model__solver': lr_config['solver'],
                'feature_selection__k': self.config['parameters']['feature_selection']['k_range'],
            },
            # Extra Trees
            {
                'model': [ExtraTreesClassifier(random_state=self.random_state)],
                'model__n_estimators': et_config['n_estimators'],
                'model__max_depth': [
                    None if d is None else d for d in et_config['max_depth']
                ],
                'feature_selection__k': self.config['parameters']['feature_selection']['k_range'],
            },
        ]
        
        return param_distributions
    
    @staticmethod
    def is_tree_based(estimator) -> bool:
        """
        Check if model is tree-based for SHAP TreeExplainer support.
        
        Parameters
        ----------
        estimator : sklearn estimator
            Model to check
            
        Returns
        -------
        bool
            True if model is tree-based
        """
        return isinstance(estimator, (RandomForestClassifier, ExtraTreesClassifier)) or \
               estimator.__class__.__name__.lower().startswith('xgb')
