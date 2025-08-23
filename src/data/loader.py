"""
Data loading utilities for proteomics, RNA-seq, and clinical data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and initial processing of multi-omics data."""
    
    def __init__(self, config: Dict):
        """
        Initialize DataLoader with configuration.
        
        Parameters
        ----------
        config : Dict
            Configuration dictionary containing data paths
        """
        self.config = config
        self.data_paths = config['data']
        
    def load_proteomics_data(self) -> pd.DataFrame:
        """
        Load and combine proteomics data from multiple panels.
        
        Returns
        -------
        pd.DataFrame
            Combined proteomics data in long format
        """
        logger.info("Loading proteomics data from multiple panels...")
        
        panels = ['oncology', 'neurology', 'inflammation', 'cardiometabolic']
        dfs = []
        
        for panel in panels:
            path = self.data_paths['proteomics'][panel]
            logger.debug(f"Loading {panel} panel from {path}")
            df = pd.read_csv(path)
            dfs.append(df)
        
        df_combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(df_combined)} proteomics measurements")
        
        return df_combined
    
    def pivot_proteomics_to_wide(self, df_proteomics: pd.DataFrame) -> pd.DataFrame:
        """
        Convert proteomics data from long to wide format.
        
        Parameters
        ----------
        df_proteomics : pd.DataFrame
            Proteomics data in long format
            
        Returns
        -------
        pd.DataFrame
            Proteomics data in wide format with participants as rows
        """
        logger.info("Pivoting proteomics data to wide format...")
        
        df_wide = (
            df_proteomics
            .pivot_table(
                index="participant_id",
                columns="UniProt",
                values="NPX",
                aggfunc="mean"
            )
            .reset_index()
        )
        
        df_wide.rename(columns={"participant_id": "PATNO"}, inplace=True)
        logger.info(f"Pivoted to shape: {df_wide.shape}")
        
        return df_wide
    
    def load_rnaseq_data(self) -> pd.DataFrame:
        """
        Load RNA-seq data.
        
        Returns
        -------
        pd.DataFrame
            RNA-seq expression data
        """
        path = self.data_paths['rnaseq']['aligned_data']
        logger.info(f"Loading RNA-seq data from {path}")
        
        df = pd.read_csv(path)
        
        if "participant_id" in df.columns:
            df.rename(columns={"participant_id": "PATNO"}, inplace=True)
        
        logger.info(f"Loaded RNA-seq data with shape: {df.shape}")
        return df
    
    def load_deg_list(self) -> List[str]:
        """
        Load list of differentially expressed genes.
        
        Returns
        -------
        List[str]
            List of significant DEG identifiers
        """
        path = self.data_paths['rnaseq']['deg_analysis']
        logger.info(f"Loading DEG analysis from {path}")
        
        df_deg = pd.read_csv(path)
        
        # Apply filters from config
        filters = self.config['parameters']['deg_filters']
        adj_p_val = filters['adj_p_val']
        log_fc = filters['log_fc']
        
        filtered_df = df_deg[
            (df_deg["adj.P.Val"] < adj_p_val) & 
            (df_deg["logFC"] > log_fc)
        ]
        
        deg_list = filtered_df["gene_id"].tolist()
        logger.info(f"Found {len(deg_list)} significant DEGs")
        
        return deg_list
    
    def load_updrs_data(self) -> pd.DataFrame:
        """
        Load and merge UPDRS clinical assessment data.
        
        Returns
        -------
        pd.DataFrame
            Combined UPDRS scores with total score calculated
        """
        logger.info("Loading UPDRS clinical data...")
        
        paths_up = {
            "part_i": self.data_paths['clinical']['updrs_part_i'],
            "part_ii": self.data_paths['clinical']['updrs_part_ii'],
            "part_iii": self.data_paths['clinical']['updrs_part_iii'],
            "part_iv": self.data_paths['clinical']['updrs_part_iv']
        }
        
        dfs_up = {}
        
        for part, path in paths_up.items():
            logger.debug(f"Loading UPDRS {part}")
            
            # Dynamic column detection
            header = pd.read_csv(path, nrows=0).columns.str.strip()
            score_cols = [
                c for c in header 
                if c.lower().startswith("mds_updrs") and c.lower().endswith("summary_score")
            ]
            
            if len(score_cols) != 1:
                raise ValueError(
                    f"Cannot uniquely identify summary_score column in {path}: {score_cols}"
                )
            
            score_col = score_cols[0]
            df = pd.read_csv(path, usecols=["participant_id", "visit_month", score_col])
            df.columns = df.columns.str.strip()
            df = df.rename(columns={"participant_id": "PATNO", score_col: part})
            df["PATNO"] = df["PATNO"].astype(str)
            df["visit_month"] = df["visit_month"].astype(int)
            dfs_up[part] = df
        
        # Merge all parts
        up = dfs_up["part_i"]
        for part in ["part_ii", "part_iii", "part_iv"]:
            before = up.shape[0]
            up = up.merge(dfs_up[part], on=["PATNO", "visit_month"], how="inner")
            after = up.shape[0]
            logger.debug(f"After merging {part}: {before} → {after}")
        
        # Calculate total score
        up["total_updrs"] = up[["part_i", "part_ii", "part_iii", "part_iv"]].sum(axis=1)
        
        logger.info(f"Loaded UPDRS data with {len(up)} observations")
        return up
    
    def merge_multiomics_data(
        self, 
        rnaseq_data: pd.DataFrame,
        proteomics_wide: pd.DataFrame,
        deg_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Merge RNA-seq and proteomics data.
        
        Parameters
        ----------
        rnaseq_data : pd.DataFrame
            RNA-seq expression data
        proteomics_wide : pd.DataFrame
            Proteomics data in wide format
        deg_list : Optional[List[str]]
            List of DEGs to include
            
        Returns
        -------
        pd.DataFrame
            Merged multi-omics dataset
        """
        logger.info("Merging multi-omics data...")
        
        merged = pd.merge(
            rnaseq_data,
            proteomics_wide,
            on="PATNO",
            how="inner"
        )
        
        logger.info(f"Merged data shape: {merged.shape}")
        
        return merged
