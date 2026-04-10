"""Validation utilities for cell matching."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_matches(
    matches: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """
    Validate and filter matches based on quality criteria.
    
    Parameters
    ----------
    matches : pd.DataFrame
        Initial matches from greedy_match_cells
    df1, df2 : pd.DataFrame
        Feature dataframes for round 1 and 2
    config : MatchingConfig
        Matching configuration with validation parameters
        
    Returns
    -------
    pd.DataFrame
        Validated matches with confidence scores
    """
    if matches.empty:
        return matches
    
    validated = matches.copy()
    
    # Add confidence score (inverse of distance, normalized to [0, 1])
    max_dist = validated['distance'].max()
    if max_dist > 0:
        validated['confidence'] = 1.0 - (validated['distance'] / (max_dist + 1.0))
    else:
        validated['confidence'] = 1.0
    
    # Filter by minimum confidence
    if config.min_confidence > 0:
        validated = validated[validated['confidence'] >= config.min_confidence]
    
    # Feature similarity validation: check relative differences
    if config.max_feature_diff > 0 and config.max_feature_diff < 1.0:
        for col in config.feature_columns:
            col1 = f"{col}_1"
            col2 = f"{col}_2"
            if col1 in validated.columns and col2 in validated.columns:
                vals1 = validated[col1].values
                vals2 = validated[col2].values
                denom = np.maximum(np.maximum(np.abs(vals1), np.abs(vals2)), 1e-6)
                relative_diff = np.abs(vals1 - vals2) / denom
                validated = validated[relative_diff < config.max_feature_diff]
    
    return validated.reset_index(drop=True)


def compute_match_quality_stats(matches: pd.DataFrame) -> dict:
    """
    Compute quality statistics for matches.
    
    Parameters
    ----------
    matches : pd.DataFrame
        Matches with distance and optional confidence columns
        
    Returns
    -------
    dict
        Statistics including min/max/mean distance, confidence, etc.
    """
    if matches.empty:
        return {
            'n_matches': 0,
            'distance_min': np.nan,
            'distance_max': np.nan,
            'distance_mean': np.nan,
            'distance_std': np.nan,
        }
    
    stats = {
        'n_matches': len(matches),
        'distance_min': matches['distance'].min(),
        'distance_max': matches['distance'].max(),
        'distance_mean': matches['distance'].mean(),
        'distance_std': matches['distance'].std(),
    }
    
    if 'confidence' in matches.columns:
        stats['confidence_min'] = matches['confidence'].min()
        stats['confidence_mean'] = matches['confidence'].mean()
    
    return stats
