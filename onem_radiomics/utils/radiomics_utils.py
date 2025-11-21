"""
Radiomics utility functions
"""

import re
import numpy as np
import logging
from typing import Dict, List, Set, Optional, Any

# Try to import pyradiomics
try:
    from radiomics import featureextractor
    RADIOMICS_AVAILABLE = True
except ImportError:
    RADIOMICS_AVAILABLE = False


def setup_radiomics_features(extractor, feature_types: List[str]):
    """
    Setup radiomics feature extraction with specified feature types.
    
    Args:
        extractor: PyRadiomics feature extractor object
        feature_types: List of feature types to enable
    """
    if not RADIOMICS_AVAILABLE:
        raise ImportError("pyradiomics is required. Install with: pip install pyradiomics")
    
    # All available feature classes
    all_features = [
        'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm', 'shape'
    ]
    
    # Disable all features first
    for feature_class in all_features:
        extractor.disableFeatureClass(feature_class)
    
    # Enable specified features
    for feature_type in feature_types:
        if feature_type in all_features:
            try:
                extractor.enableFeatureClass(feature_type)
                logging.info(f"Enabled feature class: {feature_type}")
            except Exception as e:
                logging.warning(f"Failed to enable feature class {feature_type}: {e}")
        else:
            logging.warning(f"Unknown feature class: {feature_type}")


def format_feature_names(feature_name: str) -> str:
    """
    Format radiomics feature names to be more readable and consistent.
    
    Args:
        feature_name: Original feature name from PyRadiomics
        
    Returns:
        Formatted feature name
    """
    # Replace underscores with spaces and capitalize
    formatted = feature_name.replace('_', ' ').title()
    
    # Handle common abbreviations
    abbreviations = {
        'Gldm': 'GLDM',
        'Glcm': 'GLCM', 
        'Glrlm': 'GLRLM',
        'Glszm': 'GLSZM',
        'Ngtdm': 'NGTDM',
        'Id': 'ID',
        'Idn': 'IDN',
        'Idm': 'IDM',
        'Ids': 'IDS',
        'Sze': 'SZ',
        'Lze': 'LZ',
        'Hge': 'HG',
        'Sre': 'SR',
        'Lre': 'LR',
        'Rpc': 'RPC',
        'Ccs': 'CCS',
        'Variance': 'Var',
        'Std': 'StdDev'
    }
    
    for abbr, full in abbreviations.items():
        formatted = re.sub(rf'\b{abbr}\b', full, formatted)
    
    # Replace spaces with underscores for CSV compatibility
    formatted = formatted.replace(' ', '_')
    
    # Remove any remaining special characters except underscores
    formatted = re.sub(r'[^a-zA-Z0-9_]', '', formatted)
    
    return formatted


def validate_features(features_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean extracted features.
    
    Args:
        features_dict: Dictionary of extracted features
        
    Returns:
        Cleaned features dictionary
    """
    cleaned_features = {}
    
    for key, value in features_dict.items():
        # Skip non-numeric features
        if not isinstance(value, (int, float, np.number)):
            continue
        
        # Handle special values
        if np.isnan(value) or np.isinf(value):
            # Replace NaN and Inf with None or 0 depending on context
            cleaned_features[key] = None
        else:
            cleaned_features[key] = float(value)
    
    return cleaned_features


def calculate_feature_statistics(features_matrix: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics for a matrix of features.
    
    Args:
        features_matrix: 2D array where rows are samples and columns are features
        
    Returns:
        Dictionary with statistical measures
    """
    if features_matrix.size == 0:
        return {}
    
    # Filter out NaN and Inf values
    finite_mask = np.isfinite(features_matrix)
    
    if not np.any(finite_mask):
        return {}
    
    finite_features = features_matrix[finite_mask]
    
    stats = {
        'mean': float(np.mean(finite_features)),
        'std': float(np.std(finite_features)),
        'min': float(np.min(finite_features)),
        'max': float(np.max(finite_features)),
        'median': float(np.median(finite_features)),
        'q25': float(np.percentile(finite_features, 25)),
        'q75': float(np.percentile(finite_features, 75)),
        'skewness': float(_calculate_skewness(finite_features)),
        'kurtosis': float(_calculate_kurtosis(finite_features)),
        'n_samples': features_matrix.shape[0],
        'n_features': features_matrix.shape[1],
        'missing_rate': float(1 - np.sum(finite_mask) / features_matrix.size)
    }
    
    return stats


def _calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness of data."""
    if len(data) < 2:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return 0.0
    
    skewness = np.mean(((data - mean) / std) ** 3)
    return skewness


def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate kurtosis of data."""
    if len(data) < 2:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return 0.0
    
    kurtosis = np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
    return kurtosis


def find_constant_features(features_df) -> List[str]:
    """
    Find features with constant values (zero variance).
    
    Args:
        features_df: DataFrame with features
        
    Returns:
        List of feature names with constant values
    """
    constant_features = []
    
    for column in features_df.columns:
        if features_df[column].dtype in ['int64', 'float64']:
            if features_df[column].std() == 0:
                constant_features.append(column)
    
    return constant_features


def find_highly_correlated_features(features_df, threshold: float = 0.95) -> List[tuple]:
    """
    Find pairs of highly correlated features.
    
    Args:
        features_df: DataFrame with features
        threshold: Correlation threshold
        
    Returns:
        List of tuples with (feature1, feature2, correlation)
    """
    # Select only numeric columns
    numeric_df = features_df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return []
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr().abs()
    
    # Find pairs above threshold
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if corr_value >= threshold:
                feature1 = corr_matrix.columns[i]
                feature2 = corr_matrix.columns[j]
                high_corr_pairs.append((feature1, feature2, corr_value))
    
    # Sort by correlation value (descending)
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return high_corr_pairs


def get_feature_importance_ranking(features_df, target_column: str = None) -> Dict[str, float]:
    """
    Get feature importance ranking based on variance or correlation with target.
    
    Args:
        features_df: DataFrame with features
        target_column: Target column for correlation-based ranking
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    numeric_features = features_df.select_dtypes(include=[np.number])
    
    if target_column and target_column in numeric_features.columns:
        # Correlation-based importance
        correlations = numeric_features.corr()[target_column].abs()
        correlations = correlations.drop(target_column)
        return correlations.to_dict()
    else:
        # Variance-based importance
        importance = {}
        for column in numeric_features.columns:
            variance = numeric_features[column].var()
            importance[column] = variance
        
        return importance


def create_feature_selection_report(features_df, target_column: str = None) -> str:
    """
    Create a comprehensive feature selection report.
    
    Args:
        features_df: DataFrame with features
        target_column: Target column for supervised analysis
        
    Returns:
        Report string
    """
    report = []
    report.append("Radiomics Feature Analysis Report")
    report.append("=" * 40)
    report.append("")
    
    # Basic statistics
    numeric_features = features_df.select_dtypes(include=[np.number])
    report.append(f"Total samples: {len(features_df)}")
    report.append(f"Total features: {len(numeric_features.columns)}")
    report.append("")
    
    # Missing values
    missing_counts = numeric_features.isnull().sum()
    missing_rate = missing_counts / len(features_df)
    high_missing = missing_rate[missing_rate > 0.1]
    
    if len(high_missing) > 0:
        report.append("Features with >10% missing values:")
        for feature, rate in high_missing.items():
            report.append(f"  - {feature}: {rate:.2%}")
        report.append("")
    
    # Constant features
    constant_features = find_constant_features(numeric_features)
    if len(constant_features) > 0:
        report.append(f"Constant features (zero variance): {len(constant_features)}")
        for feature in constant_features[:5]:  # Show first 5
            report.append(f"  - {feature}")
        if len(constant_features) > 5:
            report.append(f"  ... and {len(constant_features) - 5} more")
        report.append("")
    
    # Highly correlated features
    high_corr_pairs = find_highly_correlated_features(numeric_features, 0.95)
    if len(high_corr_pairs) > 0:
        report.append(f"Highly correlated feature pairs (r >= 0.95): {len(high_corr_pairs)}")
        for feature1, feature2, corr in high_corr_pairs[:5]:  # Show first 5
            report.append(f"  - {feature1} ↔ {feature2}: {corr:.3f}")
        if len(high_corr_pairs) > 5:
            report.append(f"  ... and {len(high_corr_pairs) - 5} more")
        report.append("")
    
    # Feature importance
    importance = get_feature_importance_ranking(numeric_features, target_column)
    if importance:
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        report.append("Top 10 most important features:")
        for feature, score in sorted_importance[:10]:
            report.append(f"  - {feature}: {score:.4f}")
        report.append("")
    
    return "\n".join(report)