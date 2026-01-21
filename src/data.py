"""
Data loading module for Linear Regression Studio.
Handles sklearn datasets and CSV uploads.
Optimized with caching, TTL, and performance tracking.
"""

import hashlib
import streamlit as st
from typing import Tuple, Optional, Dict, Any

# Lazy imports for heavy libraries
_pd = None
_np = None


def _get_pandas():
    """Lazy import pandas."""
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd


def _get_numpy():
    """Lazy import numpy."""
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def _get_sklearn_datasets():
    """Lazy import sklearn datasets."""
    from sklearn.datasets import fetch_california_housing, load_diabetes
    return fetch_california_housing, load_diabetes


# Cache with long TTL since these datasets don't change
@st.cache_data(ttl=3600, show_spinner=False)
def load_california_housing() -> Tuple["pd.DataFrame", str]:
    """Load California Housing dataset with caching."""
    from src.perf import timer

    with timer("load_california_housing"):
        fetch_california_housing, _ = _get_sklearn_datasets()
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        target_col = "MedHouseVal"

    return df, target_col


@st.cache_data(ttl=3600, show_spinner=False)
def load_diabetes_dataset() -> Tuple["pd.DataFrame", str]:
    """Load Diabetes dataset with caching."""
    from src.perf import timer

    with timer("load_diabetes_dataset"):
        _, load_diabetes = _get_sklearn_datasets()
        data = load_diabetes(as_frame=True)
        df = data.frame
        target_col = "target"

    return df, target_col


def _compute_file_hash(uploaded_file) -> str:
    """Compute hash of uploaded file for cache key stability."""
    content = uploaded_file.getvalue()
    return hashlib.md5(content).hexdigest()[:16]


@st.cache_data(ttl=300, show_spinner=False)
def _parse_csv_cached(file_hash: str, file_content: bytes) -> Tuple[Optional["pd.DataFrame"], Optional[str]]:
    """Parse CSV with caching based on file content hash."""
    from src.perf import timer
    pd = _get_pandas()
    import io

    with timer("parse_csv"):
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            if df.empty:
                return None, "The uploaded CSV file is empty."
            if len(df.columns) < 2:
                return None, "CSV must have at least 2 columns (features + target)."
            return df, None
        except Exception as e:
            return None, f"Error reading CSV: {str(e)}"


def load_csv_file(uploaded_file) -> Tuple[Optional["pd.DataFrame"], Optional[str]]:
    """
    Load a CSV file uploaded by user with caching.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Tuple of (dataframe, error_message)
    """
    if uploaded_file is None:
        return None, "No file uploaded."

    # Use file hash for cache key stability
    file_content = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_content).hexdigest()[:16]

    return _parse_csv_cached(file_hash, file_content)


@st.cache_data(ttl=300, show_spinner=False)
def get_dataset_info_cached(df_hash: str, shape: tuple, columns: tuple) -> Dict[str, Any]:
    """Cached version of dataset info computation."""
    return {
        "shape": shape,
        "rows": shape[0],
        "columns": shape[1],
        "column_names": list(columns),
    }


def get_dataset_info(df: "pd.DataFrame") -> Dict[str, Any]:
    """
    Get summary information about a dataset.
    Optimized to avoid repeated computation.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with dataset information
    """
    from src.perf import timer
    np = _get_numpy()

    with timer("get_dataset_info"):
        # Compute missing values efficiently (vectorized)
        null_counts = df.isnull().sum()

        info = {
            "shape": df.shape,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": null_counts.to_dict(),
            "total_missing": null_counts.sum(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
        }

    return info


def get_numeric_columns(df: "pd.DataFrame") -> list:
    """Get list of numeric columns in DataFrame."""
    np = _get_numpy()
    return df.select_dtypes(include=[np.number]).columns.tolist()


@st.cache_data(ttl=300, show_spinner=False)
def get_missing_value_summary_cached(
    missing_counts: tuple,
    missing_pcts: tuple,
    columns: tuple
) -> "pd.DataFrame":
    """Cached missing value summary computation."""
    pd = _get_pandas()

    summary = pd.DataFrame({
        "Missing Count": list(missing_counts),
        "Missing %": list(missing_pcts)
    }, index=list(columns))
    summary = summary[summary["Missing Count"] > 0]
    return summary.sort_values("Missing Count", ascending=False)


def get_missing_value_summary(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Get a summary of missing values in the dataset.
    Optimized with caching.

    Args:
        df: DataFrame to analyze

    Returns:
        DataFrame with missing value summary
    """
    from src.perf import timer

    with timer("get_missing_value_summary"):
        # Vectorized computation
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        # Use cached version with hashable inputs
        return get_missing_value_summary_cached(
            tuple(missing.values),
            tuple(missing_pct.values),
            tuple(df.columns)
        )


def validate_target_column(df: "pd.DataFrame", target_col: str) -> Tuple[bool, str]:
    """
    Validate that the target column is suitable for regression.

    Args:
        df: DataFrame
        target_col: Name of target column

    Returns:
        Tuple of (is_valid, message)
    """
    np = _get_numpy()

    if target_col not in df.columns:
        return False, f"Column '{target_col}' not found in dataset."

    if not np.issubdtype(df[target_col].dtype, np.number):
        return False, f"Target column '{target_col}' must be numeric for regression."

    if df[target_col].isnull().all():
        return False, f"Target column '{target_col}' contains only missing values."

    return True, "Target column is valid."


def get_feature_target_split(
    df: "pd.DataFrame",
    target_col: str
) -> Tuple["pd.DataFrame", "pd.Series"]:
    """
    Split DataFrame into features and target.

    Args:
        df: DataFrame
        target_col: Name of target column

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# Lazy-loaded dataset registry (loaders called on demand)
AVAILABLE_DATASETS = {
    "California Housing": {
        "loader": load_california_housing,
        "description": "California housing prices from 1990 census. Target: Median house value.",
        "target": "MedHouseVal"
    },
    "Diabetes": {
        "loader": load_diabetes_dataset,
        "description": "Diabetes progression dataset. Target: Disease progression one year after baseline.",
        "target": "target"
    }
}
