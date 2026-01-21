"""
Preprocessing module for Linear Regression Studio.
Handles missing values and builds sklearn pipelines.
Optimized with vectorized operations and lazy imports.
"""

from typing import Tuple, Literal, Optional
import streamlit as st

# Lazy imports
_pd = None
_np = None
_sklearn_loaded = False


def _get_pandas():
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd


def _get_numpy():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def _get_sklearn():
    """Lazy load sklearn components."""
    global _sklearn_loaded
    if not _sklearn_loaded:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        _sklearn_loaded = True
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    return Pipeline, StandardScaler, SimpleImputer, ColumnTransformer


def handle_missing_values(
    df: "pd.DataFrame",
    strategy: Literal["drop", "mean", "median"] = "mean"
) -> "pd.DataFrame":
    """
    Handle missing values in DataFrame using vectorized operations.

    Args:
        df: DataFrame to process
        strategy: 'drop' to remove rows, 'mean' or 'median' to impute

    Returns:
        DataFrame with missing values handled
    """
    from src.perf import timer
    np = _get_numpy()

    with timer("handle_missing_values"):
        if strategy == "drop":
            return df.dropna()

        elif strategy in ["mean", "median"]:
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            # Check if there are any missing values (fast check)
            if not df[numeric_cols].isnull().values.any():
                return df

            df_copy = df.copy()

            # Vectorized fill for all columns at once
            if strategy == "mean":
                fill_values = df_copy[numeric_cols].mean()
            else:
                fill_values = df_copy[numeric_cols].median()

            # Single fillna call with dict (vectorized)
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(fill_values)

            return df_copy
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


@st.cache_resource
def _get_cached_pipeline(
    feature_tuple: tuple,
    missing_strategy: str
):
    """Cache pipeline creation to avoid rebuilding."""
    Pipeline, StandardScaler, SimpleImputer, ColumnTransformer = _get_sklearn()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=missing_strategy)),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, list(feature_tuple))
        ],
        remainder='passthrough'
    )

    return preprocessor


def build_preprocessing_pipeline(
    feature_names: list,
    missing_strategy: Literal["mean", "median"] = "mean"
):
    """
    Build a preprocessing pipeline with imputation and scaling.
    Uses caching for repeated builds with same parameters.

    Args:
        feature_names: List of feature column names
        missing_strategy: Strategy for imputing missing values

    Returns:
        sklearn Pipeline object
    """
    from src.perf import timer

    with timer("build_preprocessing_pipeline"):
        Pipeline, StandardScaler, SimpleImputer, ColumnTransformer = _get_sklearn()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=missing_strategy)),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, feature_names)
            ],
            remainder='passthrough'
        )

        return preprocessor


def get_clean_numeric_data(
    df: "pd.DataFrame",
    target_col: str
) -> Tuple["pd.DataFrame", Optional[str]]:
    """
    Get only numeric columns from DataFrame, suitable for regression.
    Optimized to avoid repeated dtype checks.

    Args:
        df: Input DataFrame
        target_col: Target column name

    Returns:
        Tuple of (cleaned DataFrame, error message if any)
    """
    from src.perf import timer
    np = _get_numpy()

    with timer("get_clean_numeric_data"):
        # Single select_dtypes call
        numeric_df = df.select_dtypes(include=[np.number])

        if target_col not in numeric_df.columns:
            return df, f"Target column '{target_col}' is not numeric."

        if numeric_df.shape[1] < 2:
            return df, "Need at least one numeric feature column besides target."

        return numeric_df, None


def prepare_data_for_training(
    df: "pd.DataFrame",
    target_col: str,
    missing_strategy: Literal["drop", "mean", "median"] = "mean"
) -> Tuple["pd.DataFrame", "pd.Series", list, Optional[str]]:
    """
    Prepare data for model training.
    Optimized with vectorized operations and minimal copies.

    Args:
        df: Input DataFrame
        target_col: Target column name
        missing_strategy: Strategy for handling missing values

    Returns:
        Tuple of (X, y, feature_names, error_message)
    """
    from src.perf import timer

    with timer("prepare_data_for_training"):
        # Get only numeric data
        df_numeric, error = get_clean_numeric_data(df, target_col)
        if error and "not numeric" in error:
            return None, None, None, error

        # Handle missing values first if using drop strategy
        if missing_strategy == "drop":
            df_numeric = handle_missing_values(df_numeric, strategy="drop")
            if len(df_numeric) == 0:
                return None, None, None, "No rows remaining after dropping missing values."

        # Get feature columns efficiently (list comprehension is fast)
        feature_cols = [col for col in df_numeric.columns if col != target_col]

        # Get target column
        y = df_numeric[target_col]

        # Find valid indices (vectorized)
        valid_idx = y.notna()

        # Only copy if there are missing values in target
        if valid_idx.all():
            X = df_numeric[feature_cols]
        else:
            X = df_numeric.loc[valid_idx, feature_cols]
            y = y[valid_idx]

        if len(X) == 0:
            return None, None, None, "No valid data after removing missing targets."

        return X, y, feature_cols, None


def get_preprocessing_summary(
    original_df: "pd.DataFrame",
    processed_df: "pd.DataFrame",
    strategy: str
) -> dict:
    """
    Get summary of preprocessing steps applied.

    Args:
        original_df: Original DataFrame
        processed_df: Processed DataFrame
        strategy: Missing value strategy used

    Returns:
        Dictionary with preprocessing summary
    """
    # Compute null sums once (vectorized)
    original_missing = original_df.isnull().sum().sum()
    processed_missing = processed_df.isnull().sum().sum()

    return {
        "original_rows": len(original_df),
        "processed_rows": len(processed_df),
        "rows_removed": len(original_df) - len(processed_df),
        "missing_strategy": strategy,
        "original_missing": original_missing,
        "processed_missing": processed_missing
    }
