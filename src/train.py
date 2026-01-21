"""
Model training module for Linear Regression Studio.
Handles training of Linear, Ridge, and Lasso regression models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Literal, Dict, Any, Optional
import streamlit as st


def create_model(
    model_type: Literal["linear", "ridge", "lasso"],
    alpha: float = 1.0
) -> Any:
    """
    Create a regression model.

    Args:
        model_type: Type of model ('linear', 'ridge', 'lasso')
        alpha: Regularization parameter for Ridge/Lasso

    Returns:
        sklearn model object
    """
    if model_type == "linear":
        return LinearRegression()
    elif model_type == "ridge":
        return Ridge(alpha=alpha)
    elif model_type == "lasso":
        return Lasso(alpha=alpha, max_iter=10000)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_training_pipeline(
    model_type: Literal["linear", "ridge", "lasso"],
    alpha: float = 1.0,
    missing_strategy: Literal["mean", "median"] = "mean"
) -> Pipeline:
    """
    Create a complete training pipeline with preprocessing and model.

    Args:
        model_type: Type of model
        alpha: Regularization parameter
        missing_strategy: Strategy for imputing missing values

    Returns:
        sklearn Pipeline object
    """
    model = create_model(model_type, alpha)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=missing_strategy)),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    return pipeline


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets.

    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: Literal["linear", "ridge", "lasso"],
    alpha: float = 1.0,
    missing_strategy: Literal["mean", "median"] = "mean"
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a regression model.

    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model
        alpha: Regularization parameter
        missing_strategy: Strategy for missing values

    Returns:
        Tuple of (trained pipeline, training info dict)
    """
    pipeline = create_training_pipeline(
        model_type=model_type,
        alpha=alpha,
        missing_strategy=missing_strategy
    )

    pipeline.fit(X_train, y_train)

    # Get model from pipeline
    model = pipeline.named_steps['model']

    # Get coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_

    training_info = {
        "model_type": model_type,
        "alpha": alpha if model_type != "linear" else None,
        "missing_strategy": missing_strategy,
        "n_features": X_train.shape[1],
        "n_samples": X_train.shape[0],
        "coefficients": coefficients,
        "intercept": intercept,
        "feature_names": X_train.columns.tolist()
    }

    return pipeline, training_info


def predict(
    pipeline: Pipeline,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Make predictions using trained pipeline.

    Args:
        pipeline: Trained pipeline
        X: Features to predict on

    Returns:
        Predictions array
    """
    return pipeline.predict(X)


def get_model_equation(
    training_info: Dict[str, Any],
    top_n: int = 10
) -> str:
    """
    Get a readable representation of the model equation.

    Args:
        training_info: Dictionary with model training info
        top_n: Number of top coefficients to show

    Returns:
        String representation of model equation
    """
    coefficients = training_info['coefficients']
    intercept = training_info['intercept']
    feature_names = training_info['feature_names']

    # Get top N coefficients by absolute value
    coef_abs = np.abs(coefficients)
    top_indices = np.argsort(coef_abs)[::-1][:top_n]

    equation_parts = [f"{intercept:.4f}"]

    for idx in top_indices:
        coef = coefficients[idx]
        name = feature_names[idx]
        sign = "+" if coef >= 0 else "-"
        equation_parts.append(f"{sign} {abs(coef):.4f} × {name}")

    equation = "y = " + " ".join(equation_parts)

    if len(coefficients) > top_n:
        equation += f" + ... ({len(coefficients) - top_n} more terms)"

    return equation


def get_coefficient_df(training_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Get coefficients as a sorted DataFrame.

    Args:
        training_info: Dictionary with model training info

    Returns:
        DataFrame with feature names and coefficients
    """
    df = pd.DataFrame({
        'Feature': training_info['feature_names'],
        'Coefficient': training_info['coefficients']
    })
    df['Abs_Coefficient'] = np.abs(df['Coefficient'])
    df = df.sort_values('Abs_Coefficient', ascending=False)
    return df.drop('Abs_Coefficient', axis=1).reset_index(drop=True)


def get_model_summary(training_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of the trained model.

    Args:
        training_info: Dictionary with model training info

    Returns:
        Dictionary with model summary
    """
    return {
        "Model Type": training_info['model_type'].capitalize(),
        "Alpha": training_info['alpha'] if training_info['alpha'] else "N/A",
        "Number of Features": training_info['n_features'],
        "Training Samples": training_info['n_samples'],
        "Intercept": round(training_info['intercept'], 4),
        "Missing Value Strategy": training_info['missing_strategy']
    }
