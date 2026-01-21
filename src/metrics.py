"""
Metrics module for Linear Regression Studio.
Calculates evaluation metrics for regression models.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MSE value
    """
    return mean_squared_error(y_true, y_pred)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R2 value
    """
    return r2_score(y_true, y_pred)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with all metrics
    """
    return {
        "MAE": calculate_mae(y_true, y_pred),
        "MSE": calculate_mse(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "R²": calculate_r2(y_true, y_pred)
    }


def format_metrics(metrics: Dict[str, float], decimals: int = 4) -> Dict[str, str]:
    """
    Format metrics dictionary for display.

    Args:
        metrics: Dictionary with metric values
        decimals: Number of decimal places

    Returns:
        Dictionary with formatted metric strings
    """
    return {k: f"{v:.{decimals}f}" for k, v in metrics.items()}


def interpret_r2(r2: float) -> str:
    """
    Provide interpretation of R² value.

    Args:
        r2: R-squared value

    Returns:
        Interpretation string
    """
    if r2 < 0:
        return "Model performs worse than predicting the mean"
    elif r2 < 0.3:
        return "Weak fit - model explains little variance"
    elif r2 < 0.5:
        return "Moderate fit - model explains some variance"
    elif r2 < 0.7:
        return "Good fit - model explains substantial variance"
    elif r2 < 0.9:
        return "Strong fit - model explains most variance"
    else:
        return "Excellent fit - be cautious of overfitting"


def get_metrics_summary(
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Combine train and test metrics into a summary.

    Args:
        train_metrics: Metrics on training data
        test_metrics: Metrics on test data

    Returns:
        Combined metrics dictionary
    """
    return {
        "Training": train_metrics,
        "Test": test_metrics,
        "Difference": {
            k: test_metrics[k] - train_metrics[k]
            for k in train_metrics.keys()
        }
    }
