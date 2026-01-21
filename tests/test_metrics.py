"""
Tests for the metrics calculation functionality.
"""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import (
    calculate_mae,
    calculate_mse,
    calculate_rmse,
    calculate_r2,
    calculate_all_metrics,
    interpret_r2
)


class TestMAE:
    """Test cases for Mean Absolute Error."""

    def test_mae_perfect_prediction(self):
        """Test MAE is 0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        mae = calculate_mae(y_true, y_pred)

        assert mae == 0.0

    def test_mae_simple_case(self):
        """Test MAE calculation with simple case."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])  # All predictions off by 1

        mae = calculate_mae(y_true, y_pred)

        assert mae == 1.0

    def test_mae_mixed_errors(self):
        """Test MAE with mixed positive and negative errors."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.0, 1.0, 4.0, 3.0])  # Errors: -1, 1, -1, 1

        mae = calculate_mae(y_true, y_pred)

        assert mae == 1.0

    def test_mae_non_negative(self):
        """Test that MAE is always non-negative."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)

        mae = calculate_mae(y_true, y_pred)

        assert mae >= 0


class TestMSE:
    """Test cases for Mean Squared Error."""

    def test_mse_perfect_prediction(self):
        """Test MSE is 0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        mse = calculate_mse(y_true, y_pred)

        assert mse == 0.0

    def test_mse_simple_case(self):
        """Test MSE calculation with simple case."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])  # All predictions off by 1

        mse = calculate_mse(y_true, y_pred)

        # MSE = (1^2 + 1^2 + 1^2) / 3 = 1.0
        assert mse == 1.0

    def test_mse_larger_errors_penalized_more(self):
        """Test that MSE penalizes larger errors more."""
        y_true = np.array([0.0, 0.0])

        # Two small errors
        y_pred_small = np.array([1.0, 1.0])
        mse_small = calculate_mse(y_true, y_pred_small)

        # One large error
        y_pred_large = np.array([0.0, 2.0])
        mse_large = calculate_mse(y_true, y_pred_large)

        # Same total error (2), but MSE should be higher for one large error
        # small: (1 + 1) / 2 = 1
        # large: (0 + 4) / 2 = 2
        assert mse_large > mse_small

    def test_mse_non_negative(self):
        """Test that MSE is always non-negative."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)

        mse = calculate_mse(y_true, y_pred)

        assert mse >= 0


class TestRMSE:
    """Test cases for Root Mean Squared Error."""

    def test_rmse_perfect_prediction(self):
        """Test RMSE is 0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        rmse = calculate_rmse(y_true, y_pred)

        assert rmse == 0.0

    def test_rmse_is_sqrt_of_mse(self):
        """Test that RMSE equals sqrt(MSE)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5])

        rmse = calculate_rmse(y_true, y_pred)
        mse = calculate_mse(y_true, y_pred)

        assert np.isclose(rmse, np.sqrt(mse))

    def test_rmse_same_units_as_target(self):
        """Test RMSE with known values to verify units."""
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([2.0, 2.0, 2.0, 2.0])  # All errors = 2

        rmse = calculate_rmse(y_true, y_pred)

        # MSE = 4, RMSE = 2 (same unit as original error)
        assert rmse == 2.0


class TestR2:
    """Test cases for R-squared."""

    def test_r2_perfect_prediction(self):
        """Test R2 is 1.0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        r2 = calculate_r2(y_true, y_pred)

        assert r2 == 1.0

    def test_r2_mean_prediction(self):
        """Test R2 is 0 when predicting the mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_val = y_true.mean()
        y_pred = np.array([mean_val] * 5)

        r2 = calculate_r2(y_true, y_pred)

        assert np.isclose(r2, 0.0)

    def test_r2_can_be_negative(self):
        """Test that R2 can be negative for bad predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([100.0, 100.0, 100.0])  # Very bad predictions

        r2 = calculate_r2(y_true, y_pred)

        assert r2 < 0

    def test_r2_bounded_above_by_one(self):
        """Test that R2 cannot exceed 1.0."""
        np.random.seed(42)
        y_true = np.random.randn(100) * 10 + 50
        y_pred = y_true + np.random.randn(100) * 0.1

        r2 = calculate_r2(y_true, y_pred)

        assert r2 <= 1.0


class TestCalculateAllMetrics:
    """Test cases for calculating all metrics at once."""

    def test_all_metrics_returned(self):
        """Test that all metrics are returned."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        metrics = calculate_all_metrics(y_true, y_pred)

        assert 'MAE' in metrics
        assert 'MSE' in metrics
        assert 'RMSE' in metrics
        assert 'R²' in metrics

    def test_all_metrics_correct_values(self):
        """Test that all metrics have correct values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        metrics = calculate_all_metrics(y_true, y_pred)

        assert metrics['MAE'] == 0.0
        assert metrics['MSE'] == 0.0
        assert metrics['RMSE'] == 0.0
        assert metrics['R²'] == 1.0

    def test_all_metrics_consistent(self):
        """Test that all metrics are consistent with each other."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5])

        metrics = calculate_all_metrics(y_true, y_pred)

        # RMSE should be sqrt of MSE
        assert np.isclose(metrics['RMSE'], np.sqrt(metrics['MSE']))


class TestInterpretR2:
    """Test cases for R2 interpretation."""

    def test_interpret_negative_r2(self):
        """Test interpretation of negative R2."""
        interpretation = interpret_r2(-0.5)
        assert "worse than predicting the mean" in interpretation.lower()

    def test_interpret_weak_fit(self):
        """Test interpretation of weak R2."""
        interpretation = interpret_r2(0.2)
        assert "weak" in interpretation.lower()

    def test_interpret_moderate_fit(self):
        """Test interpretation of moderate R2."""
        interpretation = interpret_r2(0.4)
        assert "moderate" in interpretation.lower()

    def test_interpret_good_fit(self):
        """Test interpretation of good R2."""
        interpretation = interpret_r2(0.6)
        assert "good" in interpretation.lower()

    def test_interpret_strong_fit(self):
        """Test interpretation of strong R2."""
        interpretation = interpret_r2(0.8)
        assert "strong" in interpretation.lower()

    def test_interpret_excellent_fit(self):
        """Test interpretation of excellent R2."""
        interpretation = interpret_r2(0.95)
        assert "excellent" in interpretation.lower() or "overfitting" in interpretation.lower()
