"""
Tests for the pipeline building functionality.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import (
    build_preprocessing_pipeline,
    handle_missing_values,
    prepare_data_for_training
)
from src.train import (
    create_model,
    create_training_pipeline,
    train_model,
    split_data
)


class TestPipelineBuilding:
    """Test cases for pipeline building."""

    def test_preprocessing_pipeline_builds_without_error(self):
        """Test that preprocessing pipeline builds successfully."""
        feature_names = ['feature1', 'feature2', 'feature3']

        pipeline = build_preprocessing_pipeline(feature_names, missing_strategy='mean')

        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'transform')

    def test_preprocessing_pipeline_with_median_strategy(self):
        """Test preprocessing pipeline with median imputation."""
        feature_names = ['a', 'b']

        pipeline = build_preprocessing_pipeline(feature_names, missing_strategy='median')

        assert pipeline is not None

    def test_training_pipeline_linear(self):
        """Test training pipeline creation for linear regression."""
        pipeline = create_training_pipeline(
            model_type='linear',
            alpha=1.0,
            missing_strategy='mean'
        )

        assert isinstance(pipeline, Pipeline)
        assert 'imputer' in pipeline.named_steps
        assert 'scaler' in pipeline.named_steps
        assert 'model' in pipeline.named_steps

    def test_training_pipeline_ridge(self):
        """Test training pipeline creation for ridge regression."""
        pipeline = create_training_pipeline(
            model_type='ridge',
            alpha=0.5,
            missing_strategy='mean'
        )

        assert isinstance(pipeline, Pipeline)
        model = pipeline.named_steps['model']
        assert model.alpha == 0.5

    def test_training_pipeline_lasso(self):
        """Test training pipeline creation for lasso regression."""
        pipeline = create_training_pipeline(
            model_type='lasso',
            alpha=0.1,
            missing_strategy='median'
        )

        assert isinstance(pipeline, Pipeline)
        model = pipeline.named_steps['model']
        assert model.alpha == 0.1

    def test_pipeline_fits_synthetic_data(self):
        """Test that pipeline fits on synthetic data without error."""
        # Create synthetic data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        y = pd.Series(X['feature1'] * 2 + X['feature2'] * 0.5 + np.random.randn(100) * 0.1)

        pipeline = create_training_pipeline('linear')

        # Should not raise any exception
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == len(y)

    def test_pipeline_handles_missing_values(self):
        """Test that pipeline handles missing values correctly."""
        # Create data with missing values
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'feature2': [1.0, np.nan, 3.0, 4.0, 5.0]
        })
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        pipeline = create_training_pipeline('linear', missing_strategy='mean')

        # Should not raise any exception
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == 5
        assert not np.any(np.isnan(predictions))


class TestHandleMissingValues:
    """Test cases for missing value handling."""

    def test_handle_missing_drop(self):
        """Test dropping rows with missing values."""
        df = pd.DataFrame({
            'a': [1.0, 2.0, np.nan, 4.0],
            'b': [1.0, np.nan, 3.0, 4.0]
        })

        result = handle_missing_values(df, strategy='drop')

        assert len(result) == 2  # Only rows 0 and 3 have no missing
        assert result.isnull().sum().sum() == 0

    def test_handle_missing_mean(self):
        """Test mean imputation for missing values."""
        df = pd.DataFrame({
            'a': [1.0, 2.0, np.nan, 4.0],
            'b': [10.0, 20.0, 30.0, 40.0]
        })

        result = handle_missing_values(df, strategy='mean')

        assert len(result) == 4
        assert result.isnull().sum().sum() == 0
        # Mean of [1, 2, 4] is ~2.33
        assert np.isclose(result.loc[2, 'a'], (1 + 2 + 4) / 3)

    def test_handle_missing_median(self):
        """Test median imputation for missing values."""
        df = pd.DataFrame({
            'a': [1.0, 2.0, np.nan, 4.0, 5.0]
        })

        result = handle_missing_values(df, strategy='median')

        assert len(result) == 5
        assert result.isnull().sum().sum() == 0
        # Median of [1, 2, 4, 5] is 3.0
        assert result.loc[2, 'a'] == 3.0


class TestDataSplitting:
    """Test cases for data splitting."""

    def test_split_data_correct_sizes(self):
        """Test that data is split with correct proportions."""
        np.random.seed(42)
        X = pd.DataFrame({'a': range(100), 'b': range(100)})
        y = pd.Series(range(100))

        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_split_data_reproducibility(self):
        """Test that splits are reproducible with same random state."""
        X = pd.DataFrame({'a': range(100)})
        y = pd.Series(range(100))

        X_train1, X_test1, _, _ = split_data(X, y, test_size=0.3, random_state=123)
        X_train2, X_test2, _, _ = split_data(X, y, test_size=0.3, random_state=123)

        assert X_train1.equals(X_train2)
        assert X_test1.equals(X_test2)


class TestModelCreation:
    """Test cases for model creation."""

    def test_create_linear_model(self):
        """Test linear regression model creation."""
        model = create_model('linear')

        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_create_ridge_model(self):
        """Test ridge regression model creation."""
        model = create_model('ridge', alpha=0.5)

        assert model is not None
        assert model.alpha == 0.5

    def test_create_lasso_model(self):
        """Test lasso regression model creation."""
        model = create_model('lasso', alpha=0.1)

        assert model is not None
        assert model.alpha == 0.1

    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError):
            create_model('invalid_type')
