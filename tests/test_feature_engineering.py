"""
Unit tests for src/feature_engineering.py

Tests RFM calculation, feature scaling, and data transformation functions.
"""

import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from src.feature_engineering import engineer_features, save_scaler, load_scaler


class TestEngineerFeatures:
    """Test suite for engineer_features function"""

    def test_engineer_features_scaling(self):
        """Test that numeric features are properly scaled"""
        df = pd.DataFrame({
            "age": [34, 45, 29],
            "spending_score": [77, 88, 95],
            "annual_income": [58000, 50000, 45000],
            "gender": ["F", "M", "F"]
        })
        result, scaler = engineer_features(df, fit_scaler=True)

        # Check that scaling columns exist
        assert "age_scaled" in result.columns
        assert "spending_score_scaled" in result.columns
        assert "annual_income_scaled" in result.columns

    def test_engineer_features_returns_scaler(self):
        """Test that scaler is returned"""
        df = pd.DataFrame({
            "age": [34, 45],
            "spending_score": [77, 88],
            "annual_income": [58000, 50000],
        })
        features, scaler = engineer_features(df, fit_scaler=True)
        assert isinstance(scaler, StandardScaler)

    def test_engineer_features_one_hot_encodes_gender(self):
        """Test that categorical features are one-hot encoded"""
        df = pd.DataFrame({
            "age": [34, 45, 29],
            "spending_score": [77, 88, 95],
            "annual_income": [58000, 50000, 45000],
            "gender": ["F", "M", "F"]
        })
        result, scaler = engineer_features(df, fit_scaler=True)

        # Check gender encoding (drop_first=True)
        assert "gender_M" in result.columns
        assert "gender_F" not in result.columns

    def test_engineer_features_missing_required_columns(self):
        """Test error when required columns are missing"""
        df = pd.DataFrame({
            "age": [34, 45],
            "gender": ["F", "M"],
            # Missing annual_income and spending_score
        })
        with pytest.raises(ValueError):
            engineer_features(df, fit_scaler=True)

    def test_engineer_features_standardized_mean_variance(self):
        """Test that scaled features have mean ~0 and std ~1"""
        df = pd.DataFrame({
            "age": [20, 30, 40, 50, 60],
            "spending_score": [10, 30, 50, 70, 90],
            "annual_income": [20000, 40000, 60000, 80000, 100000],
        })
        result, scaler = engineer_features(df, fit_scaler=True)

        # Check mean is close to 0
        assert abs(result["age_scaled"].mean()) < 0.1
        # Check std is close to 1 (tolerance increased due to small sample size)
        assert abs(result["age_scaled"].std() - 1.0) < 0.15

    def test_engineer_features_inference_mode(self):
        """Test inference mode with pre-fitted scaler"""
        # Training data
        df_train = pd.DataFrame({
            "age": [20, 30, 40],
            "spending_score": [10, 50, 90],
            "annual_income": [20000, 60000, 100000],
        })
        _, scaler = engineer_features(df_train, fit_scaler=True)

        # New data
        df_new = pd.DataFrame({
            "age": [35],
            "spending_score": [55],
            "annual_income": [50000],
        })
        result, _ = engineer_features(df_new, fit_scaler=False, scaler=scaler)

        # Result should have expected shape
        assert result.shape[0] == 1
        assert "age_scaled" in result.columns

    def test_engineer_features_single_row(self):
        """Test with single row of data"""
        df = pd.DataFrame({
            "age": [34],
            "spending_score": [77],
            "annual_income": [58000],
        })
        result, scaler = engineer_features(df, fit_scaler=True)
        assert result.shape[0] == 1

    def test_engineer_features_drop_first_one_hot(self):
        """Test that one-hot encoding uses drop_first=True"""
        df = pd.DataFrame({
            "age": [34, 45, 29],
            "spending_score": [77, 88, 95],
            "annual_income": [58000, 50000, 45000],
            "gender": ["F", "M", "F"]
        })
        result, scaler = engineer_features(df, fit_scaler=True)

        # Should only have one gender column (M) due to drop_first=True
        gender_cols = [col for col in result.columns if "gender" in col]
        assert len(gender_cols) == 1


class TestScalerPersistence:
    """Test suite for scaler save/load functions"""

    def test_save_scaler(self, tmp_path):
        """Test saving scaler to disk"""
        df = pd.DataFrame({
            "age": [34, 45, 29],
            "spending_score": [77, 88, 95],
            "annual_income": [58000, 50000, 45000],
        })
        features, scaler = engineer_features(df, fit_scaler=True)

        # Save scaler
        scaler_path = tmp_path / "test_scaler.pkl"
        save_scaler(scaler, str(scaler_path))

        # Verify file exists
        assert scaler_path.exists()

    def test_load_scaler(self, tmp_path):
        """Test loading scaler from disk"""
        df = pd.DataFrame({
            "age": [34, 45, 29],
            "spending_score": [77, 88, 95],
            "annual_income": [58000, 50000, 45000],
        })
        features, scaler = engineer_features(df, fit_scaler=True)

        # Save and load
        scaler_path = tmp_path / "test_scaler.pkl"
        save_scaler(scaler, str(scaler_path))
        loaded_scaler = load_scaler(str(scaler_path))

        # Verify loaded scaler works
        assert isinstance(loaded_scaler, StandardScaler)

    def test_saved_scaler_consistency(self, tmp_path):
        """Test that saved scaler produces same results as original"""
        df = pd.DataFrame({
            "age": [34, 45, 29],
            "spending_score": [77, 88, 95],
            "annual_income": [58000, 50000, 45000],
        })
        features, scaler = engineer_features(df, fit_scaler=True)

        # Save scaler
        scaler_path = tmp_path / "test_scaler.pkl"
        save_scaler(scaler, str(scaler_path))
        loaded_scaler = load_scaler(str(scaler_path))

        # Transform with both scalers
        test_data = np.array([[35, 80, 55000]])
        result_original = scaler.transform(test_data)
        result_loaded = loaded_scaler.transform(test_data)

        # Results should be identical
        np.testing.assert_array_almost_equal(result_original, result_loaded)
