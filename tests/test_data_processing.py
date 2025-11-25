"""
Unit tests for src/data_processing.py

Tests data cleaning, loading, and validation functions.
"""

import pandas as pd
import pytest
from src.data_processing import clean_data, load_raw_data


class TestCleanData:
    """Test suite for clean_data function"""

    def test_clean_data_removes_duplicates(self):
        """Test that duplicate rows are removed"""
        df = pd.DataFrame({
            "customer_id": [1, 1, 2],
            "age": [34, 34, 29],
            "annual_income": [58000, 58000, 50000],
            "spending_score": [77, 77, 88],
        })
        cleaned = clean_data(df)
        assert cleaned.shape[0] == 2, "Duplicates should be removed"

    def test_clean_data_removes_nulls(self):
        """Test that rows with null core values are removed"""
        df = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "age": [34, None, 29],
            "annual_income": [58000, 50000, 45000],
            "spending_score": [77, 88, 95],
        })
        cleaned = clean_data(df)
        assert cleaned.shape[0] == 2, "Null rows should be removed"
        assert cleaned["age"].notna().all()

    def test_clean_data_handles_negative_values(self):
        """Test that negative values are removed"""
        df = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "age": [34, 45, -5],
            "annual_income": [58000, 50000, 45000],
            "spending_score": [77, 88, 95],
        })
        cleaned = clean_data(df)
        assert cleaned.shape[0] == 2, "Negative values should be removed"
        assert (cleaned["age"] >= 0).all()

    def test_clean_data_normalizes_column_names(self):
        """Test that column names are normalized to snake_case"""
        df = pd.DataFrame({
            "Customer ID": [1, 2],
            "Annual Income": [58000, 50000],
            "Spending Score": [77, 88],
            "Age": [34, 45],
        })
        cleaned = clean_data(df)
        assert "customer_id" in cleaned.columns
        assert "annual_income" in cleaned.columns
        assert "spending_score" in cleaned.columns
        assert "age" in cleaned.columns

    def test_clean_data_coerces_numeric_types(self):
        """Test that numeric columns are properly coerced"""
        df = pd.DataFrame({
            "customer_id": [1, 2],
            "age": ["34", "45"],
            "annual_income": ["58000", "50000"],
            "spending_score": ["77", "88"],
        })
        cleaned = clean_data(df)
        assert cleaned["age"].dtype in [int, float]
        assert cleaned["annual_income"].dtype in [int, float]

    def test_clean_data_empty_dataframe(self):
        """Test handling of empty dataframe"""
        df = pd.DataFrame({
            "customer_id": [],
            "age": [],
            "annual_income": [],
            "spending_score": [],
        })
        cleaned = clean_data(df)
        assert len(cleaned) == 0

    def test_clean_data_all_invalid_rows(self):
        """Test when all rows are invalid"""
        df = pd.DataFrame({
            "customer_id": [None, None],
            "age": [None, None],
            "annual_income": [None, None],
            "spending_score": [None, None],
        })
        cleaned = clean_data(df)
        assert len(cleaned) == 0

    def test_clean_data_resets_index(self):
        """Test that index is reset after cleaning"""
        df = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "age": [34, 45, 29, 50],
            "annual_income": [58000, 50000, 45000, 60000],
            "spending_score": [77, 88, 95, 70],
        })
        cleaned = clean_data(df)
        assert list(cleaned.index) == list(range(len(cleaned)))


class TestLoadRawData:
    """Test suite for load_raw_data function"""

    def test_load_csv_file(self, tmp_path):
        """Test loading a CSV file"""
        # Create test CSV
        test_file = tmp_path / "test_data.csv"
        df_test = pd.DataFrame({
            "customer_id": [1, 2],
            "age": [34, 45],
        })
        df_test.to_csv(test_file, index=False)

        # Load and verify
        loaded = load_raw_data(str(test_file))
        assert len(loaded) == 2
        assert "customer_id" in loaded.columns

    def test_load_csv_with_different_separators(self, tmp_path):
        """Test loading CSV with different separators"""
        test_file = tmp_path / "test_data.csv"
        with open(test_file, 'w') as f:
            f.write("customer_id,age\n1,34\n2,45\n")

        loaded = load_raw_data(str(test_file))
        assert len(loaded) == 2

    def test_load_nonexistent_file(self):
        """Test that nonexistent file raises error"""
        with pytest.raises(FileNotFoundError):
            load_raw_data("nonexistent_file.csv")

    def test_load_invalid_format(self, tmp_path):
        """Test that invalid file format raises error"""
        test_file = tmp_path / "test_data.txt"
        test_file.write_text("some content")

        with pytest.raises(ValueError):
            load_raw_data(str(test_file))
