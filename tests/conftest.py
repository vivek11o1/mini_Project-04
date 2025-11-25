"""
Pytest configuration and fixtures for customer-segmentation-analysis

This file provides shared fixtures and configuration for all tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_customer_data():
    """Fixture: Sample customer data for testing"""
    return pd.DataFrame({
        "CustomerID": [12001, 12002, 12003, 12004],
        "Recency": [11, 143, 124, 1],
        "Frequency": [12, 6, 9, 7],
        "Monetary": [9248.85, 3084.42, 2900.17, 4120.29],
    })


@pytest.fixture
def sample_rfm_data():
    """Fixture: Sample RFM data with multiple customers"""
    return pd.DataFrame({
        "CustomerID": [1, 2, 3, 4, 5],
        "Recency": [5, 30, 50, 10, 100],
        "Frequency": [20, 5, 8, 15, 2],
        "Monetary": [1000, 500, 750, 1500, 200],
        "Monetary_log": [6.91, 6.21, 6.63, 7.31, 5.30],
    })


@pytest.fixture
def sample_scaled_features():
    """Fixture: Pre-scaled features for clustering tests"""
    np.random.seed(42)
    return np.array([
        [-0.95, -0.97, -0.65],
        [-0.67, 0.65, 0.99],
        [2.68, -1.30, -1.24],
        [2.20, -0.32, -1.36],
        [0.10, 0.95, 1.25],
    ])


@pytest.fixture
def sample_transaction_data():
    """Fixture: Sample transaction data"""
    return pd.DataFrame({
        "InvoiceNo": ["INV001", "INV002", "INV003", "INV004"],
        "CustomerID": [12001, 12001, 12002, 12002],
        "InvoiceDate": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-20"],
        "Quantity": [5, 3, 10, 2],
        "UnitPrice": [10.0, 15.0, 8.0, 20.0],
    })


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture: Temporary directory for test outputs"""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture(autouse=True)
def reset_matplotlib():
    """Fixture: Reset matplotlib state between tests"""
    import matplotlib.pyplot as plt
    yield
    plt.close("all")


@pytest.fixture
def expected_rfm_columns():
    """Fixture: Expected columns in RFM data"""
    return ["CustomerID", "Recency", "Frequency", "Monetary", "Monetary_log"]


@pytest.fixture
def expected_cluster_columns():
    """Fixture: Expected columns after clustering"""
    return ["CustomerID", "Recency", "Frequency", "Monetary", "Monetary_log", "cluster"]
