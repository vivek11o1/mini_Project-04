"""
RFM (Recency, Frequency, Monetary) Feature Engineering Module

This module provides functions to calculate RFM metrics from transaction data
and scale them for clustering or other ML tasks.
"""

import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def calculate_rfm(
    df: pd.DataFrame,
    transaction_date_col: str = "transaction_date",
    customer_id_col: str = "customer_id",
    amount_col: str = "amount",
) -> pd.DataFrame:
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics from transaction data.

    Args:
        df: Transaction-level DataFrame with dates, customer IDs, and amounts
        transaction_date_col: Name of the date column (should be datetime)
        customer_id_col: Name of the customer ID column
        amount_col: Name of the transaction amount column

    Returns:
        DataFrame with RFM metrics aggregated by customer
    """
    # Ensure date column is datetime
    df = df.copy()
    df[transaction_date_col] = pd.to_datetime(df[transaction_date_col])
    
    # Filter out invalid transactions (missing IDs, negative amounts, etc.)
    df = df.dropna(subset=[customer_id_col, amount_col])
    df = df[df[amount_col] > 0]

    # Reference date (latest date in the data)
    reference_date = df[transaction_date_col].max()

    # Calculate Recency (days since last purchase)
    recency = df.groupby(customer_id_col)[transaction_date_col].max()
    recency = (reference_date - recency).dt.days

    # Calculate Frequency (number of purchases)
    frequency = df.groupby(customer_id_col).size()

    # Calculate Monetary (total spending)
    monetary = df.groupby(customer_id_col)[amount_col].sum()

    # Combine into RFM DataFrame
    rfm = pd.DataFrame(
        {
            customer_id_col: recency.index,
            "Recency": recency.values,
            "Frequency": frequency.values,
            "Monetary": monetary.values,
        }
    )

    # Rename customer_id column to CustomerID if needed
    if customer_id_col != "CustomerID":
        rfm.rename(columns={customer_id_col: "CustomerID"}, inplace=True)

    # Optional: Add Monetary_log for better distribution
    rfm["Monetary_log"] = np.log1p(rfm["Monetary"])

    return rfm


def scale_rfm(
    rfm_df: pd.DataFrame,
    fit_scaler: bool = False,
    scaler: StandardScaler = None,
) -> tuple:
    """
    Scale RFM metrics using StandardScaler for clustering.

    Args:
        rfm_df: DataFrame with RFM columns (Recency, Frequency, Monetary)
        fit_scaler: If True, fit a new scaler. If False, use provided scaler.
        scaler: Pre-fitted StandardScaler (required if fit_scaler=False)

    Returns:
        Tuple of (scaled_df, scaler)
    """
    rfm_columns = ["Recency", "Frequency", "Monetary"]
    missing = [c for c in rfm_columns if c not in rfm_df.columns]
    if missing:
        raise ValueError(f"Missing RFM columns: {missing}")

    features_to_scale = rfm_df[rfm_columns].copy()

    if fit_scaler:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(features_to_scale)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for inference mode")
        scaled_values = scaler.transform(features_to_scale)

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=[f"{col}_scaled" for col in rfm_columns],
        index=rfm_df.index,
    )

    # Preserve customer ID if present
    if "CustomerID" in rfm_df.columns:
        scaled_df.insert(0, "CustomerID", rfm_df["CustomerID"].values)

    return scaled_df, scaler


def save_scaler(scaler: StandardScaler, path: str = "models/scaler.pkl") -> None:
    """Save a fitted StandardScaler to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str = "models/scaler.pkl") -> StandardScaler:
    """Load a fitted StandardScaler from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
