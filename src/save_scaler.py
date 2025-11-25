#!/usr/bin/env python3
"""
Utility script to save the RFM scaler from training data.

This script loads the RFM raw data, fits the StandardScaler, and saves it
for use in production segmentation.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rfm_features import save_scaler


def create_scaler_from_training_data(
    raw_rfm_path: str = "data/processed/rfm_raw.csv",
    scaler_output_path: str = "models/scaler.pkl",
) -> StandardScaler:
    """
    Load training RFM data and create/save the StandardScaler.

    Args:
        raw_rfm_path: Path to the raw RFM data used for training
        scaler_output_path: Path where to save the fitted scaler

    Returns:
        The fitted StandardScaler object
    """
    print("\n" + "=" * 80)
    print("CREATING & SAVING RFM SCALER")
    print("=" * 80)

    # Load raw RFM data
    print(f"\nLoading training RFM data from {raw_rfm_path}...")
    rfm_raw = pd.read_csv(raw_rfm_path, index_col=0)
    print(f"  ✓ Loaded {len(rfm_raw)} customers with columns: {list(rfm_raw.columns)}")

    # Create scaler from training data
    print("\nFitting StandardScaler on training RFM features...")
    rfm_columns = ["Recency", "Frequency", "Monetary"]
    missing_cols = [c for c in rfm_columns if c not in rfm_raw.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    scaler = StandardScaler()
    scaler.fit(rfm_raw[rfm_columns])

    # Display scaler statistics
    print("  ✓ Scaler fitted successfully")
    print(f"\n  Scaler statistics:")
    for i, col in enumerate(rfm_columns):
        print(f"    {col}:")
        print(f"      - Mean: {scaler.mean_[i]:.4f}")
        print(f"      - Std Dev: {scaler.scale_[i]:.4f}")

    # Save scaler
    print(f"\nSaving scaler to {scaler_output_path}...")
    save_scaler(scaler, scaler_output_path)
    print(f"  ✓ Scaler saved successfully")

    print("\n" + "=" * 80)
    return scaler


if __name__ == "__main__":
    try:
        scaler = create_scaler_from_training_data()
        print("\n✓ Scaler creation and saving complete!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
