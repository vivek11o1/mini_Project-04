#!/usr/bin/env python
"""
Test script to simulate Streamlit CSV upload and identify the exact error
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import timedelta

def clean_data(df):
    """Clean and preprocess customer transaction data (from app.py)"""
    df = df.copy()
    
    # Remove rows with missing CustomerID
    df = df.dropna(subset=["CustomerID"])
    
    # Remove negative quantities (returns/cancellations)
    df = df[df["Quantity"] > 0]
    
    # Remove cancelled invoices (start with 'C')
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    
    # Remove non-positive prices
    df = df[df["UnitPrice"] > 0]
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Convert date column
    if "invoicedate" in df.columns:
        df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors='coerce')
        df = df.dropna(subset=["invoicedate"])
    
    # Convert CustomerID to int
    df["customerid"] = df["customerid"].astype(int)
    
    return df


def generate_rfm(df_clean):
    """Generate RFM metrics (from app.py)"""
    if "invoicedate" not in df_clean.columns:
        return None
    
    # Calculate total amount per transaction
    df_clean = df_clean.copy()
    df_clean["amount"] = df_clean["quantity"] * df_clean["unitprice"]
    
    # Reference date
    reference_date = df_clean["invoicedate"].max()
    
    # Recency: days since last purchase
    recency = df_clean.groupby("customerid")["invoicedate"].max()
    recency = (reference_date - recency).dt.days
    
    # Frequency: number of purchases
    frequency = df_clean.groupby("customerid").size()
    
    # Monetary: total spending
    monetary = df_clean.groupby("customerid")["amount"].sum()
    
    # Combine into RFM
    rfm = pd.DataFrame({
        "CustomerID": recency.index,
        "Recency": recency.values,
        "Frequency": frequency.values,
        "Monetary": monetary.values
    })
    
    rfm.set_index("CustomerID", inplace=True)
    rfm["Monetary_log"] = np.log1p(rfm["Monetary"])
    
    return rfm


def main():
    print("=" * 70)
    print("STREAMLIT CSV UPLOAD TEST SIMULATION")
    print("=" * 70)
    
    # Load sample data
    csv_path = Path("data/raw/Online Retail.csv")
    print(f"\n[1] Loading CSV from {csv_path}")
    
    if not csv_path.exists():
        print(f"    ❌ File not found!")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        print(f"    ✅ Loaded {len(df)} rows")
        print(f"    Columns: {list(df.columns)}")
        print(f"    First row:\n{df.iloc[0]}")
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False
    
    # Try cleaning
    print(f"\n[2] Cleaning data...")
    try:
        df_clean = clean_data(df)
        print(f"    ✅ Cleaned {len(df_clean)} rows")
        print(f"    Columns after cleaning: {list(df_clean.columns)}")
        print(f"    First row:\n{df_clean.iloc[0]}")
    except Exception as e:
        print(f"    ❌ Error during cleaning: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Try RFM calculation
    print(f"\n[3] Calculating RFM...")
    try:
        rfm = generate_rfm(df_clean)
        if rfm is None:
            print(f"    ❌ RFM calculation returned None")
            return False
        
        print(f"    ✅ Calculated RFM for {len(rfm)} customers")
        print(f"    RFM columns: {list(rfm.columns)}")
        print(f"    Sample RFM:\n{rfm.head()}")
        print(f"    RFM stats:\n{rfm.describe()}")
    except Exception as e:
        print(f"    ❌ Error during RFM calculation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Try loading models
    print(f"\n[4] Loading models...")
    try:
        model_path = Path("models/kmeans_model.pkl")
        scaler_path = Path("models/scaler.pkl")
        
        with open(model_path, "rb") as f:
            kmeans = pickle.load(f)
        print(f"    ✅ Loaded K-Means model (n_clusters={kmeans.n_clusters})")
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"    ✅ Loaded scaler")
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False
    
    # Try scaling features
    print(f"\n[5] Scaling features...")
    try:
        features = rfm[["Recency", "Frequency", "Monetary"]]
        print(f"    Features shape: {features.shape}")
        print(f"    Features:\n{features.head()}")
        
        scaled = scaler.transform(features)
        print(f"    ✅ Scaled features shape: {scaled.shape}")
    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Try clustering
    print(f"\n[6] Clustering...")
    try:
        clusters = kmeans.predict(scaled)
        print(f"    ✅ Cluster assignments shape: {clusters.shape}")
        print(f"    Cluster distribution: {np.bincount(clusters)}")
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - STREAMLIT UPLOAD SHOULD WORK")
    print("=" * 70)
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
