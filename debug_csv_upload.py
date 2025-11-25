#!/usr/bin/env python
"""
Debug script to test CSV upload functionality and RFM processing
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_processing import clean_transactions
from rfm_features import calculate_rfm

def test_csv_processing():
    """Test the complete CSV processing pipeline"""
    
    print("=" * 60)
    print("CSV UPLOAD DEBUGGING SCRIPT")
    print("=" * 60)
    
    # Test with sample data
    csv_path = Path("data/raw/Online Retail.csv")
    
    if not csv_path.exists():
        print(f"❌ Sample data not found at {csv_path}")
        return False
    
    print(f"\n✅ Found sample data at {csv_path}")
    
    # 1. Test CSV loading
    print("\n[1] Testing CSV Loading...")
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✅ Loaded {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Data types:\n{df.dtypes}")
    except Exception as e:
        print(f"   ❌ Error loading CSV: {e}")
        return False
    
    # 2. Test data cleaning
    print("\n[2] Testing Data Cleaning...")
    try:
        df_clean = clean_transactions(df)
        print(f"   ✅ Cleaned {len(df_clean)} rows (removed {len(df) - len(df_clean)} rows)")
        print(f"   Remaining columns: {list(df_clean.columns)}")
    except Exception as e:
        print(f"   ❌ Error cleaning data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Test RFM calculation
    print("\n[3] Testing RFM Calculation...")
    try:
        rfm = calculate_rfm(df_clean)
        print(f"   ✅ Calculated RFM for {len(rfm)} customers")
        print(f"   RFM columns: {list(rfm.columns)}")
        print(f"   Sample RFM data:\n{rfm.head()}")
    except Exception as e:
        print(f"   ❌ Error calculating RFM: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Test model loading
    print("\n[4] Testing Model Loading...")
    model_path = Path("models/kmeans_model.pkl")
    scaler_path = Path("models/scaler.pkl")
    
    try:
        if not model_path.exists():
            print(f"   ❌ Model not found at {model_path}")
            return False
        
        if not scaler_path.exists():
            print(f"   ❌ Scaler not found at {scaler_path}")
            return False
        
        with open(model_path, "rb") as f:
            kmeans = pickle.load(f)
        print(f"   ✅ Loaded K-Means model (clusters: {kmeans.n_clusters})")
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"   ✅ Loaded scaler")
    except Exception as e:
        print(f"   ❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Test feature scaling
    print("\n[5] Testing Feature Scaling...")
    try:
        features = rfm[["Recency", "Frequency", "Monetary_log"]]
        scaled = scaler.transform(features)
        print(f"   ✅ Scaled {scaled.shape[0]} customers with {scaled.shape[1]} features")
    except Exception as e:
        print(f"   ❌ Error scaling features: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Test clustering
    print("\n[6] Testing Clustering...")
    try:
        clusters = kmeans.predict(scaled)
        print(f"   ✅ Assigned clusters: {np.bincount(clusters)}")
    except Exception as e:
        print(f"   ❌ Error clustering: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - CSV UPLOAD SHOULD WORK")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_csv_processing()
    sys.exit(0 if success else 1)
