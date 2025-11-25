#!/usr/bin/env python
"""
Retrain K-Means model with proper 3 RFM features
"""
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the training data
print("Loading training RFM data...")
rfm_path = Path("data/processed/rfm_segmented.csv")

if not rfm_path.exists():
    print(f"❌ RFM file not found at {rfm_path}")
    print("Creating temporary RFM from raw data...")
    
    # Fall back to loading from raw data and calculating RFM
    from src.data_processing import clean_data
    from src.rfm_features import calculate_rfm
    import sys
    sys.path.insert(0, str(Path.cwd()))
    
    csv_path = Path("data/raw/Online Retail.csv")
    df = pd.read_csv(csv_path)
    
    # Standardize columns to lowercase
    df.columns = df.columns.str.strip().str.lower()
    
    # Rename for consistency with the project
    if "invoicedate" in df.columns:
        df = df.rename(columns={
            "invoicedate": "transaction_date",
            "customerid": "customer_id",
            "quantity": "amount"  # Actually we need unitprice * quantity
        })
        df["amount"] = df["quantity"] * df["unitprice"]
    
    # Clean
    df = df.dropna(subset=["customer_id"])
    df = df[df["quantity"] > 0]
    df = df[~df.get("invoiceno", "").astype(str).str.startswith("C")]
    df = df[df["unitprice"] > 0]
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    
    # Calculate RFM
    rfm = calculate_rfm(df, "transaction_date", "customer_id", "amount")
else:
    print(f"Loading from {rfm_path}")
    rfm = pd.read_csv(rfm_path, index_col=0)

print(f"✅ Loaded RFM for {len(rfm)} customers")
print(f"Columns: {list(rfm.columns)}")
print(f"\nRFM Statistics:\n{rfm.describe()}")

# Extract features
print("\n" + "="*60)
print("Training K-Means Model")
print("="*60)

features = rfm[["Recency", "Frequency", "Monetary"]].copy()
print(f"\nFeatures shape: {features.shape}")
print(f"Features stats:\n{features.describe()}")

# Check if we have a scaler
scaler_path = Path("models/scaler.pkl")
if scaler_path.exists():
    print(f"\n✅ Using existing scaler from {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
else:
    print(f"\nTraining new StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(features)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✅ Saved scaler to {scaler_path}")

# Scale features
scaled_features = scaler.transform(features)
print(f"\nScaled features shape: {scaled_features.shape}")

# Train K-Means with different cluster numbers
print(f"\nTesting different K values...")
inertias = []
silhouette_scores = []
K_range = range(2, 8)

from sklearn.metrics import silhouette_score

for k in K_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_test.fit_predict(scaled_features)
    inertias.append(kmeans_test.inertia_)
    sil_score = silhouette_score(scaled_features, labels)
    silhouette_scores.append(sil_score)
    print(f"  K={k}: Inertia={kmeans_test.inertia_:.2f}, Silhouette={sil_score:.3f}")

# Use K=4 for 4 personas
optimal_k = 4
print(f"\nTraining final K-Means with K={optimal_k}...")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_final.fit(scaled_features)

print(f"✅ Trained K-Means model:")
print(f"  - Clusters: {kmeans_final.n_clusters}")
print(f"  - Features: {kmeans_final.n_features_in_}")
print(f"  - Inertia: {kmeans_final.inertia_:.2f}")
print(f"  - Iterations: {kmeans_final.n_iter_}")

# Save model
model_path = Path("models/kmeans_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(kmeans_final, f)
print(f"✅ Saved K-Means model to {model_path}")

# Verify
print(f"\n" + "="*60)
print("Verification")
print("="*60)

with open(model_path, "rb") as f:
    test_model = pickle.load(f)
print(f"✅ Loaded model from disk:")
print(f"  - Clusters: {test_model.n_clusters}")
print(f"  - Features expected: {test_model.n_features_in_}")

# Test prediction
clusters = test_model.predict(scaled_features)
print(f"\n✅ Cluster distribution:")
print(f"{np.bincount(clusters)}")

print(f"\n✅ MODEL SUCCESSFULLY RETRAINED AND SAVED")
