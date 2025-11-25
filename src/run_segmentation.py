#!/usr/bin/env python3
"""
Production-Ready Customer Segmentation Script

This script processes new customer transaction data and assigns them to
predefined customer segments (personas) using a pre-trained K-Means model
and RFM features.

Usage:
    python run_segmentation.py --input data/raw/new_transactions.csv --output data/processed/new_customers_segmented.csv
    python run_segmentation.py  # Uses default paths

Workflow:
    1. Load new transaction data
    2. Calculate RFM metrics for each customer
    3. Scale RFM features using the pre-trained scaler
    4. Assign segments using the pre-trained K-Means model
    5. Classify into personas (CHAMPIONS, LOYAL, AT-RISK, GROWTH)
    6. Save segmented output with actionable recommendations
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import load_raw_data
from src.rfm_features import calculate_rfm, scale_rfm, load_scaler


# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/kmeans_model.pkl"
DEFAULT_INPUT = "data/raw/new_transactions.csv"
DEFAULT_OUTPUT = "data/processed/new_customers_segmented.csv"

# Persona definitions based on RFM characteristics
PERSONA_MAP = {
    0: {
        "name": "CHAMPIONS",
        "description": "Best customers - frequent, recent, high-value",
        "priority": "HIGHEST",
        "goal": "Retain and maximize lifetime value",
        "tactics": [
            "VIP loyalty programs with exclusive perks",
            "Early access to new products",
            "Personal account manager",
            "Referral incentives",
        ],
        "channels": "Email, SMS, Account manager",
        "kpi": "Retention rate > 95%, NPS > 80",
    },
    1: {
        "name": "LOYAL CUSTOMERS",
        "description": "Regular buyers with consistent engagement",
        "priority": "HIGH",
        "goal": "Increase purchase frequency and average order value",
        "tactics": [
            "Personalized product recommendations",
            "Loyalty rewards program",
            "Regular engagement campaigns",
            "Birthday and anniversary offers",
        ],
        "channels": "Email, SMS, Push notifications",
        "kpi": "Frequency +15%, AOV growth +10%",
    },
    2: {
        "name": "AT-RISK / DORMANT",
        "description": "Previously valuable but inactive - need reactivation",
        "priority": "HIGH",
        "goal": "Reactivate and restore engagement",
        "tactics": [
            "Win-back campaigns with special discounts",
            "Reason for inactivity survey",
            "Limited-time exclusive offers",
            "Product upgrade suggestions",
        ],
        "channels": "Email, SMS, Display ads",
        "kpi": "Reactivation rate > 20%",
    },
    3: {
        "name": "GROWTH OPPORTUNITY",
        "description": "New or occasional buyers - nurture for growth",
        "priority": "MEDIUM",
        "goal": "Develop into loyal customers",
        "tactics": [
            "Welcome series and onboarding",
            "Educational content",
            "Cross-sell and upsell recommendations",
            "Referral program incentives",
        ],
        "channels": "Email, SMS, Social media",
        "kpi": "Conversion to repeat > 40%, Frequency +25%",
    },
}


# ============================================================================
# SEGMENTATION FUNCTIONS
# ============================================================================


def load_model(model_path: str = MODEL_PATH) -> object:
    """Load the pre-trained K-Means model from disk."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def assign_persona(cluster: int) -> dict:
    """
    Assign persona information based on cluster ID.

    Args:
        cluster: Cluster ID from K-Means

    Returns:
        Dictionary with persona details
    """
    return PERSONA_MAP.get(cluster, {})


def segment_customers(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    scaler_path: str = SCALER_PATH,
    model_path: str = MODEL_PATH,
) -> dict:
    """
    Complete segmentation pipeline for new customer data.

    Args:
        input_path: Path to raw transaction data CSV
        output_path: Path to save segmented output CSV
        scaler_path: Path to saved StandardScaler
        model_path: Path to saved K-Means model

    Returns:
        Dictionary with segmentation results and statistics
    """
    print("\n" + "=" * 80)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 80)

    # Step 1: Load and clean data
    print("\n[1/5] Loading and cleaning data...")
    try:
        raw_data = load_raw_data(input_path)
        print(f"  ✓ Loaded {len(raw_data)} transactions from {input_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Step 2: Calculate RFM metrics
    print("\n[2/5] Calculating RFM metrics...")
    # For Online Retail data: InvoiceDate, CustomerID, UnitPrice * Quantity
    # Normalize column names first
    transaction_data = raw_data.copy()
    transaction_data.columns = transaction_data.columns.str.strip().str.lower().str.replace(" ", "_")
    
    if "invoicedate" in transaction_data.columns:
        transaction_data.rename(columns={"invoicedate": "transaction_date"}, inplace=True)
    if "customerid" in transaction_data.columns:
        transaction_data.rename(columns={"customerid": "customer_id"}, inplace=True)
    
    # Calculate amount (quantity * unit_price)
    if "quantity" in transaction_data.columns and "unitprice" in transaction_data.columns:
        transaction_data["amount"] = transaction_data["quantity"] * transaction_data["unitprice"]
    
    rfm_data = calculate_rfm(
        transaction_data,
        transaction_date_col="transaction_date",
        customer_id_col="customer_id",
        amount_col="amount",
    )
    print(f"  ✓ Calculated RFM for {len(rfm_data)} unique customers")
    print(f"  ✓ RFM stats:")
    print(f"    - Recency: {rfm_data['Recency'].mean():.1f} days (avg)")
    print(f"    - Frequency: {rfm_data['Frequency'].mean():.1f} orders (avg)")
    print(f"    - Monetary: £{rfm_data['Monetary'].mean():.2f} (avg)")

    # Step 3: Scale RFM features
    print("\n[3/5] Scaling RFM features...")
    try:
        scaler = load_scaler(scaler_path)
        print(f"  ✓ Loaded scaler from {scaler_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    scaled_features, _ = scale_rfm(rfm_data, fit_scaler=False, scaler=scaler)
    print(f"  ✓ Scaled RFM features: shape {scaled_features.shape}")

    # Step 4: Assign segments using K-Means
    print("\n[4/5] Assigning segments...")
    try:
        model = load_model(model_path)
        print(f"  ✓ Loaded K-Means model from {model_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Get cluster assignments
    features_only = scaled_features.drop("CustomerID", axis=1) if "CustomerID" in scaled_features.columns else scaled_features
    clusters = model.predict(features_only)
    rfm_data["Cluster"] = clusters
    print(f"  ✓ Assigned {len(clusters)} customers to {len(np.unique(clusters))} clusters")

    # Print cluster distribution
    cluster_dist = rfm_data["Cluster"].value_counts().sort_index()
    for cluster, count in cluster_dist.items():
        pct = (count / len(rfm_data)) * 100
        persona = assign_persona(cluster)
        print(f"    - Cluster {cluster} ({persona.get('name', 'Unknown')}): {count} customers ({pct:.1f}%)")

    # Step 5: Assign personas and create output
    print("\n[5/5] Classifying personas and generating output...")

    # Create output dataframe
    output_df = rfm_data.copy()
    output_df["Persona_Name"] = output_df["Cluster"].apply(
        lambda x: assign_persona(x).get("name", "Unknown")
    )
    output_df["Persona_Description"] = output_df["Cluster"].apply(
        lambda x: assign_persona(x).get("description", "")
    )
    output_df["Priority"] = output_df["Cluster"].apply(
        lambda x: assign_persona(x).get("priority", "")
    )
    output_df["Goal"] = output_df["Cluster"].apply(
        lambda x: assign_persona(x).get("goal", "")
    )
    output_df["Recommended_Tactics"] = output_df["Cluster"].apply(
        lambda x: " | ".join(assign_persona(x).get("tactics", []))
    )
    output_df["Marketing_Channels"] = output_df["Cluster"].apply(
        lambda x: assign_persona(x).get("channels", "")
    )
    output_df["Success_KPI"] = output_df["Cluster"].apply(
        lambda x: assign_persona(x).get("kpi", "")
    )

    # Reorder columns for clarity
    column_order = [
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "Monetary_log",
        "Cluster",
        "Persona_Name",
        "Persona_Description",
        "Priority",
        "Goal",
        "Recommended_Tactics",
        "Marketing_Channels",
        "Success_KPI",
    ]
    available_cols = [c for c in column_order if c in output_df.columns]
    output_df = output_df[available_cols]

    # Save output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved segmented customer data to {output_path}")

    # Generate summary statistics
    print("\n" + "=" * 80)
    print("SEGMENTATION SUMMARY")
    print("=" * 80)

    total_customers = len(output_df)
    total_revenue = output_df["Monetary"].sum()

    print(f"\nCustomer Base:")
    print(f"  • Total Customers: {total_customers:,}")
    print(f"  • Total Revenue: £{total_revenue:,.2f}")
    print(f"  • Average Customer Value: £{total_revenue / total_customers:.2f}")

    print(f"\nSegmentation Breakdown:")
    for cluster in sorted(output_df["Cluster"].unique()):
        segment = output_df[output_df["Cluster"] == cluster]
        persona = assign_persona(cluster)
        seg_customers = len(segment)
        seg_revenue = segment["Monetary"].sum()
        seg_pct_customers = (seg_customers / total_customers) * 100
        seg_pct_revenue = (seg_revenue / total_revenue) * 100

        print(f"\n  Cluster {cluster}: {persona.get('name', 'Unknown')}")
        print(f"    • Customers: {seg_customers} ({seg_pct_customers:.1f}% of base)")
        print(f"    • Revenue: £{seg_revenue:,.2f} ({seg_pct_revenue:.1f}% of total)")
        print(f"    • Avg CLV: £{seg_revenue / seg_customers:.2f}")
        print(f"    • Priority: {persona.get('priority', '')}")
        print(f"    • Goal: {persona.get('goal', '')}")

    print("\n" + "=" * 80)

    return {
        "output_path": output_path,
        "total_customers": total_customers,
        "total_revenue": total_revenue,
        "segmented_df": output_df,
        "cluster_distribution": output_df["Cluster"].value_counts().to_dict(),
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """Command-line interface for the segmentation pipeline."""
    parser = argparse.ArgumentParser(
        description="Segment customers using pre-trained K-Means model and RFM features"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Path to input transaction data (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Path to output segmented data (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default=SCALER_PATH,
        help=f"Path to saved scaler (default: {SCALER_PATH})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help=f"Path to saved K-Means model (default: {MODEL_PATH})",
    )

    args = parser.parse_args()

    try:
        result = segment_customers(
            input_path=args.input,
            output_path=args.output,
            scaler_path=args.scaler,
            model_path=args.model,
        )
        print(f"\n✓ Segmentation complete!")
        print(f"✓ Output saved to: {result['output_path']}")
        return 0
    except Exception as e:
        print(f"\n✗ Error during segmentation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
