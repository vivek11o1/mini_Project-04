"""Model loading and segmentation helpers for the FastAPI service.

The original repository does not commit its ``.pkl`` artefacts (see
``.gitignore``), so they are rebuilt from ``data/processed/rfm_segmented.csv``
on first use, using the same settings as ``retrain_model.py``.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.rfm_features import calculate_rfm  # noqa: E402
from src.run_segmentation import assign_persona  # noqa: E402

MODEL_PATH = BASE_DIR / "models" / "kmeans_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
TRAINING_DATA_PATH = BASE_DIR / "data" / "processed" / "rfm_segmented.csv"

FEATURES = ["Recency", "Frequency", "Monetary"]
RANDOM_STATE = 42
N_CLUSTERS = 4


def ensure_artifacts() -> None:
    """Train and save the scaler + K-Means model if they do not exist yet."""
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        return

    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot build model artefacts: training data not found at {TRAINING_DATA_PATH}"
        )

    rfm = pd.read_csv(TRAINING_DATA_PATH)
    missing = [column for column in FEATURES if column not in rfm.columns]
    if missing:
        raise ValueError(f"Training data is missing RFM columns: {missing}")

    features = rfm[FEATURES].astype(float)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    model.fit(scaled_features)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCALER_PATH.open("wb") as file:
        pickle.dump(scaler, file)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)


def load_artifacts() -> tuple[KMeans, StandardScaler]:
    """Load (or create) the K-Means model and the fitted StandardScaler."""
    ensure_artifacts()
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    with SCALER_PATH.open("rb") as file:
        scaler = pickle.load(file)
    return model, scaler


def _prediction_record(
    recency: float,
    frequency: float,
    monetary: float,
    cluster: int,
) -> Dict[str, Any]:
    persona = assign_persona(int(cluster))
    return {
        "recency": float(recency),
        "frequency": float(frequency),
        "monetary": float(monetary),
        "cluster": int(cluster),
        "persona_name": persona.get("name", "Unknown"),
        "description": persona.get("description", ""),
        "priority": persona.get("priority", ""),
        "goal": persona.get("goal", ""),
        "recommended_tactics": persona.get("tactics", []),
        "marketing_channels": persona.get("channels", ""),
        "success_kpi": persona.get("kpi", ""),
    }


def predict_rfm(
    rfm: pd.DataFrame,
    model: KMeans,
    scaler: StandardScaler,
) -> List[Dict[str, Any]]:
    """Assign a persona to every customer in an RFM DataFrame."""
    missing = [column for column in FEATURES if column not in rfm.columns]
    if missing:
        raise ValueError(f"Missing RFM columns: {missing}")

    features = rfm[FEATURES].astype(float)
    if features.isna().any().any():
        raise ValueError("RFM values must not be missing")

    clusters = model.predict(scaler.transform(features))
    return [
        _prediction_record(row.Recency, row.Frequency, row.Monetary, cluster)
        for row, cluster in zip(features.itertuples(index=False), clusters)
    ]


def transactions_to_rfm(transactions: pd.DataFrame) -> pd.DataFrame:
    """Convert raw retail transactions into per-customer RFM metrics."""
    data = transactions.copy()
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

    if "invoicedate" in data.columns:
        data.rename(columns={"invoicedate": "transaction_date"}, inplace=True)
    if "customerid" in data.columns:
        data.rename(columns={"customerid": "customer_id"}, inplace=True)
    if "amount" not in data.columns and {"quantity", "unitprice"} <= set(data.columns):
        data["amount"] = data["quantity"] * data["unitprice"]

    required = {"transaction_date", "customer_id", "amount"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "Transaction CSV must contain InvoiceDate, CustomerID, Quantity and UnitPrice "
            f"(missing: {', '.join(sorted(missing))})"
        )

    return calculate_rfm(
        data,
        transaction_date_col="transaction_date",
        customer_id_col="customer_id",
        amount_col="amount",
    )
