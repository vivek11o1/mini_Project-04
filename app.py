"""FastAPI service for the Customer Segmentation project (Swagger UI at /docs).

Run locally:
    uvicorn app:app --reload
"""

from __future__ import annotations

import io
from typing import List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from helper import load_artifacts, predict_rfm, transactions_to_rfm

model, scaler = load_artifacts()

app = FastAPI(
    title="Customer Segmentation API",
    description=(
        "Assign customers to marketing personas using RFM analysis "
        "(Recency, Frequency, Monetary) and a pre-trained K-Means model. "
        "Try the endpoints below directly from this Swagger UI."
    ),
    version="1.0.0",
)


class CustomerRequest(BaseModel):
    recency: float = Field(..., ge=0, description="Days since the customer's last purchase")
    frequency: float = Field(..., ge=0, description="Number of purchases made by the customer")
    monetary: float = Field(..., ge=0, description="Total amount spent by the customer")


class CustomerBatchRequest(BaseModel):
    customers: List[CustomerRequest] = Field(..., min_length=1, description="Customers to segment")


class Prediction(BaseModel):
    recency: float
    frequency: float
    monetary: float
    cluster: int
    persona_name: str
    description: str
    priority: str
    goal: str
    recommended_tactics: List[str]
    marketing_channels: str
    success_kpi: str


class BatchResponse(BaseModel):
    predictions: List[Prediction]


def _segment(customers: List[CustomerRequest]) -> BatchResponse:
    frame = pd.DataFrame(
        [
            {
                "Recency": customer.recency,
                "Frequency": customer.frequency,
                "Monetary": customer.monetary,
            }
            for customer in customers
        ]
    )
    return BatchResponse(predictions=predict_rfm(frame, model, scaler))


@app.get("/", tags=["info"], summary="API information")
def root():
    return {
        "name": "Customer Segmentation API",
        "model": "RFM features + K-Means clustering (K=4)",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["info"], summary="Health check")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "n_clusters": model.n_clusters,
        "features": ["Recency", "Frequency", "Monetary"],
        "personas": [
            "CHAMPIONS",
            "LOYAL CUSTOMERS",
            "AT-RISK / DORMANT",
            "GROWTH OPPORTUNITY",
        ],
    }


@app.post("/predict", response_model=Prediction, tags=["prediction"], summary="Segment one customer")
def predict(customer: CustomerRequest):
    """Return the assigned cluster, persona, priority and marketing tactics."""
    return _segment([customer]).predictions[0]


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    tags=["prediction"],
    summary="Segment several customers",
)
def predict_batch(batch: CustomerBatchRequest):
    """Segment a list of customers in one request."""
    return _segment(batch.customers)


@app.post(
    "/predict/transactions",
    response_model=BatchResponse,
    tags=["prediction"],
    summary="Segment customers from a raw transactions CSV",
)
async def predict_transactions(
    file: UploadFile = File(..., description="CSV with InvoiceDate, CustomerID, Quantity, UnitPrice"),
):
    """Calculate RFM metrics from retail transactions, then assign personas."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    try:
        transactions = pd.read_csv(io.BytesIO(await file.read()))
        rfm = transactions_to_rfm(transactions)
        results = predict_rfm(rfm, model, scaler)
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except pd.errors.ParserError as error:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {error}") from error

    if not results:
        raise HTTPException(status_code=400, detail="No valid customers found in the CSV")

    return BatchResponse(predictions=results)
