import pandas as pd


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized snake_case column names."""
    renamed = (
        df.copy()
        .rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
    )
    return renamed


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for the customer dataset.

    - Normalize column names.
    - Drop duplicates based on customer_id when present, otherwise full row.
    - Remove rows with missing core fields.
    - Coerce numeric columns to numbers and drop invalid rows.
    """
    normalized = _standardize_columns(df)

    # Prefer de-duplicating on customer_id to avoid dropping legitimate identical rows.
    if "customer_id" in normalized.columns:
        normalized = normalized.drop_duplicates(subset=["customer_id"])
    else:
        normalized = normalized.drop_duplicates()

    core_columns = [c for c in ["age", "annual_income", "spending_score"] if c in normalized.columns]
    normalized = normalized.dropna(subset=core_columns)

    for col in core_columns:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    normalized = normalized.dropna(subset=core_columns)

    for col in ["age", "annual_income", "spending_score"]:
        if col in normalized.columns:
            normalized = normalized[normalized[col] >= 0]

    return normalized.reset_index(drop=True)


def load_raw_data(path: str) -> pd.DataFrame:
    """Load a raw CSV/Excel file into a DataFrame."""
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format for path: {path}")
