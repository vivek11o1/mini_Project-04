import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame, fit_scaler: bool = False, scaler=None) -> tuple:
    """
    Create model-ready features for clustering.

    - Scale numeric columns (age, annual_income, spending_score).
    - One-hot encode gender if present.

    Args:
        df: Input DataFrame
        fit_scaler: If True, fit a new scaler and save it. If False, use provided scaler.
        scaler: Pre-fitted scaler object (used for inference)

    Returns:
        Tuple of (engineered_features_df, scaler_object)
    """
    expected_numeric = ["age", "annual_income", "spending_score"]
    missing = [c for c in expected_numeric if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    features = df.copy()

    if fit_scaler:
        # Training mode: fit a new scaler
        scaler = StandardScaler()
        scaled_numeric = scaler.fit_transform(features[expected_numeric])
    else:
        # Inference mode: use provided scaler
        if scaler is None:
            raise ValueError("Scaler must be provided for inference mode")
        scaled_numeric = scaler.transform(features[expected_numeric])

    numeric_df = pd.DataFrame(
        scaled_numeric,
        columns=[f"{col}_scaled" for col in expected_numeric],
        index=features.index,
    )

    gender_df = pd.DataFrame(index=features.index)
    if "gender" in features.columns:
        gender_df = pd.get_dummies(features["gender"], prefix="gender", drop_first=True)

    engineered = pd.concat([numeric_df, gender_df], axis=1)
    return engineered, scaler


def save_scaler(scaler, path: str = "models/scaler.pkl") -> None:
    """Save a fitted StandardScaler to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str = "models/scaler.pkl") -> StandardScaler:
    """Load a fitted StandardScaler from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
