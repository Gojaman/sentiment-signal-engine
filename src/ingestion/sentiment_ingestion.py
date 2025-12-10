import os
import pandas as pd
from typing import Optional


def load_sentiment_csv(
    path: str,
    asset_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a generic sentiment CSV.
    Expected columns:
        timestamp, asset, text
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sentiment CSV not found: {path}")

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Ensure required columns
    required = ["timestamp", "asset", "text"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from sentiment CSV")

    # Filter BTC-USD / ETH-USD etc
    if asset_filter:
        df = df[df["asset"] == asset_filter]

    # Convert timestamp to datetime — ensure UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df   # <<<< MUST BE HERE
