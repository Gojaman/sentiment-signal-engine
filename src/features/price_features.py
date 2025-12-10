import pandas as pd
import numpy as np


def add_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    df = df.copy()
    df["return"] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def add_moving_averages(
    df: pd.DataFrame,
    price_col: str = "close",
    windows=(10, 20, 50),
) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"ma_{w}"] = df[price_col].rolling(window=w).mean()
    return df


def add_volatility(
    df: pd.DataFrame,
    return_col: str = "return",
    window: int = 20,
) -> pd.DataFrame:
    df = df.copy()
    if return_col not in df.columns:
        raise ValueError(f"{return_col} not found in DataFrame")
    df[f"vol_{window}"] = df[return_col].rolling(window=window).std()
    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) on a price series.
    """
    delta = series.diff()
    gain = delta.where(delta > 0.0, 0.0)
    loss = -delta.where(delta < 0.0, 0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def add_rsi(
    df: pd.DataFrame,
    price_col: str = "close",
    window: int = 14,
) -> pd.DataFrame:
    df = df.copy()
    df[f"rsi_{window}"] = compute_rsi(df[price_col], window=window)
    return df


def build_price_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all standard price-based features in one go.
    """
    df = add_log_returns(df)
    df = add_moving_averages(df)
    df = add_volatility(df)
    df = add_rsi(df)
    df = df.dropna()
    return df
