import os
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import yfinance as yf


ASSETS: List[str] = ["BTC-USD", "ETH-USD"]
DATA_DIR = "data"
DEFAULT_DAYS = 90


def fetch_price_history(
    ticker: str,
    days: int = DEFAULT_DAYS,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Fetch OHLCV price data for the given ticker using Yahoo Finance.

    :param ticker: e.g. "BTC-USD", "ETH-USD"
    :param days: number of days of history to pull
    :param interval: bar interval, e.g. "1h", "30m", "1d"
    :return: pandas DataFrame indexed by timestamp
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    # Ensure timezone-naive index
    df.index = df.index.tz_localize(None)
    return df


def ensure_data_dir(path: str = DATA_DIR) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_to_csv(df: pd.DataFrame, symbol: str, data_dir: str = DATA_DIR) -> str:
    ensure_data_dir(data_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol.replace('-', '_')}_{timestamp}.csv"
    full_path = os.path.join(data_dir, filename)
    df.to_csv(full_path, index_label="timestamp")
    return full_path


def main() -> None:
    print("Starting price ingestion with Yahoo Finance...")
    for ticker in ASSETS:
        print(f"Fetching data for {ticker}...")
        df = fetch_price_history(ticker)
        path = save_to_csv(df, ticker)
        print(f"✅ Saved {len(df)} rows for {ticker} to {path}")
    print("Done.")


if __name__ == "__main__":
    main()
