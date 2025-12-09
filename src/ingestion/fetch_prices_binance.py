import os
from datetime import datetime
from typing import List

import pandas as pd
from binance.client import Client


SYMBOLS: List[str] = ["BTCUSDT", "ETHUSDT"]
DATA_DIR = "data"


def get_binance_client() -> Client:
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    return Client(api_key, api_secret)


def fetch_klines(
    client: Client,
    symbol: str,
    interval: str = Client.KLINE_INTERVAL_1HOUR,
    limit: int = 500,
) -> pd.DataFrame:
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_vol",
        "taker_buy_quote_vol",
        "ignore",
    ]

    klines: List[List] = client.get_klines(
        symbol=symbol,
        interval=interval,
        limit=limit,
    )

    df = pd.DataFrame(klines, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df.set_index("open_time")
    return df[["open", "high", "low", "close", "volume"]]


def save_to_csv(df: pd.DataFrame, symbol: str, data_dir: str = DATA_DIR) -> str:
    os.makedirs(data_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_{timestamp}.csv"
    full_path = os.path.join(data_dir, filename)
    df.to_csv(full_path, index_label="timestamp")
    return full_path


def main() -> None:
    client = get_binance_client()
    for symbol in SYMBOLS:
        print(f"Fetching klines for {symbol}...")
        df = fetch_klines(client, symbol=symbol)
        path = save_to_csv(df, symbol)
        print(f"✅ Saved {len(df)} rows for {symbol} to {path}")
    print("Done.")


if __name__ == "__main__":
    main()
