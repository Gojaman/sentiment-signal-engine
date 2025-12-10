import os
import sys

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loader import load_price_data
from src.features.price_features import build_price_feature_set
from src.models.signal_engine import (
    generate_rule_based_signal,
    generate_combined_signal,
)
from src.ingestion.sentiment_ingestion import load_sentiment_csv
from src.features.sentiment_features import (
    apply_sentiment_scorer,
    aggregate_sentiment_to_prices,
    get_sentiment_scorer,
)
import matplotlib.pyplot as plt
import numpy as np


def compute_strategy_returns(df, signal_col: str = "signal") -> tuple[float, "pd.DataFrame"]:
    """
    Very simple backtest:
      - Use signal at time t to hold position for next bar's return
      - signal: +1 (long), 0 (flat), -1 (short)
    """
    df = df.copy()
    if "return" not in df.columns:
        raise ValueError("DataFrame must contain 'return' column")

    df["strategy_ret"] = df[signal_col].shift(1) * df["return"]
    df["cumret"] = df["strategy_ret"].cumsum().apply(np.exp)
    final_value = df["cumret"].iloc[-1]
    return final_value, df


def main():
    # 1) Load prices + build features
    price = load_price_data(symbol_filter="BTC_USD")
    feat = build_price_feature_set(price)

    # 2) Price-only signal
    price_with_signal = generate_rule_based_signal(feat)

    # 3) Sentiment pipeline
    sentiment_raw = load_sentiment_csv("data/sentiment_sample.csv", asset_filter="BTC-USD")
    scorer = get_sentiment_scorer()  # uses SENTIMENT_ENGINE env
    sentiment_scored = apply_sentiment_scorer(sentiment_raw, scorer=scorer)
    sentiment_aligned = aggregate_sentiment_to_prices(sentiment_scored, price_with_signal)

    # 4) Combined signal
    combined = generate_combined_signal(price_with_signal, sentiment_aligned)

    # 5) Backtest price-only
    final_price_only, df_price_bt = compute_strategy_returns(
        price_with_signal, signal_col="signal"
    )

    # 6) Backtest price + sentiment
    final_combined, df_combined_bt = compute_strategy_returns(
        combined, signal_col="signal_combined"
    )

    print(f"Final portfolio value - price only: {final_price_only:.3f}")
    print(f"Final portfolio value - combined:  {final_combined:.3f}")

    # 7) Plot cumulative returns
    plt.figure()
    df_price_bt["cumret"].plot(label="Price-only strategy")
    df_combined_bt["cumret"].plot(label="Price + Sentiment strategy")
    plt.legend()
    plt.title("Cumulative Returns Comparison")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
