import os
import sys

# --- Make project root importable ---
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


def main():
    # 1) Load and featurize price data
    price = load_price_data(symbol_filter="BTC_USD")
    feat = build_price_feature_set(price)
    price_with_signal = generate_rule_based_signal(feat)

    # 2) Load & score sentiment
    sentiment_raw = load_sentiment_csv(
        "data/sentiment_sample.csv",
        asset_filter="BTC-USD",
    )
    scorer = get_sentiment_scorer()  # uses SENTIMENT_ENGINE
    sentiment_scored = apply_sentiment_scorer(sentiment_raw, scorer=scorer)

    # 3) Align sentiment to price timestamps
    sentiment_aligned = aggregate_sentiment_to_prices(
        sentiment_scored,
        price_with_signal,
    )

    # 4) Build combined signal
    combined = generate_combined_signal(price_with_signal, sentiment_aligned)

    # 5) Plot
    close = combined["close"]
    buys = combined[combined["signal_combined"] == 1]
    sells = combined[combined["signal_combined"] == -1]

    plt.figure()
    close.plot(label="Close Price")
    plt.scatter(buys.index, buys["close"], marker="^", label="BUY (combined)", s=60)
    plt.scatter(sells.index, sells["close"], marker="v", label="SELL (combined)", s=60)
    plt.legend()
    plt.title("BTC-USD with Combined Price + Sentiment Signals")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
