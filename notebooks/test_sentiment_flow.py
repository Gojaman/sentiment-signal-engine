import os
import sys

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.utils.data_loader import load_price_data
from src.ingestion.sentiment_ingestion import load_sentiment_csv
from src.features.sentiment_features import (
    apply_sentiment_scorer,
    aggregate_sentiment_to_prices,
    get_sentiment_scorer,
)
from src.utils.config import SentimentEngine


def main(override_engine: SentimentEngine | None = None):
    price = load_price_data(symbol_filter="BTC_USD")
    sent_raw = load_sentiment_csv("data/sentiment_sample.csv", asset_filter="BTC-USD")

    scorer = get_sentiment_scorer(override_engine=override_engine)
    sent_scored = apply_sentiment_scorer(sent_raw, scorer=scorer)
    aligned = aggregate_sentiment_to_prices(sent_scored, price)

    print("Raw sentiment with scores:")
    print(sent_scored)

    print("\nAligned sentiment (head):")
    print(aligned.head())
    print("\nAligned sentiment (tail):")
    print(aligned.tail())


if __name__ == "__main__":
    # Use env-configured engine (default: naive)
    main()
    # Or force Claude:
    # from src.utils.config import SentimentEngine
    # main(override_engine=SentimentEngine.CLAUDE)
