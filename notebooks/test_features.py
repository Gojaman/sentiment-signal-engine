import os
import sys

# Add project root so we can import src.*
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.data_loader import load_price_data
from src.features.price_features import build_price_feature_set


def main():
    btc = load_price_data(symbol_filter="BTC_USD")
    features = build_price_feature_set(btc)

    print(features.head())
    print(features.tail())
    print(features.columns)


if __name__ == "__main__":
    main()
