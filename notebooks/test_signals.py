import os
import sys

# Add project root so we can import src.*
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.data_loader import load_price_data
from src.features.price_features import build_price_feature_set
from src.models.signal_engine import generate_rule_based_signal
import matplotlib.pyplot as plt


def main():
    btc = load_price_data(symbol_filter="BTC_USD")
    feat = build_price_feature_set(btc)
    with_signals = generate_rule_based_signal(feat)

    print(with_signals[["close", "ma_20", "rsi_14", "signal"]].tail())

    # Simple visualization: price + buy/sell markers
    close = with_signals["close"]
    buys = with_signals[with_signals["signal"] == 1]
    sells = with_signals[with_signals["signal"] == -1]

    plt.figure()
    close.plot(label="Close Price")
    plt.scatter(buys.index, buys["close"], marker="^", label="BUY", s=50)
    plt.scatter(sells.index, sells["close"], marker="v", label="SELL", s=50)
    plt.legend()
    plt.title("BTC-USD with Rule-Based Signals")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
