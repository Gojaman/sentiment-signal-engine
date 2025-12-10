import os
import sys

# Add project root to sys.path so we can import src.*
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.data_loader import load_price_data
import matplotlib.pyplot as plt


def main():
    btc = load_price_data(symbol_filter="BTC_USD")
    close = btc["close"]

    plt.figure()
    close.plot(title="BTC-USD Close Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
