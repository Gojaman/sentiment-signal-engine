import pandas as pd


def generate_rule_based_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple rule-based signal using MA and RSI.
    signal:
      +1 = BUY
       0 = HOLD
      -1 = SELL
    """
    df = df.copy()

    # Ensure required cols exist
    required_cols = ["close", "ma_20", "rsi_14"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    conditions_buy = (df["close"] > df["ma_20"]) & (df["rsi_14"] > 55)
    conditions_sell = (df["close"] < df["ma_20"]) & (df["rsi_14"] < 45)

    df["signal"] = 0
    df.loc[conditions_buy, "signal"] = 1
    df.loc[conditions_sell, "signal"] = -1

    return df


def generate_combined_signal(
    price_df: pd.DataFrame,
    sentiment_aligned: pd.DataFrame,
    sentiment_col: str = "sentiment_score",
) -> pd.DataFrame:
    """
    Combine price-based signal with sentiment.

    - price_df: must contain 'signal' column from generate_rule_based_signal
    - sentiment_aligned: index-aligned DataFrame with 'sentiment_score'

    Rules (you can tweak later):
      sentiment_score > 0.55 -> sentiment_signal = +1
      sentiment_score < 0.45 -> sentiment_signal = -1
      otherwise              -> sentiment_signal = 0

    Combined:
      - If sentiment_signal == 0 -> keep price signal
      - If same sign             -> keep that sign
      - If opposite sign         -> flatten to 0 (stay out)
    """
    if "signal" not in price_df.columns:
        raise ValueError("price_df must contain 'signal' column")

    if sentiment_col not in sentiment_aligned.columns:
        raise ValueError(
            f"sentiment_aligned must contain '{sentiment_col}' column"
        )

    df = price_df.copy()

    # Attach sentiment_score
    df[sentiment_col] = sentiment_aligned[sentiment_col]

    # Build sentiment signal
    s = df[sentiment_col].fillna(0.5)  # neutral if missing
    sentiment_signal = pd.Series(0, index=df.index, dtype=int)
    sentiment_signal[s > 0.55] = 1
    sentiment_signal[s < 0.45] = -1

    df["signal_price"] = df["signal"]
    df["signal_sentiment"] = sentiment_signal

    combined = []

    for sp, ss in zip(df["signal_price"], df["signal_sentiment"]):
        if ss == 0:
            # sentiment neutral → trust price
            combined.append(sp)
        elif sp == 0:
            # no price signal → can follow sentiment
            combined.append(ss)
        elif sp == ss:
            # both agree
            combined.append(sp)
        else:
            # disagreement → flatten (risk control)
            combined.append(0)

    df["signal_combined"] = combined

    return df
