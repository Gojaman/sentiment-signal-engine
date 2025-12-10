import json
from typing import Callable, Optional

import pandas as pd

from src.utils.config import (
    SentimentEngine,
    get_sentiment_engine,
    anthropic_api_key,
)

try:
    import anthropic
except ImportError:
    anthropic = None


# ================================================
#  SIMPLE LOCAL (NAIVE) LEXICON SENTIMENT
# ================================================

def simple_lexicon_sentiment(text: str) -> float:
    """
    Very naive sentiment scorer. Returns value in [0, 1].
    """
    if not isinstance(text, str) or not text.strip():
        return 0.5

    text_lower = text.lower()
    positive = ["surge", "rally", "approval", "growth", "bullish", "strong", "support"]
    negative = ["crash", "dump", "concern", "fear", "regulation", "selloff", "ban"]

    score = 0.5
    for w in positive:
        if w in text_lower:
            score += 0.1
    for w in negative:
        if w in text_lower:
            score -= 0.1

    return max(0.0, min(1.0, score))


# ================================================
#  CLAUDE SENTIMENT (ANTHROPIC API)
# ================================================

def _get_anthropic_client():
    """Create Anthropic client using ANTHROPIC_API_KEY."""
    if anthropic is None:
        raise ImportError("anthropic library not installed. Run: pip install anthropic")

    key = anthropic_api_key()
    if not key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set.")
    return anthropic.Anthropic(api_key=key)


def claude_sentiment_scorer(text: str) -> float:
    """Call Claude to score sentiment in [0, 1]."""
    if not isinstance(text, str) or not text.strip():
        return 0.5

    try:
        client = _get_anthropic_client()
    except Exception:
        print("CLAUDE DEBUG: client init failed, falling back to 0.5")
        return 0.5

    prompt = f"""
You are a financial sentiment classifier.

Text:
\"\"\"{text}\"\"\"

Respond ONLY with JSON:
{{"score": number between 0.0 and 1.0}}
"""

    try:
        print("CLAUDE DEBUG: sending prompt:", text[:60], "...")
        response = client.messages.create(
            model="claude-3-haiku-latest",
            max_tokens=64,
            temperature=0.0,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        content = response.content
        if not content:
            print("CLAUDE DEBUG: empty content, falling back to 0.5")
            return 0.5

        text_out = content[0].text.strip()
        print("CLAUDE DEBUG: raw response:", text_out)

        try:
            obj = json.loads(text_out)
            score = float(obj.get("score", 0.5))
        except Exception:
            score = float(text_out.strip())

        return max(0.0, min(1.0, score))

    except Exception as e:
        print("CLAUDE DEBUG: API error:", repr(e))
        return 0.5


# ================================================
#  SELECT SENTIMENT ENGINE
# ================================================

def get_sentiment_scorer(
    override_engine: Optional[SentimentEngine] = None,
) -> Callable[[str], float]:
    """
    Return a scorer function based on config/env.
    """
    engine = override_engine or get_sentiment_engine()
    print("DEBUG — Using sentiment engine:", engine)

    if engine == SentimentEngine.CLAUDE:
        return claude_sentiment_scorer

    return simple_lexicon_sentiment


# ================================================
#  APPLY SENTIMENT TO DATAFRAME
# ================================================

def apply_sentiment_scorer(
    df: pd.DataFrame,
    scorer: Optional[Callable[[str], float]] = None,
) -> pd.DataFrame:
    """
    Apply sentiment scoring to a DataFrame.
    Requires column 'text'.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("apply_sentiment_scorer received None instead of DataFrame")

    if "text" not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column.")

    df = df.copy()
    scorer_fn = scorer or get_sentiment_scorer()

    df["sentiment_score"] = df["text"].apply(scorer_fn)
    return df


# ================================================
#  ALIGN SENTIMENT WITH PRICE DATA
# ================================================

def aggregate_sentiment_to_prices(
    sentiment_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reindex sentiment to match price timestamps using forward-fill.

    Fixes timezone mismatches:
    - Sentiment may be UTC (aware)
    - Price timestamps are naive
    """
    if sentiment_df is None or sentiment_df.empty:
        raise ValueError("Sentiment DataFrame is empty.")

    if price_df is None or price_df.empty:
        raise ValueError("Price DataFrame is empty.")

    s = sentiment_df.copy()

    # Ensure timestamp is index & UTC
    if "timestamp" in s.columns:
        s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True)
        s = s.set_index("timestamp")
    else:
        s.index = pd.to_datetime(s.index, utc=True)

    price_index = pd.to_datetime(price_df.index, utc=True)

    s = s.sort_index()
    s = s[["sentiment_score"]]

    aligned = s.reindex(price_index, method="ffill")
    aligned.index = price_df.index  # restore naive

    return aligned

