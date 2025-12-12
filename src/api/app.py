from __future__ import annotations

import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils.data_loader import load_price_data
from src.features.price_features import build_price_feature_set
from src.models.signal_engine import generate_rule_based_signal, generate_combined_signal
from src.ingestion.sentiment_ingestion import load_sentiment_csv
from src.features.sentiment_features import apply_sentiment_scorer, aggregate_sentiment_to_prices, get_sentiment_scorer

app = FastAPI(title="Market Sentiment Signal Engine", version="0.1.0")


class SentimentScoreRequest(BaseModel):
    text: str = Field(..., min_length=1)
    asset: Optional[str] = None


class SentimentScoreResponse(BaseModel):
    score: float
    engine: str


class SignalResponse(BaseModel):
    asset: str
    mode: str
    latest_timestamp: str
    latest_signal: int
    latest_sentiment: Optional[float] = None


def _load_price_pipeline(asset: str):
    symbol_filter = asset.replace("-", "_")  # BTC-USD -> BTC_USD
    price = load_price_data(symbol_filter=symbol_filter)
    feat = build_price_feature_set(price)
    price_sig = generate_rule_based_signal(feat)
    return price_sig


def _load_aligned_sentiment(asset: str, price_df):
    path = os.getenv("SENTIMENT_CSV_PATH", "data/sentiment_sample.csv")
    sent_raw = load_sentiment_csv(path, asset_filter=asset)
    scorer = get_sentiment_scorer()  # naive or claude (env-controlled)
    sent_scored = apply_sentiment_scorer(sent_raw, scorer=scorer)
    sent_aligned = aggregate_sentiment_to_prices(sent_scored, price_df)
    return sent_aligned


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sentiment/score", response_model=SentimentScoreResponse)
def sentiment_score(req: SentimentScoreRequest):
    scorer = get_sentiment_scorer()
    score = float(scorer(req.text))
    score = max(0.0, min(1.0, score))
    return SentimentScoreResponse(score=score, engine=os.getenv("SENTIMENT_ENGINE", "naive"))


@app.get("/signal", response_model=SignalResponse)
def get_signal(
    asset: str = "BTC-USD",
    mode: Literal["price_only", "combined"] = "combined",
):
    price_sig = _load_price_pipeline(asset)

    latest_ts = price_sig.index[-1]
    latest_signal = int(price_sig["signal"].iloc[-1])
    latest_sentiment: Optional[float] = None

    if mode == "combined":
        sent_aligned = _load_aligned_sentiment(asset, price_sig)
        combined = generate_combined_signal(price_sig, sent_aligned)
        latest_signal = int(combined["signal_combined"].iloc[-1])
        latest_sentiment = float(combined["sentiment_score"].iloc[-1])

    return SignalResponse(
        asset=asset,
        mode=mode,
        latest_timestamp=latest_ts.isoformat(),
        latest_signal=latest_signal,
        latest_sentiment=latest_sentiment,
    )
