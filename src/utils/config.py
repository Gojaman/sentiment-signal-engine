import os
from enum import Enum


class SentimentEngine(str, Enum):
    NAIVE = "naive"
    CLAUDE = "claude"


def get_env_var(name: str, default: str | None = None) -> str | None:
    """
    Safe environment variable getter.
    """
    value = os.getenv(name, default)
    return value


def get_sentiment_engine() -> SentimentEngine:
    """
    Choose which sentiment engine to use based on env var SENTIMENT_ENGINE.
    """
    raw = os.getenv("SENTIMENT_ENGINE", SentimentEngine.NAIVE.value).lower()
    if raw == SentimentEngine.CLAUDE.value:
        return SentimentEngine.CLAUDE
    return SentimentEngine.NAIVE


def anthropic_api_key() -> str | None:
    """
    Return Anthropic API key if set.
    """
    return get_env_var("ANTHROPIC_API_KEY")
