"""FinBERT news sentiment pipeline.

Extracted from your notebook (cells 0-9): Finnhub news pull -> clean ->
FinBERT scoring -> daily aggregated signed sentiment score.
"""
import datetime as dt

import finnhub
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config

MODEL_NAME = "ProsusAI/finbert"
_device = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = (
            AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            .to(_device)
            .eval()
        )
    return _tokenizer, _model


def fetch_news(ticker: str = config.TICKER, days: int = config.NEWS_LOOKBACK_DAYS) -> pd.DataFrame:
    """Pull and clean company news from Finnhub (your cells 1-4)."""
    client = finnhub.Client(api_key=config.FINNHUB_API_KEY)
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    news = client.company_news(ticker, _from=str(start), to=str(end))
    df = pd.DataFrame(news)
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
    df["news_text"] = (
        df["headline"].fillna("") + ". " + df["summary"].fillna("")
    ).str.strip()
    df = (
        df[df["news_text"].str.len() > 5]
        .drop_duplicates(subset="headline")
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    return df


@torch.no_grad()
def score_texts(texts, batch_size: int = 16, max_length: int = 512) -> np.ndarray:
    """FinBERT softmax probabilities per text (your cell 7)."""
    tokenizer, model = _load_model()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(_device)
        logits = model(**enc).logits
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(out)


def score_news(news_df: pd.DataFrame) -> pd.DataFrame:
    """Attach FinBERT probabilities + signed score to a news frame."""
    _, model = _load_model()
    id2label = model.config.id2label
    label2id = {v.lower(): k for k, v in id2label.items()}
    pos_i, neg_i, neu_i = label2id["positive"], label2id["negative"], label2id["neutral"]

    probs = score_texts(news_df["news_text"].tolist())
    news_df = news_df.copy()
    news_df["prob_positive"] = probs[:, pos_i]
    news_df["prob_negative"] = probs[:, neg_i]
    news_df["prob_neutral"] = probs[:, neu_i]
    news_df["sentiment"] = probs.argmax(axis=1)
    news_df["sentiment"] = news_df["sentiment"].map(
        {pos_i: "positive", neg_i: "negative", neu_i: "neutral"}
    )
    # Signed score: P(pos) - P(neg), in [-1, 1]
    news_df["Sentiment_Score"] = news_df["prob_positive"] - news_df["prob_negative"]
    return news_df


def daily_sentiment(scored_news: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to a daily mean signed score + article count (your cell 9)."""
    return (
        scored_news.set_index("datetime")
        .resample("D")["Sentiment_Score"]
        .agg(mean="mean", n="count")
        .dropna()
    )


def get_daily_sentiment(ticker: str = config.TICKER) -> pd.DataFrame:
    """End-to-end: fetch -> score -> daily aggregate."""
    news = fetch_news(ticker)
    if news.empty:
        return pd.DataFrame(columns=["mean", "n"])
    return daily_sentiment(score_news(news))
