"""Feature engineering + price data loading.

Extracted from your notebook (cells 11-16), with one bug fix:
your original code assigned `sma_ratio_10` twice — the second assignment
(20-day SMA) overwrote the first, so the 10-day feature was lost.
Fixed here as `sma_ratio_10` and `sma_ratio_20`.
"""
import numpy as np
import pandas as pd
import yfinance as yf

import config


def load_prices(ticker: str = config.TICKER, lookback: str = config.LOOKBACK) -> pd.DataFrame:
    raw = yf.download(ticker, period=lookback, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    raw.index = pd.to_datetime(raw.index)
    return raw


def align_sentiment(close: pd.Series, daily_sent: pd.DataFrame):
    """Align daily sentiment to trading days (your cell 13)."""
    if daily_sent is not None and not daily_sent.empty and "mean" in daily_sent.columns:
        sent_src = daily_sent["mean"].copy()
        sent_src.index = pd.to_datetime(sent_src.index)
    else:
        sent_src = pd.Series(dtype=float)

    sentiment = sent_src.reindex(close.index)
    has_sentiment = sentiment.notna().astype("float")
    sentiment = sentiment.ffill().fillna(0.0)
    return sentiment, has_sentiment


def compute_features(close, volume, sent, has_sent) -> pd.DataFrame:
    """Your cell 14, with the sma_ratio bug fixed."""
    f = pd.DataFrame(index=close.index)
    r = np.log(close).diff()

    # Lagged returns + rolling vol
    f["ret_1"] = r
    f["ret_2"] = r.shift(1)
    f["ret_3"] = r.shift(2)
    f["ret_5"] = r.shift(4)
    f["roll_mean_5"] = r.rolling(5).mean()
    f["roll_std_5"] = r.rolling(5).std()
    f["roll_std_10"] = r.rolling(10).std()

    # Price relative to moving averages  (BUG FIX: was sma_ratio_10 twice)
    f["sma_ratio_10"] = close / close.rolling(10).mean() - 1
    f["sma_ratio_20"] = close / close.rolling(20).mean() - 1

    # RSI(14)
    d = close.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    f["rsi_14"] = 100 - 100 / (1 + rs)

    # Volume signal
    f["vol_ratio_10"] = volume / volume.rolling(10).mean() - 1

    # Sentiment (lagged one day to avoid lookahead)
    f["sent_1"] = sent.shift(1)
    f["sent_roll_3"] = sent.shift(1).rolling(3).mean()
    f["has_sent"] = has_sent.shift(1)

    return f


def build_dataset(close, volume, sentiment, has_sentiment, test_days: int = config.TEST_DAYS):
    """Feature matrix + next-day log-return target, chronological split (your cell 16)."""
    F = compute_features(close, volume, sentiment, has_sentiment)
    feat_cols = list(F.columns)

    returns = np.log(close).diff().rename("ret")
    X_all = F.shift(1)  # features of day t-1 predict return of day t (no lookahead)
    data = pd.concat([X_all, returns], axis=1).dropna()
    X, y = data[feat_cols], data["ret"]

    split = X.index[-test_days]
    train_mask = X.index < split
    test_mask = X.index >= split
    return {
        "F": F,
        "feat_cols": feat_cols,
        "X_train": X[train_mask], "X_test": X[test_mask],
        "y_train": y[train_mask], "y_test": y[test_mask],
        "split": split,
    }
