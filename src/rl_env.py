"""RL trading environment with custom signals (gym-anytrading pattern).

Follows the custom-signals approach from
https://github.com/nicknochnack/Reinforcement-Learning-for-Trading-Custom-Signals
— override StocksEnv._process_data — but the custom features are YOUR
FinBERT sentiment and YOUR ML ensemble forecast instead of SMA/RSI/OBV.

This module also builds the bridge dataframe your notebook was missing:
cells 30-31 referenced df["sentiment"] / df["ml_forecast"], which didn't
exist yet. `build_rl_dataframe` constructs it from your pipeline outputs.
"""
import numpy as np
import pandas as pd
from gym_anytrading.envs import StocksEnv

import config


def build_rl_dataframe(prices: pd.DataFrame, sentiment: pd.Series,
                       ml_forecast: pd.Series) -> pd.DataFrame:
    """Assemble OHLCV + sentiment + ml_forecast into the frame the env needs.

    ml_forecast should be WALK-FORWARD predictions (see forecasting.py),
    not in-sample fits, or the agent trains on leaked information.
    """
    df = prices.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }).copy()
    df["sentiment"] = sentiment.reindex(df.index).ffill().fillna(0.0)
    df["ml_forecast"] = ml_forecast.reindex(df.index)
    # Drop the warm-up region where walk-forward predictions don't exist yet
    df = df.dropna(subset=["ml_forecast"])
    return df


def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, "Close"].to_numpy()[start:end]
    sentiment = env.df.loc[:, "sentiment"].to_numpy()[start:end]
    ml_forecast = env.df.loc[:, "ml_forecast"].to_numpy()[start:end]

    def zscore(x):
        std = x.std()
        return (x - x.mean()) / std if std > 0 else x * 0

    signal_features = np.column_stack((
        prices,
        zscore(sentiment),
        zscore(ml_forecast),
    ))
    return prices, signal_features


class SignalTradingEnv(StocksEnv):
    _process_data = add_signals


def make_envs(rl_df: pd.DataFrame, window_size: int = config.RL_WINDOW_SIZE,
              train_frac: float = 0.8):
    train_end = int(len(rl_df) * train_frac)
    env_train = SignalTradingEnv(df=rl_df, window_size=window_size,
                                 frame_bound=(window_size, train_end))
    env_test = SignalTradingEnv(df=rl_df, window_size=window_size,
                                frame_bound=(train_end, len(rl_df)))
    return env_train, env_test
