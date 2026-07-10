"""Train the PPO trading agent end-to-end.

Run from the project root:
    python -m src.rl_train

Pipeline: prices -> FinBERT daily sentiment -> features -> walk-forward
ML forecasts -> RL dataframe -> PPO training -> evaluation -> save model.
"""
import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import config
from src import features, forecasting, rl_env, sentiment


def build_pipeline_data(ticker: str = config.TICKER, use_live_sentiment: bool = True):
    prices = features.load_prices(ticker)
    close, volume = prices["close"].astype(float), prices["volume"].astype(float)

    if use_live_sentiment and config.FINNHUB_API_KEY:
        daily_sent = sentiment.get_daily_sentiment(ticker)
    else:
        daily_sent = pd.DataFrame()  # runs with neutral sentiment
    sent, has_sent = features.align_sentiment(close, daily_sent)

    ds = features.build_dataset(close, volume, sent, has_sent)

    # Leakage-free historical forecasts for the RL state
    print("Generating walk-forward ML forecasts (this refits models repeatedly)...")
    wf_preds = forecasting.walk_forward_predictions(
        pd.concat([ds["X_train"], ds["X_test"]]),
        pd.concat([ds["y_train"], ds["y_test"]]),
    )

    rl_df = rl_env.build_rl_dataframe(prices, sent, wf_preds)
    return rl_df, ds, prices, sent


def train(ticker: str = config.TICKER, timesteps: int = config.RL_TIMESTEPS):
    rl_df, ds, prices, sent = build_pipeline_data(ticker)
    print(f"RL dataframe: {len(rl_df)} rows "
          f"({rl_df.index.min().date()} -> {rl_df.index.max().date()})")

    env_train, env_test = rl_env.make_envs(rl_df)
    vec_env = DummyVecEnv([lambda: env_train])

    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4,
                n_steps=2048, batch_size=64, gamma=0.99)
    model.learn(total_timesteps=timesteps)

    # Evaluate on held-out window
    obs, info = env_test.reset()
    done, total_reward = False, 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Test window total reward: {total_reward:.2f}")
    print(f"Test window total profit: {info.get('total_profit', float('nan')):.4f}")

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    path = os.path.join(config.MODELS_DIR, f"ppo_{ticker}.zip")
    model.save(path)
    print(f"Saved model -> {path}")

    # Persist the RL dataframe so the dashboard / daily runner can reuse it
    rl_df.to_parquet(os.path.join(config.DATA_DIR, f"rl_df_{ticker}.parquet"))
    return model, rl_df


if __name__ == "__main__":
    train()
