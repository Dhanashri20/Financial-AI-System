"""Daily pipeline: refresh data -> get RL decision -> submit paper order.

Run manually or on a schedule (cron / Task Scheduler / n8n) after market
close or before open:

    python scripts/run_daily.py

Every decision is logged to data/decision_log.csv whether or not a trade
fires — that log is your evidence base when you review paper results.
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from stable_baselines3 import PPO

import config
from src import signals
from src.broker import PaperBroker, RiskGateError
from src.rl_env import SignalTradingEnv
from src.rl_train import build_pipeline_data

LOG_PATH = os.path.join(config.DATA_DIR, "decision_log.csv")


def main():
    ticker = config.TICKER
    model = PPO.load(os.path.join(config.MODELS_DIR, f"ppo_{ticker}.zip"))

    # Refresh data (prices + news sentiment + walk-forward forecast tail)
    rl_df, *_ = build_pipeline_data(ticker)
    rl_df.to_parquet(os.path.join(config.DATA_DIR, f"rl_df_{ticker}.parquet"))

    # Latest observation
    env = SignalTradingEnv(df=rl_df, window_size=config.RL_WINDOW_SIZE,
                           frame_bound=(config.RL_WINDOW_SIZE, len(rl_df)))
    obs, _ = env.reset()
    done = False
    while not done:
        last_obs = obs
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    latest = rl_df.iloc[-1]
    decision = signals.decide(model, last_obs, prev_action=None,
                              ml_forecast=float(latest["ml_forecast"]),
                              sentiment=float(latest["sentiment"]))
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] {ticker}: {decision.action} "
          f"(conf {decision.confidence:.0%}) — {decision.rationale}")

    order_result = "no_trade"
    if decision.action in ("BUY", "SELL"):
        try:
            broker = PaperBroker()
            order = broker.submit(ticker, decision.action, qty=5)
            order_result = order["id"]
        except RiskGateError as e:
            order_result = f"blocked: {e}"
            print(order_result)

    # Append to decision log
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker,
        "close": float(latest["Close"]),
        "sentiment": decision.sentiment,
        "ml_forecast": decision.ml_forecast,
        "action": decision.action,
        "confidence": decision.confidence,
        "order_result": order_result,
    }])
    header = not os.path.exists(LOG_PATH)
    row.to_csv(LOG_PATH, mode="a", header=header, index=False)
    print(f"Logged decision -> {LOG_PATH}")


if __name__ == "__main__":
    main()
