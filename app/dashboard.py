"""Streamlit dashboard — run from the project root:

    streamlit run app/dashboard.py

Tabs: Overview | Forecast | Sentiment | RL Decision | Paper Trading.
Loads the trained PPO model and cached RL dataframe produced by
`python -m src.rl_train`. Trading actions go through the risk-gated
Alpaca paper connector.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src import signals

st.set_page_config(page_title="AI Trading Decision-Support", layout="wide")

TICKER = config.TICKER
MODEL_PATH = os.path.join(config.MODELS_DIR, f"ppo_{TICKER}.zip")
RL_DF_PATH = os.path.join(config.DATA_DIR, f"rl_df_{TICKER}.parquet")


# ---------------- data / model loading ----------------
@st.cache_resource
def load_model():
    from stable_baselines3 import PPO
    return PPO.load(MODEL_PATH)


@st.cache_data
def load_rl_df():
    return pd.read_parquet(RL_DF_PATH)


missing = [p for p in (MODEL_PATH, RL_DF_PATH) if not os.path.exists(p)]
if missing:
    st.error(
        "Missing artifacts:\n\n" + "\n".join(f"- `{m}`" for m in missing) +
        "\n\nRun `python -m src.rl_train` first to train the agent and cache data."
    )
    st.stop()

model = load_model()
rl_df = load_rl_df()

# Latest observation = last `window_size` rows of signal features
from src.rl_env import SignalTradingEnv  # noqa: E402

env = SignalTradingEnv(df=rl_df, window_size=config.RL_WINDOW_SIZE,
                       frame_bound=(config.RL_WINDOW_SIZE, len(rl_df)))
obs, _ = env.reset()
# Roll env forward to the most recent observation
done = False
while not done:
    last_obs = obs
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

latest = rl_df.iloc[-1]
decision = signals.decide(
    model, last_obs, prev_action=None,
    ml_forecast=float(latest["ml_forecast"]),
    sentiment=float(latest["sentiment"]),
)

# ---------------- sidebar ----------------
st.sidebar.title("⚙️ Controls")
st.sidebar.metric("Ticker", TICKER)
st.sidebar.caption(f"Data through {rl_df.index.max().date()}")
qty = st.sidebar.number_input("Order quantity", 1, config.MAX_POSITION_QTY, 5)
st.sidebar.markdown("---")
st.sidebar.caption("Circuit breakers: "
                   f"max daily loss {config.MAX_DAILY_LOSS_PCT}%, "
                   f"max qty {config.MAX_POSITION_QTY}, "
                   f"min confidence {config.MIN_RL_CONFIDENCE:.0%}")

if "order_log" not in st.session_state:
    st.session_state.order_log = []

# ---------------- main ----------------
st.title("📊 AI Trading Decision-Support Dashboard")
tabs = st.tabs(["Overview", "Forecast", "Sentiment", "RL Decision", "Paper Trading"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close", f"${latest['Close']:.2f}")
    color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}[decision.action]
    c2.metric("Signal", f"{color} {decision.action}")
    c3.metric("Confidence", f"{decision.confidence:.0%}")
    c4.metric("Sentiment", f"{decision.sentiment:+.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rl_df.index[-120:], y=rl_df["Close"].iloc[-120:],
                             mode="lines", name="Close"))
    fig.update_layout(title=f"{TICKER} — Close (last 120 trading days)", height=350)
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("ML Ensemble Forecast (walk-forward)")
    st.metric("Predicted next-day log return", f"{decision.ml_forecast:+.4f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rl_df.index[-120:], y=rl_df["ml_forecast"].iloc[-120:],
                             mode="lines", name="ml_forecast"))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(title="Walk-forward predicted next-day log return", height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Predictions are out-of-sample (expanding-window refits) — the same "
               "quality of signal the RL agent was trained on.")

with tabs[2]:
    st.subheader("FinBERT Daily Sentiment")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rl_df.index[-120:], y=rl_df["sentiment"].iloc[-120:],
                             mode="lines+markers", name="sentiment"))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(title="Daily signed sentiment score (P(pos) − P(neg))", height=320)
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("RL Agent Decision")
    st.markdown(f"### {decision.action} — confidence {decision.confidence:.0%}")
    st.info(decision.rationale)
    st.caption("Decision fuses the PPO policy over (price, z-scored sentiment, "
               "z-scored ML forecast) windows, gated by the confidence threshold.")

with tabs[4]:
    st.subheader("Paper Trading (Alpaca)")
    st.warning("Orders go to Alpaca's PAPER endpoint. Every order passes "
               "risk circuit breakers first.", icon="⚠️")

    col_a, col_b, col_c = st.columns(3)
    do_trade = col_a.button(f"Submit {decision.action} x{qty} (paper)",
                            disabled=decision.action == "HOLD")
    show_acct = col_b.button("Refresh account")
    kill = col_c.button("🛑 KILL SWITCH — flatten all")

    try:
        from src.broker import PaperBroker, RiskGateError
        broker = PaperBroker()

        if do_trade:
            try:
                order = broker.submit(TICKER, decision.action, qty)
                st.session_state.order_log.append(order)
                st.success(f"Order accepted: {order}")
            except RiskGateError as e:
                st.error(str(e))

        if kill:
            broker.flatten_all()
            st.warning("All positions closed.")

        if show_acct or do_trade:
            acct = broker.account()
            a1, a2, a3 = st.columns(3)
            a1.metric("Equity", f"${float(acct.equity):,.2f}")
            a2.metric("Buying Power", f"${float(acct.buying_power):,.2f}")
            a3.metric("Day P&L",
                      f"${float(acct.equity) - float(acct.last_equity):+,.2f}")
    except RuntimeError as e:
        st.error(str(e))

    st.markdown("#### Session order log")
    if st.session_state.order_log:
        st.dataframe(pd.DataFrame(st.session_state.order_log),
                     use_container_width=True)
    else:
        st.info("No orders this session.")
