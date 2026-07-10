# AI Trading Decision-Support System

End-to-end pipeline: **FinBERT news sentiment → ML return forecasting
(XGBoost + LightGBM ensemble, walk-forward) → PPO reinforcement-learning
policy → Buy/Sell/Hold signal → Streamlit dashboard → risk-gated Alpaca
paper trading.**

The RL layer follows the custom-signals pattern from
[nicknochnack/Reinforcement-Learning-for-Trading-Custom-Signals](https://github.com/nicknochnack/Reinforcement-Learning-for-Trading-Custom-Signals),
with the agent's state built from this project's own sentiment and forecast
signals instead of generic technical indicators.

## Project structure

```
ai-trading-system/
├── config.py                 # settings + risk limits (secrets via .env)
├── requirements.txt
├── .env.example              # copy to .env, add your keys
├── src/
│   ├── sentiment.py          # Finnhub news -> FinBERT -> daily score
│   ├── features.py           # price load + feature engineering
│   ├── forecasting.py        # XGB/LGBM ensemble + walk-forward preds
│   ├── rl_env.py             # custom StocksEnv (price+sentiment+forecast)
│   ├── rl_train.py           # end-to-end PPO training
│   ├── signals.py            # fuse RL + forecast + sentiment -> decision
│   └── broker.py             # Alpaca connector with circuit breakers
├── app/
│   └── dashboard.py          # Streamlit dashboard
├── scripts/
│   └── run_daily.py          # scheduled daily decision + paper order
├── notebooks/
│   └── News_Sentiment_Analysis.ipynb   # original research notebook
├── models/                   # trained PPO agents (gitignored)
└── data/                     # cached frames + decision log (gitignored)
```

## Setup (VS Code / local)

```bash
git clone <your-repo>
cd ai-trading-system
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then edit .env with your keys
```

Get keys:
- **Finnhub** (news): free key at https://finnhub.io — put in `FINNHUB_API_KEY`.
  ⚠️ If your old key was ever committed in a notebook, **rotate it**.
- **Alpaca** (paper trading): free account at https://alpaca.markets →
  dashboard → "Paper Trading" → generate API keys → put in
  `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`. Keep `ALPACA_BASE_URL` as the
  paper endpoint.

## Run

**1. Train the RL agent** (also caches data for the dashboard):
```bash
python -m src.rl_train
```
This fetches prices + news, scores sentiment with FinBERT, generates
walk-forward ML forecasts (slow the first time — it refits repeatedly to
avoid leakage), trains PPO for 100k steps, evaluates on the held-out 20%,
and saves `models/ppo_AAPL.zip` + `data/rl_df_AAPL.parquet`.

**2. Launch the dashboard:**
```bash
streamlit run app/dashboard.py
```
Opens at http://localhost:8501. Tabs: Overview, Forecast, Sentiment,
RL Decision, Paper Trading (order submission + kill switch).

**3. Daily paper trading (scheduled):**
```bash
python scripts/run_daily.py
```
Schedule with cron (`30 8 * * 1-5`), Windows Task Scheduler, or an n8n
schedule node. Every decision is appended to `data/decision_log.csv`.

## How paper trading connects

`src/broker.py` wraps `alpaca-py`'s `TradingClient` pointed at
`https://paper-api.alpaca.markets`. Every order passes a **risk gate**
first: daily-loss halt, position-size cap, max-open-positions, no-short
guard. The dashboard's Paper Trading tab and `run_daily.py` both call the
same `PaperBroker.submit()` — one code path, so what you rehearse in paper
is exactly what would run live.

**Going live later** = change `ALPACA_BASE_URL` to
`https://api.alpaca.markets` and swap in live keys. Do this only after
4-6+ weeks of paper results that include at least one bad market stretch,
and start with tiny size. Nothing in this repo is financial advice.

## Databricks notes

- Upload the repo (Repos → Add Repo) and `%pip install -r requirements.txt`
  in a cluster notebook, or install libs on the cluster.
- Replace the parquet caches with Delta tables (`spark.createDataFrame(rl_df)
  .write.saveAsTable(...)`) and add MLflow tracking in `rl_train.py`
  (`mlflow.log_params`, `mlflow.log_artifact(model_path)`).
- Streamlit doesn't run inside Databricks notebooks — run the dashboard
  locally against exported data, or use Databricks Apps to host it.
- Store keys in Databricks Secrets (`dbutils.secrets.get`) instead of `.env`.

## Evaluation discipline

- The ML forecasts fed to RL training are **walk-forward** (expanding-window
  refits), not in-sample fits — in-sample predictions would teach the agent
  to trust a signal far better than it will ever be live.
- Compare the agent against buy-and-hold and the raw ML signal thresholded.
  If the RL layer doesn't beat those baselines out-of-sample, say so —
  a negative result honestly reported is more credible than an inflated one.
- Review `data/decision_log.csv` weekly during paper trading: hit rate,
  average win/loss, behavior on down days, how often circuit breakers fired.
