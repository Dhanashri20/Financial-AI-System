"""Central configuration. Secrets come from environment variables (.env)."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Secrets (NEVER hardcode these) ---
FINNHUB_API_KEY = os.environ.get("FINNHUB_KEY")
# FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
# Paper trading endpoint. For live trading later, change to https://api.alpaca.markets
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# --- Model / data settings (mirrors your notebook CONFIG cell) ---
TICKER = "NVDA"
LOOKBACK = "3y"          # yfinance history
HORIZON = 7              # forecast days ahead
TEST_DAYS = 60           # backtest holdout
NEWS_LOOKBACK_DAYS = 55  # Finnhub news window

# --- RL settings ---
RL_WINDOW_SIZE = 12
RL_TIMESTEPS = 100_000

# --- Risk / circuit breakers (paper AND live) ---
MAX_POSITION_QTY = 10          # max shares per order
MAX_DAILY_LOSS_PCT = 2.0       # halt trading if account down >2% today
MAX_OPEN_POSITIONS = 3
MIN_RL_CONFIDENCE = 0.55       # skip trades below this confidence

# --- Paths ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TICKER = "NVDA"
HF_TOKEN = os.environ.get("HF_TOKEN")
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT")
