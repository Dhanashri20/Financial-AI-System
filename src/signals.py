"""Fuse RL action + ML forecast + sentiment into the final BUY/SELL/HOLD.

gym-anytrading's StocksEnv action space is binary (0=Sell/Short, 1=Buy/Long).
We derive a 3-way decision: an action only becomes BUY/SELL when it changes
the current position; repeating the same stance = HOLD. A confidence proxy
comes from the policy's action probabilities.
"""
from dataclasses import dataclass

import numpy as np
import torch

import config


@dataclass
class Decision:
    action: str          # BUY / SELL / HOLD
    confidence: float    # policy probability of chosen action
    ml_forecast: float   # predicted next-day log return
    sentiment: float     # latest daily signed sentiment
    rationale: str = ""


def rl_action_confidence(model, obs) -> tuple[int, float]:
    """Chosen action + its probability from the PPO policy."""
    obs_t, _ = model.policy.obs_to_tensor(np.array(obs)[None])
    dist = model.policy.get_distribution(obs_t)
    probs = dist.distribution.probs.detach().cpu().numpy().ravel()
    action = int(probs.argmax())
    return action, float(probs[action])


def decide(model, obs, prev_action: int | None,
           ml_forecast: float, sentiment: float) -> Decision:
    action_id, conf = rl_action_confidence(model, obs)

    if prev_action is not None and action_id == prev_action:
        label = "HOLD"
    else:
        label = "BUY" if action_id == 1 else "SELL"

    # Confidence gate: below threshold, don't trade
    if label != "HOLD" and conf < config.MIN_RL_CONFIDENCE:
        label = "HOLD"

    rationale = (
        f"RL policy favors {'long' if action_id == 1 else 'short/flat'} "
        f"with confidence {conf:.0%}. ML ensemble forecasts next-day log return "
        f"of {ml_forecast:+.4f}; latest sentiment score {sentiment:+.2f}."
    )
    return Decision(label, conf, ml_forecast, sentiment, rationale)
