"""ML return forecasting (XGBoost + LightGBM ensemble).

Keeps the two models from your notebook that most reliably beat the naive
baseline (cells 21-22). SARIMAX/Prophet/GRU stay in the notebook as research;
in production you serve your best models, not all of them.

Also provides walk-forward historical predictions — this matters for RL
training: if you feed the RL agent in-sample predictions, it learns to trust
a forecast that looks far better than it will be live (leakage).
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def make_models():
    return {
        "XGBoost": XGBRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        ),
    }


def fit_ensemble(X_train, y_train):
    models = make_models()
    for m in models.values():
        m.fit(X_train, y_train)
    return models


def predict_ensemble(models, X) -> pd.Series:
    """Mean of model predictions = predicted next-day log return."""
    preds = np.column_stack([m.predict(X) for m in models.values()])
    return pd.Series(preds.mean(axis=1), index=X.index, name="ml_forecast")


def walk_forward_predictions(X, y, refit_every: int = 21, min_train: int = 252) -> pd.Series:
    """Leakage-free historical predictions for RL training.

    Expanding window: refit the ensemble every `refit_every` trading days,
    predict only days the models never saw. Days before `min_train` get NaN.
    """
    preds = pd.Series(index=X.index, dtype=float)
    i = min_train
    models = None
    while i < len(X):
        j = min(i + refit_every, len(X))
        models = fit_ensemble(X.iloc[:i], y.iloc[:i])
        preds.iloc[i:j] = predict_ensemble(models, X.iloc[i:j]).values
        i = j
    return preds
