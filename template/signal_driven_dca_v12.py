import logging
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COL = "PriceUSD_coinmetrics"

MIN_W = 1e-6
MA_WINDOW = 200
DYNAMIC_STRENGTH = 1.5   

SMART_ROLL_WINDOW = 30
DISP_ROLL_WINDOW = 14


# =========================
# Utils
# =========================
def _clean_array(arr: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(arr), arr, 0.0)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(window, min_periods=max(5, window // 3)).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================
# Stable Allocation (unchanged)
# =========================
def _compute_stable_signal(raw: np.ndarray) -> np.ndarray:
    n = len(raw)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    cumsum = np.cumsum(raw)
    running_mean = cumsum / np.arange(1, n + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        signal = raw / running_mean
    return np.where(np.isfinite(signal), signal, 1.0)


def allocate_sequential_stable(raw, n_past, locked_weights=None):
    n = len(raw)
    if n == 0:
        return np.array([])
    if n_past <= 0:
        return np.full(n, 1.0 / n)

    n_past = min(n_past, n)
    w = np.zeros(n)
    base_weight = 1.0 / n

    if locked_weights is not None and len(locked_weights) >= n_past:
        w[:n_past] = locked_weights[:n_past]
    else:
        for i in range(n_past):
            signal = _compute_stable_signal(raw[: i + 1])[-1]
            w[i] = signal * base_weight

    past_sum = w[:n_past].sum()
    target_budget = n_past / n
    if past_sum > target_budget + 1e-10:
        w[:n_past] *= target_budget / past_sum

    n_future = n - n_past
    if n_future > 1:
        w[n_past : n - 1] = base_weight

    w[n - 1] = max(1.0 - w[: n - 1].sum(), 0.0)
    return w


# =========================
# Polymarket Features
# =========================
def _find_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_polymarket_smart_features():
    try:
        from template.prelude_template import load_polymarket_data
    except ImportError:
        from prelude_template import load_polymarket_data

    data = load_polymarket_data()
    if "markets" not in data or "odds_history" not in data:
        return pd.DataFrame()

    markets = data["markets"].copy()
    odds = data["odds_history"].copy()

    market_id_col = _find_first_existing_col(markets, ["market_id", "id"])
    volume_col = _find_first_existing_col(markets, ["volume", "volume_num"])
    odds_market_id_col = _find_first_existing_col(odds, ["market_id", "id"])
    odds_ts_col = _find_first_existing_col(odds, ["timestamp", "time"])
    odds_price_col = _find_first_existing_col(odds, ["price", "probability"])

    if not all([market_id_col, volume_col, odds_market_id_col, odds_ts_col, odds_price_col]):
        return pd.DataFrame()

    markets[volume_col] = pd.to_numeric(markets[volume_col], errors="coerce").fillna(0)
    top_ids = set(
        markets.sort_values(volume_col, ascending=False)
        .head(100)[market_id_col]
        .astype(str)
    )

    odds = odds[odds[odds_market_id_col].astype(str).isin(top_ids)].copy()
    odds[odds_ts_col] = pd.to_datetime(odds[odds_ts_col], utc=True, errors="coerce")
    odds = odds.dropna(subset=[odds_ts_col])
    odds["date"] = odds[odds_ts_col].dt.tz_convert(None).dt.normalize()

    odds[odds_price_col] = pd.to_numeric(odds[odds_price_col], errors="coerce")
    odds = odds.dropna(subset=[odds_price_col])

    daily_updates = odds.groupby("date").size().rename("smart_odds_updates")
    daily_disp = odds.groupby("date")[odds_price_col].std().rename("smart_price_std")

    return pd.concat([daily_updates, daily_disp], axis=1)


# =========================
# Feature Engineering
# =========================
def precompute_features(df):

    price = df[PRICE_COL].loc["2010-07-18":].copy()

    ma = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    price_vs_ma = ((price / ma) - 1).clip(-1, 1).fillna(0.0)

    pm = load_polymarket_smart_features()

    pm = pm.reindex(price.index).fillna(0.0)

    activity_z = _rolling_zscore(np.log1p(pm["smart_odds_updates"]), SMART_ROLL_WINDOW).clip(-3, 3)
    dispersion_z = _rolling_zscore(pm["smart_price_std"], DISP_ROLL_WINDOW).clip(-3, 3)

    features = pd.DataFrame({
        PRICE_COL: price,
        "price_vs_ma": price_vs_ma,
        "activity_z": activity_z,
        "dispersion_z": dispersion_z,
    })

    # 🔥 防 leakage
    features = features.shift(1).fillna(0.0)

    # 🔥 CLEAN SIGNAL（唯一定义）
    features["signal_raw"] = (
        0.5 * features["activity_z"]
        - 0.3 * features["dispersion_z"]
        - 0.2 * features["price_vs_ma"]
    )

    return features


# =========================
# Multiplier（稳定版）
# =========================
def compute_dynamic_multiplier(signal):

    signal = np.clip(signal, -2, 2)

    adjustment = np.tanh(signal * DYNAMIC_STRENGTH)

    multiplier = 1.0 + adjustment

    return np.where(np.isfinite(multiplier), multiplier, 1.0)


# =========================
# Weight Computation
# =========================
def compute_weights_fast(features_df, start_date, end_date, n_past=None, locked_weights=None):

    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    signal = _clean_array(df["signal_raw"].values)

    dyn = compute_dynamic_multiplier(signal)

    raw = base * dyn

    if n_past is None:
        n_past = n

    weights = allocate_sequential_stable(raw, n_past, locked_weights)

    return pd.Series(weights, index=df.index)


def compute_window_weights(features_df, start_date, end_date, current_date, locked_weights=None):

    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame({col: 0.0 for col in features_df.columns}, index=missing)
        features_df = pd.concat([features_df, placeholder]).sort_index()

    past_end = min(current_date, end_date)

    if start_date <= past_end:
        n_past = len(pd.date_range(start=start_date, end=past_end))
    else:
        n_past = 0

    weights = compute_weights_fast(features_df, start_date, end_date, n_past, locked_weights)

    return weights.reindex(full_range, fill_value=0.0)