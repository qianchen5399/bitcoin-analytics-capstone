"""Dynamic DCA weight computation using:
- 200-day MA as the primary signal
- Conditional Polymarket overlay only when MA is weak/uncertain
- BTC volatility dampening as risk control

Design goals:
- Preserve the strong baseline behavior from the template
- Only allow Polymarket to influence allocation when the MA signal is ambiguous
- Prevent look-ahead bias by lagging all signal inputs by 1 day
- Preserve template compatibility:
    - precompute_features(df)
    - compute_window_weights(features_df, start_date, end_date, current_date, ...)
"""

import logging
import numpy as np
import pandas as pd

# =============================================================================
# Constants
# =============================================================================

PRICE_COL = "PriceUSD_coinmetrics"

MIN_W = 1e-6
MA_WINDOW = 200
DYNAMIC_STRENGTH = 2.0  # keep close to template baseline

SMART_ROLL_WINDOW = 30
DISP_ROLL_WINDOW = 14

# BTC volatility settings
VOL_WINDOW = 14
VOL_NORM_WINDOW = 90
VOL_THRESHOLD = 0.80
VOL_MAX_DAMPEN = 0.20

# Polymarket thresholds
ACTIVITY_THRESHOLD = 1.0
DISPERSION_THRESHOLD = 1.0

# MA ambiguity gating
# PM gets most influence when |price_vs_ma| is small.
# As |price_vs_ma| grows, PM influence fades out.
MA_AMBIGUITY_SCALE = 0.12

FEATS = [
    "price_vs_ma",
    "smart_odds_updates_z",
    "smart_price_std_z",
    "btc_volatility_pct",
    "pm_active",
    "pm_gate",
    "signal_raw",
]


# =============================================================================
# Generic Helpers
# =============================================================================


def _clean_array(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0."""
    return np.where(np.isfinite(arr), arr, 0.0)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score with conservative min periods."""
    min_periods = max(5, window // 3)
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rolling_percentile(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling percentile rank of current value within recent history."""
    if min_periods is None:
        min_periods = max(10, window // 3)

    def _pct(x: pd.Series) -> float:
        if len(x) <= 1:
            return 0.5
        return (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1)

    return series.rolling(window, min_periods=min_periods).apply(_pct, raw=False).fillna(0.5)


def _find_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =============================================================================
# Stable Allocation Helpers
# =============================================================================


def _compute_stable_signal(raw: np.ndarray) -> np.ndarray:
    """Compute stable signal weights using cumulative mean normalization."""
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


def allocate_sequential_stable(
    raw: np.ndarray,
    n_past: int,
    locked_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Allocate weights with lock-on-compute stability."""
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

    w = np.maximum(w, 0.0)
    total = w.sum()
    if total > 0:
        w = w / total
    else:
        w = np.full(n, 1.0 / n)

    return w


# =============================================================================
# Polymarket Feature Loading
# =============================================================================


def load_polymarket_smart_features() -> pd.DataFrame:
    """Build daily Polymarket smart-money features.

    Output columns:
    - smart_odds_updates: daily count of odds updates from top-volume BTC-related markets
    - smart_price_std: cross-sectional daily price dispersion
    - smart_mean_price: cross-sectional daily mean price
    - pm_active: 1.0 when PM activity exists on that date, else 0.0
    """
    try:
        from template.prelude_template import load_polymarket_data
    except ImportError:
        from prelude_template import load_polymarket_data

    data = load_polymarket_data()
    if "markets" not in data or "odds_history" not in data:
        logging.warning(
            "Polymarket markets/odds_history not available. Using neutral PM features."
        )
        return pd.DataFrame()

    markets = data["markets"].copy()
    odds = data["odds_history"].copy()

    market_id_col = _find_first_existing_col(markets, ["market_id", "id"])
    volume_col = _find_first_existing_col(markets, ["volume", "volume_num", "liquidity"])
    question_col = _find_first_existing_col(markets, ["question", "title"])

    odds_market_id_col = _find_first_existing_col(odds, ["market_id", "id"])
    odds_ts_col = _find_first_existing_col(odds, ["timestamp", "ts", "time"])
    odds_price_col = _find_first_existing_col(odds, ["price", "probability", "mid_price"])

    if not all([market_id_col, volume_col, odds_market_id_col, odds_ts_col, odds_price_col]):
        logging.warning("Polymarket schema mismatch. Using neutral PM features.")
        return pd.DataFrame()

    if question_col is not None:
        btc_mask = markets[question_col].astype(str).str.contains(
            "bitcoin|btc", case=False, na=False
        )
        markets = markets.loc[btc_mask].copy()

    if markets.empty:
        logging.warning("No BTC-related Polymarket markets found. Using neutral PM features.")
        return pd.DataFrame()

    markets[volume_col] = pd.to_numeric(markets[volume_col], errors="coerce").fillna(0.0)

    top_markets = (
        markets.sort_values(volume_col, ascending=False)
        .head(100)[[market_id_col]]
        .drop_duplicates()
    )
    top_ids = set(top_markets[market_id_col].astype(str))

    odds = odds[odds[odds_market_id_col].astype(str).isin(top_ids)].copy()
    if odds.empty:
        logging.warning("No odds rows for selected top Polymarket markets.")
        return pd.DataFrame()

    odds[odds_ts_col] = pd.to_datetime(odds[odds_ts_col], errors="coerce", utc=True)
    odds = odds.dropna(subset=[odds_ts_col]).copy()
    odds["date"] = odds[odds_ts_col].dt.tz_convert(None).dt.normalize()

    odds[odds_price_col] = pd.to_numeric(odds[odds_price_col], errors="coerce")
    odds = odds.dropna(subset=[odds_price_col]).copy()

    if odds.empty:
        logging.warning("No valid Polymarket price rows after cleaning.")
        return pd.DataFrame()

    daily_updates = odds.groupby("date").size().rename("smart_odds_updates").astype(float)
    daily_disp = (
        odds.groupby("date")[odds_price_col]
        .std()
        .rename("smart_price_std")
        .astype(float)
        .fillna(0.0)
    )
    daily_mean = (
        odds.groupby("date")[odds_price_col]
        .mean()
        .rename("smart_mean_price")
        .astype(float)
        .fillna(0.5)
    )

    out = pd.concat([daily_updates, daily_disp, daily_mean], axis=1).sort_index()
    out["pm_active"] = (out["smart_odds_updates"] > 0).astype(float)

    if not out.empty:
        logging.info(
            "Built Polymarket smart features: %d rows, %s to %s",
            len(out),
            out.index.min().date(),
            out.index.max().date(),
        )

    return out


# =============================================================================
# Feature Engineering
# =============================================================================


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute model features for weight calculation."""
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    price = df[PRICE_COL].loc["2010-07-18":].copy()

    # Baseline MA feature
    ma = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma) - 1).clip(-1, 1).fillna(0.0)

    # BTC volatility feature
    returns = price.pct_change()
    btc_volatility = returns.rolling(VOL_WINDOW, min_periods=max(5, VOL_WINDOW // 2)).std()
    btc_volatility_pct = _rolling_percentile(
        btc_volatility,
        VOL_NORM_WINDOW,
        min_periods=max(10, VOL_NORM_WINDOW // 3),
    ).clip(0.0, 1.0)

    # Load PM features
    pm = load_polymarket_smart_features()

    if not pm.empty:
        pm = pm.reindex(price.index)
        pm["smart_odds_updates"] = pm["smart_odds_updates"].fillna(0.0)
        pm["smart_price_std"] = pm["smart_price_std"].fillna(0.0)
        pm["smart_mean_price"] = pm["smart_mean_price"].fillna(0.5)
        pm["pm_active"] = pm["pm_active"].fillna(0.0)
    else:
        pm = pd.DataFrame(index=price.index)
        pm["smart_odds_updates"] = 0.0
        pm["smart_price_std"] = 0.0
        pm["smart_mean_price"] = 0.5
        pm["pm_active"] = 0.0

    smart_odds_updates_z = _rolling_zscore(
        np.log1p(pm["smart_odds_updates"]),
        SMART_ROLL_WINDOW,
    ).clip(-3, 3)

    smart_price_std_z = _rolling_zscore(
        pm["smart_price_std"],
        DISP_ROLL_WINDOW,
    ).clip(-3, 3)

    features = pd.DataFrame(
        {
            PRICE_COL: price,
            "price_ma": ma,
            "price_vs_ma": price_vs_ma,
            "btc_volatility": btc_volatility.fillna(0.0),
            "btc_volatility_pct": btc_volatility_pct.fillna(0.5),
            "smart_odds_updates": pm["smart_odds_updates"],
            "smart_price_std": pm["smart_price_std"],
            "smart_mean_price": pm["smart_mean_price"],
            "smart_odds_updates_z": smart_odds_updates_z,
            "smart_price_std_z": smart_price_std_z,
            "pm_active": pm["pm_active"],
        },
        index=price.index,
    )

    # Lag all signal inputs by 1 day
    signal_cols = [
        "price_vs_ma",
        "btc_volatility_pct",
        "smart_odds_updates_z",
        "smart_price_std_z",
        "pm_active",
    ]
    features[signal_cols] = features[signal_cols].shift(1)

    # PM gate: strongest when MA is ambiguous, weakest when MA is strong
    abs_ma = features["price_vs_ma"].abs().fillna(0.0)
    features["pm_gate"] = 1.0 - np.clip(abs_ma / MA_AMBIGUITY_SCALE, 0.0, 1.0)

    # Debug combined signal
    act_boost = np.where(
        features["smart_odds_updates_z"].abs() > ACTIVITY_THRESHOLD,
        np.clip(features["smart_odds_updates_z"], -2.0, 2.0),
        0.0,
    )

    disp_penalty = np.where(
        features["smart_price_std_z"] > DISPERSION_THRESHOLD,
        np.clip(features["smart_price_std_z"], 0.0, 2.0),
        0.0,
    )

    features["signal_raw"] = (
        -1.00 * features["price_vs_ma"].fillna(0.0)
        + features["pm_gate"].fillna(0.0)
        * features["pm_active"].fillna(0.0)
        * (0.10 * act_boost - 0.14 * disp_penalty)
    )

    return features.fillna(0.0)


# =============================================================================
# Dynamic Multiplier
# =============================================================================


def compute_dynamic_multiplier(
    price_vs_ma: np.ndarray,
    smart_odds_updates_z: np.ndarray,
    smart_price_std_z: np.ndarray,
    btc_volatility_pct: np.ndarray | None = None,
    pm_active: np.ndarray | None = None,
    pm_gate: np.ndarray | None = None,
) -> np.ndarray:
    """Compute weight multiplier.

    Logic:
    - MA remains the main directional driver
    - PM only influences allocation when MA is weak/ambiguous
    - PM activity helps only if unusually strong
    - PM dispersion penalizes only if unusually high
    - High BTC volatility dampens the final signal
    """
    base_signal = -price_vs_ma

    if pm_active is None:
        pm_active = np.ones_like(price_vs_ma)
    if pm_gate is None:
        pm_gate = np.ones_like(price_vs_ma)
    if btc_volatility_pct is None:
        btc_volatility_pct = np.full_like(price_vs_ma, 0.5)

    activity_boost = np.where(
        np.abs(smart_odds_updates_z) > ACTIVITY_THRESHOLD,
        np.clip(smart_odds_updates_z, -2.0, 2.0),
        0.0,
    )

    dispersion_penalty = np.where(
        smart_price_std_z > DISPERSION_THRESHOLD,
        np.clip(smart_price_std_z, 0.0, 2.0),
        0.0,
    )

    # PM overlay only as a small conditional modifier
    pm_overlay = 0.10 * activity_boost - 0.14 * dispersion_penalty
    pm_overlay = np.clip(pm_overlay, -0.22, 0.10)

    signal = base_signal + pm_active * pm_gate * pm_overlay

    # Volatility dampening
    vol_damp = np.where(
        btc_volatility_pct > VOL_THRESHOLD,
        1.0
        - VOL_MAX_DAMPEN
        * (btc_volatility_pct - VOL_THRESHOLD)
        / max(1.0 - VOL_THRESHOLD, 1e-6),
        1.0,
    )
    vol_damp = np.clip(vol_damp, 1.0 - VOL_MAX_DAMPEN, 1.0)

    signal = signal * vol_damp

    adjustment = np.clip(signal * DYNAMIC_STRENGTH, -3, 3)
    multiplier = np.exp(adjustment)

    return np.where(np.isfinite(multiplier), multiplier, 1.0)


# =============================================================================
# Weight Computation API
# =============================================================================


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date window using precomputed features."""
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    price_vs_ma = _clean_array(df["price_vs_ma"].values)
    smart_odds_updates_z = _clean_array(df["smart_odds_updates_z"].values)
    smart_price_std_z = _clean_array(df["smart_price_std_z"].values)
    btc_volatility_pct = _clean_array(df["btc_volatility_pct"].values)
    pm_active = _clean_array(df["pm_active"].values)
    pm_gate = _clean_array(df["pm_gate"].values)

    dyn = compute_dynamic_multiplier(
        price_vs_ma=price_vs_ma,
        smart_odds_updates_z=smart_odds_updates_z,
        smart_price_std_z=smart_price_std_z,
        btc_volatility_pct=btc_volatility_pct,
        pm_active=pm_active,
        pm_gate=pm_gate,
    )
    raw = base * dyn

    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past, locked_weights)

    return pd.Series(weights, index=df.index)


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date range with lock-on-compute stability."""
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.0 for col in features_df.columns},
            index=missing,
        )
        if "pm_active" in placeholder.columns:
            placeholder["pm_active"] = 0.0
        if "pm_gate" in placeholder.columns:
            placeholder["pm_gate"] = 0.0
        if "btc_volatility_pct" in placeholder.columns:
            placeholder["btc_volatility_pct"] = 0.5
        features_df = pd.concat([features_df, placeholder]).sort_index()

    past_end = min(current_date, end_date)
    if start_date <= past_end:
        n_past = len(pd.date_range(start=start_date, end=past_end, freq="D"))
    else:
        n_past = 0

    weights = compute_weights_fast(
        features_df,
        start_date,
        end_date,
        n_past,
        locked_weights,
    )
    return weights.reindex(full_range, fill_value=0.0)