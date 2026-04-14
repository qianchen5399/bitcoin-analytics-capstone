import logging
import numpy as np
import pandas as pd

PRICE_COL = "PriceUSD_coinmetrics"

MIN_W = 1e-6
MA_WINDOW = 200
DYNAMIC_STRENGTH = 2.2

SMART_ROLL_WINDOW = 30
DISP_ROLL_WINDOW = 14

VOL_WINDOW = 14
VOL_NORM_WINDOW = 90
VOL_THRESHOLD = 0.80
VOL_MAX_DAMPEN = 0.20

MA_AMBIGUITY_SCALE = 0.12


# =========================
# Utilities
# =========================
def _clean_array(arr: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(arr), arr, 0.0)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    min_periods = max(5, window // 3)
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    def _pct(x: pd.Series) -> float:
        if len(x) <= 1:
            return 0.5
        return (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1)

    return (
        series.rolling(window, min_periods=max(10, window // 3))
        .apply(_pct, raw=False)
        .fillna(0.5)
    )


def _find_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =========================
# Stable Allocation
# =========================
def _compute_stable_signal(raw: np.ndarray) -> np.ndarray:
    n = len(raw)
    if n <= 1:
        return np.ones(n)

    cumsum = np.cumsum(raw)
    running_mean = cumsum / np.arange(1, n + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        signal = raw / running_mean

    return np.where(np.isfinite(signal), signal, 1.0)


def allocate_sequential_stable(raw, n_past, locked_weights=None):
    n = len(raw)
    if n == 0:
        return np.array([])

    base = 1.0 / n
    w = np.zeros(n)

    if n_past <= 0:
        return np.full(n, base)

    n_past = min(n_past, n)

    if locked_weights is not None and len(locked_weights) >= n_past:
        w[:n_past] = locked_weights[:n_past]
    else:
        for i in range(n_past):
            signal = _compute_stable_signal(raw[: i + 1])[-1]
            w[i] = signal * base

    past_sum = w[:n_past].sum()
    target = n_past / n

    if past_sum > target:
        w[:n_past] *= target / past_sum

    if n - n_past > 1:
        w[n_past : n - 1] = base

    w[-1] = max(1.0 - w[:-1].sum(), 0.0)

    total = w.sum()
    return w / total if total > 0 else np.full(n, base)


# =========================
# Polymarket Features
# =========================
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

    updates = odds.groupby("date").size().rename("smart_odds_updates")
    dispersion = odds.groupby("date")[odds_price_col].std().rename("smart_price_std")

    return pd.concat([updates, dispersion], axis=1)


# =========================
# Feature Engineering
# =========================
def precompute_features(df):
    price = df[PRICE_COL].loc["2010-07-18":].copy()

    ma = price.rolling(MA_WINDOW, min_periods=100).mean()
    price_vs_ma = ((price / ma) - 1).clip(-1, 1).fillna(0.0)

    returns = price.pct_change(fill_method=None)
    vol = returns.rolling(VOL_WINDOW, min_periods=max(5, VOL_WINDOW // 2)).std()
    vol_pct = _rolling_percentile(vol, VOL_NORM_WINDOW)

    pm = load_polymarket_smart_features().reindex(price.index).fillna(0.0)

    activity_z = _rolling_zscore(np.log1p(pm["smart_odds_updates"]), SMART_ROLL_WINDOW)
    dispersion_z = _rolling_zscore(pm["smart_price_std"], DISP_ROLL_WINDOW)

    features = pd.DataFrame(
        {
            PRICE_COL: price,
            "price_vs_ma": price_vs_ma,
            "activity_z": activity_z,
            "dispersion_z": dispersion_z,
            "vol_pct": vol_pct,
        },
        index=price.index,
    )

    # forward-chaining safe
    features = features.shift(1).fillna(0.0)

    abs_ma = features["price_vs_ma"].abs()
    features["pm_gate"] = 1.0 - np.clip(abs_ma / MA_AMBIGUITY_SCALE, 0.0, 1.0)

    activity_pct = _rolling_percentile(features["activity_z"], 60)
    dispersion_pct = _rolling_percentile(features["dispersion_z"], 60)

    # V5: slightly stronger PM overlay
    pm_overlay = 0.35 * (activity_pct - 0.5) - 0.08 * (dispersion_pct - 0.5)

    features["signal_raw"] = (
        -1.0 * features["price_vs_ma"] + features["pm_gate"] * pm_overlay
    )

    return features.fillna(0.0)


# =========================
# Multiplier
# =========================
def compute_dynamic_multiplier(signal, vol_pct):
    signal = np.clip(signal, -2.0, 2.0)

    vol_damp = np.where(
        vol_pct > VOL_THRESHOLD,
        1.0 - VOL_MAX_DAMPEN * (vol_pct - VOL_THRESHOLD) / (1.0 - VOL_THRESHOLD),
        1.0,
    )
    vol_damp = np.clip(vol_damp, 0.8, 1.0)

    signal = signal * vol_damp

    adjustment = np.clip(signal * DYNAMIC_STRENGTH, -3.0, 3.0)
    multiplier = np.exp(adjustment)

    return np.where(np.isfinite(multiplier), multiplier, 1.0)


# =========================
# Weight Computation
# =========================
def compute_weights_fast(
    features_df,
    start_date,
    end_date,
    n_past=None,
    locked_weights=None,
):
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    signal = _clean_array(df["signal_raw"].values)
    vol_pct = _clean_array(df["vol_pct"].values)

    dyn = compute_dynamic_multiplier(signal, vol_pct)
    raw = base * dyn

    if n_past is None:
        n_past = n

    weights = allocate_sequential_stable(raw, n_past, locked_weights)
    return pd.Series(weights, index=df.index)


def compute_window_weights(
    features_df,
    start_date,
    end_date,
    current_date,
    locked_weights=None,
):
    full_range = pd.date_range(start=start_date, end=end_date)

    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        filler = pd.DataFrame(0.0, index=missing, columns=features_df.columns)
        if "vol_pct" in filler.columns:
            filler["vol_pct"] = 0.5
        features_df = pd.concat([features_df, filler]).sort_index()

    past_end = min(current_date, end_date)
    n_past = (
        len(pd.date_range(start=start_date, end=past_end))
        if start_date <= past_end
        else 0
    )

    weights = compute_weights_fast(
        features_df,
        start_date,
        end_date,
        n_past,
        locked_weights,
    )

    return weights.reindex(full_range, fill_value=0.0)