import logging
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COL = "PriceUSD_coinmetrics"

MIN_W = 1e-6
MA_WINDOW = 200
DYNAMIC_STRENGTH = 2.5

SMART_ROLL_WINDOW = 30
DISP_ROLL_WINDOW = 14

FEATS = [
    "price_vs_ma",
    "smart_odds_updates_z",
    "smart_price_std_z",
    "signal_raw",
]


def _clean_array(arr: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(arr), arr, 0.0)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(window, min_periods=max(5, window // 3)).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


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


def allocate_sequential_stable(
    raw: np.ndarray,
    n_past: int,
    locked_weights: np.ndarray | None = None,
) -> np.ndarray:
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


def _find_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_polymarket_smart_features() -> pd.DataFrame:
    """
    Build daily smart-money style features from Polymarket data:
    - smart_odds_updates: count of daily odds updates from top-volume markets
    - smart_price_std: cross-sectional price dispersion among top-volume markets
    """
    try:
        from template.prelude_template import load_polymarket_data
    except ImportError:
        from prelude_template import load_polymarket_data

    data = load_polymarket_data()
    if "markets" not in data or "odds_history" not in data:
        logging.warning("Polymarket markets/odds_history not available. Using neutral features.")
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
        logging.warning("Polymarket schema mismatch. Using neutral features.")
        return pd.DataFrame()

    # Optional: keep BTC-related markets only if question/title exists
    if question_col is not None:
        btc_mask = markets[question_col].astype(str).str.contains("bitcoin|btc", case=False, na=False)
        btc_markets = markets.loc[btc_mask].copy()
        if btc_markets.empty:
            logging.warning("No BTC-related markets found in Polymarket markets. Using neutral features.")
            return pd.DataFrame()
    else:
        btc_markets = markets.copy()

    btc_markets[volume_col] = pd.to_numeric(btc_markets[volume_col], errors="coerce").fillna(0)
    top_markets = (
        btc_markets.sort_values(volume_col, ascending=False)
        .head(100)[[market_id_col]]
        .drop_duplicates()
    )
    top_ids = set(top_markets[market_id_col].astype(str))

    odds = odds.copy()
    odds = odds[odds[odds_market_id_col].astype(str).isin(top_ids)].copy()
    if odds.empty:
        logging.warning("No odds_history rows for selected top markets. Using neutral features.")
        return pd.DataFrame()

    odds[odds_ts_col] = pd.to_datetime(odds[odds_ts_col], errors="coerce", utc=True)
    odds = odds.dropna(subset=[odds_ts_col]).copy()
    odds["date"] = odds[odds_ts_col].dt.tz_convert(None).dt.normalize()

    odds[odds_price_col] = pd.to_numeric(odds[odds_price_col], errors="coerce")
    odds = odds.dropna(subset=[odds_price_col]).copy()

    # Daily update count
    daily_updates = (
        odds.groupby("date")
        .size()
        .rename("smart_odds_updates")
        .astype(float)
    )

    # Cross-sectional daily dispersion
    daily_disp = (
        odds.groupby("date")[odds_price_col]
        .std()
        .rename("smart_price_std")
        .astype(float)
        .fillna(0.0)
    )

    # Optional extra feature if later needed
    daily_mean = (
        odds.groupby("date")[odds_price_col]
        .mean()
        .rename("smart_mean_price")
        .astype(float)
    )

    out = pd.concat([daily_updates, daily_disp, daily_mean], axis=1).sort_index()
    logging.info(
        "Built Polymarket smart features: %d rows, %s to %s",
        len(out),
        out.index.min().date() if len(out) else None,
        out.index.max().date() if len(out) else None,
    )
    return out


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    price = df[PRICE_COL].loc["2010-07-18":].copy()

    ma = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma) - 1).clip(-1, 1).fillna(0.0)

    pm = load_polymarket_smart_features()

    if not pm.empty:
        pm = pm.reindex(price.index)
        pm["smart_odds_updates"] = pm["smart_odds_updates"].fillna(0.0)
        pm["smart_price_std"] = pm["smart_price_std"].fillna(pm["smart_price_std"].median())
        pm["smart_mean_price"] = pm["smart_mean_price"].fillna(0.5)
    else:
        pm = pd.DataFrame(index=price.index)
        pm["smart_odds_updates"] = 0.0
        pm["smart_price_std"] = 0.0
        pm["smart_mean_price"] = 0.5

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
            "smart_odds_updates": pm["smart_odds_updates"],
            "smart_price_std": pm["smart_price_std"],
            "smart_mean_price": pm["smart_mean_price"],
            "smart_odds_updates_z": smart_odds_updates_z,
            "smart_price_std_z": smart_price_std_z,
        },
        index=price.index,
    )

    signal_cols = [
        "price_vs_ma",
        "smart_odds_updates_z",
        "smart_price_std_z",
    ]
    features[signal_cols] = features[signal_cols].shift(1)

    features["signal_raw"] = (
        0.55 * features["smart_odds_updates_z"].fillna(0.0)
        - 0.35 * features["smart_price_std_z"].fillna(0.0)
        - 0.25 * features["price_vs_ma"].fillna(0.0)
    )

    return features.fillna(0.0)


def compute_dynamic_multiplier(
    price_vs_ma: np.ndarray,
    smart_odds_updates_z: np.ndarray,
    smart_price_std_z: np.ndarray,
) -> np.ndarray:
    """
    Interpretation:
    - smart_odds_updates_z high  -> more market activity / regime attention -> mildly increase buying
    - smart_price_std_z high     -> more disagreement / uncertainty        -> reduce buying
    - price_vs_ma high           -> price above trend                      -> reduce buying
    """
    signal = (
        0.55 * smart_odds_updates_z
        - 0.35 * smart_price_std_z
        - 0.25 * price_vs_ma
    )

    adjustment = np.clip(signal * DYNAMIC_STRENGTH, -3, 3)
    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    price_vs_ma = _clean_array(df["price_vs_ma"].values)
    smart_odds_updates_z = _clean_array(df["smart_odds_updates_z"].values)
    smart_price_std_z = _clean_array(df["smart_price_std_z"].values)

    dyn = compute_dynamic_multiplier(
        price_vs_ma=price_vs_ma,
        smart_odds_updates_z=smart_odds_updates_z,
        smart_price_std_z=smart_price_std_z,
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
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.0 for col in features_df.columns},
            index=missing,
        )
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