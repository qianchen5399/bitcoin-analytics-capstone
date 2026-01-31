import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')

# Define Paths
COINMETRICS_PATH = os.path.join(DATA_DIR, 'Coin Metrics', 'coinmetrics_btc.csv')
POLYMARKET_DIR = os.path.join(DATA_DIR, 'Polymarket')

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# Set visual style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# --- 2. Data Loading & Cleaning ---

def load_bitcoin_data(filepath):
    print(f"[INFO] Loading Bitcoin data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'])
        df['date'] = df['time'].dt.date
        return df
    except Exception as e:
        print(f"[ERROR] Loading Bitcoin data: {e}")
        return None

def load_polymarket_data(datadir):
    print(f"[INFO] Loading Polymarket data from {datadir}...")
    markets_path = os.path.join(datadir, 'finance_politics_markets.parquet')
    odds_path = os.path.join(datadir, 'finance_politics_odds_history.parquet')
    tokens_path = os.path.join(datadir, 'finance_politics_tokens.parquet') 
    
    data = {}
    try:
        # A. Load Markets
        if os.path.exists(markets_path):
            markets_df = pd.read_parquet(markets_path)
            data['markets'] = markets_df
            print(f"   [Done] Loaded {len(markets_df)} markets.")
            
        # B. Load Odds with Timestamp Fix
        if os.path.exists(odds_path):
            odds_df = pd.read_parquet(odds_path)
            
            # --- Timestamp Fix Logic ---
            if 'timestamp' in odds_df.columns:
                # Force numeric conversion first to handle mixed types
                ts_numeric = pd.to_numeric(odds_df['timestamp'], errors='coerce')
                current_max_date = pd.to_datetime(ts_numeric).max()
                
                if current_max_date.year < 1980:
                    print("   [WARNING] Detected 1970s timestamp bug. Applying fix (unit='ms')...")
                    odds_df['timestamp'] = pd.to_datetime(ts_numeric, unit='ms', utc=True)
                    print(f"   [Fixed] New max date: {odds_df['timestamp'].max()}")
                else:
                    print("   [Info] Timestamps appear correct.")
                
                odds_df['date'] = odds_df['timestamp'].dt.date
            
            data['odds'] = odds_df
            print(f"   [Done] Loaded {len(odds_df)} odds history records.")

        # C. Load Tokens
        if os.path.exists(tokens_path):
            data['tokens'] = pd.read_parquet(tokens_path)
            print(f"   [Done] Loaded tokens table.")
            
        return data if data else None
    except Exception as e:
        print(f"[ERROR] Loading Polymarket data: {e}")
        return None

# --- 3. Feature Engineering ---

def process_smart_money_features(poly_data):
    """Filters for Top 100 markets and aggregates signals."""
    print("\n[INFO] Processing 'Smart Money' features (Top 100 Markets)...")
    markets = poly_data.get('markets')
    odds = poly_data.get('odds')
    
    if markets is None or odds is None:
        print("[ERROR] Missing markets or odds data.")
        return None

    # 1. Identify Top 100 Markets
    top_100_ids = markets.nlargest(100, 'volume')['market_id']
    
    # 2. Filter Odds
    odds_smart = odds[odds['market_id'].isin(top_100_ids)]
    
    # 3. Aggregate Daily
    daily_features = (
        odds_smart
        .groupby('date')
        .agg(
            smart_odds_updates=('price', 'size'),
            smart_mean_price=('price', 'mean'),
            smart_price_std=('price', 'std')
        )
        .reset_index()
    )
    print(f"   [Done] Generated {len(daily_features)} daily feature records.")
    return daily_features

def align_datasets(btc_df, pm_daily):
    """Joins Bitcoin data with Polymarket features."""
    print("[INFO] Aligning datasets...")
    btc_daily = btc_df.set_index('date').sort_index()
    pm_daily = pm_daily.set_index('date').sort_index()
    
    aligned = btc_daily.join(pm_daily, how='inner').sort_index()
    aligned = aligned.ffill()
    print(f"   [Done] Aligned dataset shape: {aligned.shape}")
    return aligned

# --- 4. Visualization Functions ---

def plot_btc_price_log(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['PriceUSD'], label='BTC Price')
    plt.yscale('log')
    plt.title('Bitcoin Price History (Log Scale)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, '01_btc_price_log.png')
    plt.savefig(save_path)
    print(f"   [Saved] {save_path}")
    plt.close()

def plot_dual_axis(aligned_df):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color_btc = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Bitcoin Price (USD)', color=color_btc)
    ax1.plot(aligned_df.index, aligned_df['PriceUSD'], color=color_btc, alpha=0.6, label='BTC Price')
    ax1.tick_params(axis='y', labelcolor=color_btc)
    ax1.set_yscale('log')
    
    ax2 = ax1.twinx()
    color_poly = 'tab:red'
    ax2.set_ylabel('Polymarket Activity (Updates)', color=color_poly)
    ax2.plot(aligned_df.index, aligned_df['smart_odds_updates'], color=color_poly, alpha=0.5, label='Smart Money Activity')
    ax2.tick_params(axis='y', labelcolor=color_poly)
    
    plt.title('Bitcoin Price vs. "Smart Money" Prediction Activity')
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, '02_btc_vs_polymarket_activity.png')
    plt.savefig(save_path)
    print(f"   [Saved] {save_path}")
    plt.close()

def plot_correlation_matrix(aligned_df):
    cols = ['PriceUSD', 'TxCnt', 'smart_odds_updates', 'smart_price_std']
    valid_cols = [c for c in cols if c in aligned_df.columns]
    
    plt.figure(figsize=(8, 6))
    corr = aligned_df[valid_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation: BTC Metrics vs. Refined Polymarket Signals')
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, '03_correlation_matrix.png')
    plt.savefig(save_path)
    print(f"   [Saved] {save_path}")
    plt.close()

def plot_lead_lag(poly_data, btc_df):
    print("\n[INFO] Running Lead-Lag Analysis...")
    markets = poly_data['markets']
    odds = poly_data['odds']
    tokens = poly_data.get('tokens')
    
    if tokens is None:
        print("   [Skip] Tokens table missing.")
        return None

    top_id = markets.sort_values('volume', ascending=False).iloc[0]['market_id']
    question = markets[markets['market_id'] == top_id]['question'].iloc[0]
    print(f"   [Info] Analyzing King Market: {question}")
    
    market_odds = odds[odds['market_id'] == top_id].merge(
        tokens[['token_id', 'outcome']], on='token_id', how='left'
    )
    bull_odds = market_odds[market_odds['outcome'] == 'Yes'].copy()
    
    bull_daily = bull_odds.groupby('date')['price'].mean().rename('prob')
    btc_daily = btc_df.set_index('date')['PriceUSD']
    analysis_df = pd.concat([btc_daily, bull_daily], axis=1, join='inner').sort_index()
    
    btc_ret = analysis_df['PriceUSD'].pct_change()
    odds_ret = analysis_df['prob'].pct_change()
    
    lags = range(-10, 11)
    corrs = []
    
    valid_idx = btc_ret.dropna().index.intersection(odds_ret.dropna().index)
    
    for l in lags:
        c = btc_ret.loc[valid_idx].corr(odds_ret.loc[valid_idx].shift(l))
        corrs.append(c)
        
    plt.figure(figsize=(10, 5))
    colors = ['gray' if c < 0 else 'skyblue' for c in corrs]
    bars = plt.bar(lags, corrs, color=colors, edgecolor='black')
    
    max_idx = np.argmax(np.abs(corrs))
    bars[max_idx].set_color('orange')
    bars[max_idx].set_label(f'Max Corr at Lag {lags[max_idx]}')
    
    plt.axvline(0, color='red', linestyle='--', label='Zero Lag')
    plt.title(f"Lead-Lag Analysis: {question}\n(Positive Lag = Odds Lead Price)")
    plt.xlabel("Lag (Days)")
    plt.ylabel("Correlation Coefficient")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, '04_lead_lag_analysis.png')
    plt.savefig(save_path)
    print(f"   [Saved] {save_path}")
    plt.close()
    
    return analysis_df

def plot_kde_regimes(analysis_df):
    print("[INFO] Running KDE Analysis...")
    analysis_df['fwd_ret_30d'] = analysis_df['PriceUSD'].pct_change(30).shift(-30)
    kde_data = analysis_df.dropna(subset=['fwd_ret_30d', 'prob'])
    
    high_sent = kde_data[kde_data['prob'] > 0.60]['fwd_ret_30d']
    low_sent = kde_data[kde_data['prob'] <= 0.60]['fwd_ret_30d']
    
    plt.figure(figsize=(10, 5))
    sns.kdeplot(high_sent, fill=True, color='forestgreen', label='High Sentiment (>60%)', alpha=0.3)
    sns.kdeplot(low_sent, fill=True, color='gray', label='Low/Normal Sentiment', alpha=0.3)
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Bitcoin 30-Day Forward Returns by Prediction Market Sentiment')
    plt.xlabel('30-Day Forward Return')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, '05_kde_regimes.png')
    plt.savefig(save_path)
    print(f"   [Saved] {save_path}")
    plt.close()

# --- 5. Main Execution ---
def main():
    print("--- Starting EDA Pipeline ---")
    
    # 1. Load Data
    btc_df = load_bitcoin_data(COINMETRICS_PATH)
    poly_data = load_polymarket_data(POLYMARKET_DIR)

    if btc_df is None or poly_data is None:
        print("[ERROR] Data loading failed. Check paths.")
        return

    # 2. Basic Plots
    plot_btc_price_log(btc_df)
    
    # 3. Feature Engineering & Alignment
    pm_daily_features = process_smart_money_features(poly_data)
    
    if pm_daily_features is not None:
        aligned_df = align_datasets(btc_df, pm_daily_features)
        
        # 4. Advanced Plots
        plot_dual_axis(aligned_df)
        plot_correlation_matrix(aligned_df)
    
    # 5. King Market Analysis
    if 'tokens' in poly_data:
        analysis_df = plot_lead_lag(poly_data, btc_df)
        if analysis_df is not None:
            plot_kde_regimes(analysis_df)

    print(f"\n[SUCCESS] Pipeline Complete. All visualizations saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()