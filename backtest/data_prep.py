import pandas as pd
import numpy as np
import os


def prepare_backtest_data(universe, data_dir='data/raw/'):
    close_prices = {}
    volumes = {}
    highs = {}
    lows = {}

    loaded = []
    skipped = []

    for symbol in universe:
        filepath = os.path.join(data_dir, f"{symbol}_1h.csv")
        if not os.path.exists(filepath):
            skipped.append(symbol)
            continue
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        if len(df) < 500:
            skipped.append(symbol)
            continue
        close_prices[symbol] = df['close']
        volumes[symbol] = df['volume'] * df['close']
        highs[symbol] = df['high']
        lows[symbol] = df['low']
        loaded.append(symbol)

    if skipped:
        print(f"  SKIPPED (missing/short): {skipped}")
    print(f"  Loaded: {len(loaded)} coins")

    close_df = pd.DataFrame(close_prices)
    volume_df = pd.DataFrame(volumes)
    high_df = pd.DataFrame(highs)
    low_df = pd.DataFrame(lows)

    # Forward-fill small gaps (up to 3 hours)
    close_df = close_df.ffill(limit=3)
    volume_df = volume_df.ffill(limit=3)
    high_df = high_df.ffill(limit=3)
    low_df = low_df.ffill(limit=3)

    # Count rows before dropping
    rows_before = len(close_df)

    # Single valid mask across all DataFrames
    valid = (close_df.notna().all(axis=1) &
             volume_df.notna().all(axis=1) &
             high_df.notna().all(axis=1) &
             low_df.notna().all(axis=1))

    close_df = close_df[valid]
    volume_df = volume_df[valid]
    high_df = high_df[valid]
    low_df = low_df[valid]

    rows_after = len(close_df)
    pct_kept = rows_after / rows_before * 100 if rows_before > 0 else 0

    print(f"  Aligned: {rows_after} timestamps ({pct_kept:.1f}% of {rows_before})")
    print(f"  Range: {close_df.index[0]} to {close_df.index[-1]}")

    if pct_kept < 80:
        print(f"  WARNING: Lost {100-pct_kept:.1f}% of data during alignment!")
        print(f"  Check which coin has the shortest history.")

    returns_df = close_df.pct_change()

    return {
        'close': close_df,
        'volume': volume_df,
        'high': high_df,
        'low': low_df,
        'returns': returns_df,
    }