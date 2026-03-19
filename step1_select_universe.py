import pandas as pd
import numpy as np
import os

def load_all_data(data_dir='data/raw/'):
    data = {}
    for f in os.listdir(data_dir):
        if f.endswith('_1h.csv'):
            symbol = f.replace('_1h.csv', '')
            try:
                df = pd.read_csv(os.path.join(data_dir, f),
                                 index_col='timestamp', parse_dates=True)
                data[symbol] = df
            except Exception as e:
                print(f"  Failed to load {symbol}: {e}")
    print(f"Loaded {len(data)} coins from {data_dir}\n")
    return data


def compute_stats(data):
    stats = []
    for symbol, df in data.items():
        if len(df) < 500:
            continue
        returns = df['close'].pct_change().dropna()
        dollar_vol = (df['close'] * df['volume']).resample('D').sum().mean()
        ann_vol = returns.std() * np.sqrt(24 * 365)
        total_ret = df['close'].iloc[-1] / df['close'].iloc[0] - 1
        zero_vol = (df['volume'] == 0).mean()

        stats.append({
            'symbol': symbol,
            'num_candles': len(df),
            'avg_daily_volume_usd': dollar_vol,
            'annualized_vol': ann_vol,
            'total_return': total_ret,
            'zero_volume_pct': zero_vol,
            'start_date': str(df.index[0].date()),
            'end_date': str(df.index[-1].date()),
        })
    return pd.DataFrame(stats).sort_values('avg_daily_volume_usd', ascending=False)


def select_universe(stats, min_candles=7000, min_vol=0.30, max_vol=2.50,
                    max_zero_vol=0.05, target_size=25):
    f = stats.copy()
    print(f"Starting: {len(f)} coins\n")

    # Filter 1: enough history (prevents data alignment destroying months of data)
    f = f[f['num_candles'] >= min_candles]
    print(f"After min candles >= {min_candles}: {len(f)}")

    # Filter 2: must move enough for long-only momentum to work
    f = f[f['annualized_vol'] >= min_vol]
    print(f"After ann vol >= {min_vol:.0%}: {len(f)}")

    # Filter 3: not so crazy that price is pure noise
    f = f[f['annualized_vol'] <= max_vol]
    print(f"After ann vol <= {max_vol:.0%}: {len(f)}")

    # Filter 4: not dead
    f = f[f['zero_volume_pct'] <= max_zero_vol]
    print(f"After zero-vol <= {max_zero_vol:.0%}: {len(f)}")

    # RANK: ascending=True so HIGHEST value gets HIGHEST rank number
    # Higher rank = better (more liquid / more volatile)
    f['volume_rank'] = f['avg_daily_volume_usd'].rank(ascending=True, pct=True)
    f['vol_rank'] = f['annualized_vol'].rank(ascending=True, pct=True)
    f['composite'] = 0.60 * f['volume_rank'] + 0.40 * f['vol_rank']

    # Sort: highest composite first = best coins
    f = f.sort_values('composite', ascending=False)
    universe = f.head(target_size)['symbol'].tolist()

    print(f"\n{'='*80}")
    print(f"SELECTED UNIVERSE: {len(universe)} coins")
    print(f"{'='*80}")
    for i, s in enumerate(universe, 1):
        r = f[f['symbol'] == s].iloc[0]
        print(f"  {i:2d}. {s:12s}  "
              f"vol=${r['avg_daily_volume_usd']/1e6:>8.1f}M  "
              f"ann_vol={r['annualized_vol']:>6.1%}  "
              f"ret={r['total_return']:>8.1%}  "
              f"candles={int(r['num_candles']):>5d}  "
              f"composite={r['composite']:.3f}")
    print(f"{'='*80}\n")

    # Sanity check: BTC, ETH, SOL must be present
    for must_have in ['BTC', 'ETH', 'SOL']:
        if must_have in universe:
            print(f"  CHECK: {must_have} is in universe")
        else:
            print(f"  WARNING: {must_have} is MISSING from universe!")

    return universe, f


if __name__ == "__main__":
    data = load_all_data()
    stats = compute_stats(data)

    os.makedirs('data', exist_ok=True)
    os.makedirs('analysis', exist_ok=True)

    stats.to_csv('analysis/universe_stats_all.csv', index=False)
    print(f"Saved all stats to analysis/universe_stats_all.csv\n")

    universe, ranked = select_universe(stats, target_size=25)

    with open('data/universe.txt', 'w') as f:
        f.write('\n'.join(universe))
    print(f"Saved universe to data/universe.txt")

    ranked.head(30).to_csv('analysis/universe_stats_ranked.csv', index=False)
    print(f"Saved ranked stats to analysis/universe_stats_ranked.csv")