# analysis/universe_selection.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def load_all_data(data_dir: str = 'data/raw/') -> dict:
    """Load all downloaded CSVs into a dict of DataFrames."""
    data = {}
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory {data_dir} does not exist!")
        return data
    
    files = [f for f in os.listdir(data_dir) if f.endswith('_1h.csv')]
    
    print(f"📂 Found {len(files)} CSV files in {data_dir}")
    
    for f in files:
        symbol = f.replace('_1h.csv', '')
        try:
            df = pd.read_csv(
                os.path.join(data_dir, f), 
                index_col='timestamp', 
                parse_dates=True
            )
            data[symbol] = df
        except Exception as e:
            print(f"  ⚠️  Failed to load {symbol}: {e}")
    
    print(f"✅ Loaded {len(data)} coins successfully\n")
    return data


def compute_universe_stats(data: dict) -> pd.DataFrame:
    """Compute key statistics for universe filtering."""
    stats = []
    
    print("📊 Computing statistics for each coin...\n")
    
    for symbol, df in data.items():
        if len(df) < 2000:  # Need at least ~83 days of hourly data
            print(f"  ⚠️  {symbol}: Only {len(df)} candles, skipping")
            continue
        
        # Hourly returns
        returns = df['close'].pct_change().dropna()
        
        # Daily volume (in USD/USDT)
        df['dollar_volume'] = df['close'] * df['volume']
        avg_daily_volume = df['dollar_volume'].resample('D').sum().mean()
        
        # Volatility (annualized from hourly)
        hourly_vol = returns.std()
        annualized_vol = hourly_vol * np.sqrt(24 * 365)
        
        # Average daily range (high-low as % of close)
        df['daily_range_pct'] = (df['high'] - df['low']) / df['close']
        avg_daily_range = df['daily_range_pct'].resample('D').mean().mean()
        
        # Data completeness
        expected_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
        completeness = len(df) / max(expected_hours, 1)
        
        # Total return over period
        total_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
        
        # Sharpe-like metric (annualized return / volatility)
        days_elapsed = (df.index[-1] - df.index[0]).days
        if days_elapsed > 0:
            annual_return = total_return * (365 / days_elapsed)
        else:
            annual_return = 0
        
        sharpe_approx = annual_return / annualized_vol if annualized_vol > 0 else 0
        
        # Number of zero-volume hours (dead periods)
        zero_vol_pct = (df['volume'] == 0).mean()
        
        # Recent volume trend (last 30 days vs overall avg)
        recent_hours = min(30 * 24, len(df))
        recent_vol = df['dollar_volume'].tail(recent_hours).mean()
        volume_trend = recent_vol / avg_daily_volume if avg_daily_volume > 0 else 0
        
        stats.append({
            'symbol': symbol,
            'avg_daily_volume_usd': avg_daily_volume,
            'recent_volume_trend': volume_trend,
            'annualized_vol': annualized_vol,
            'avg_daily_range_pct': avg_daily_range,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_approx': sharpe_approx,
            'data_completeness': completeness,
            'zero_volume_pct': zero_vol_pct,
            'num_candles': len(df),
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d')
        })
    
    stats_df = pd.DataFrame(stats).sort_values('avg_daily_volume_usd', ascending=False)
    return stats_df


def select_universe(stats_df: pd.DataFrame, 
                     min_daily_volume: float = 1_000_000,    # $1M minimum (lower for crypto)
                     min_completeness: float = 0.90,          # 90% data coverage
                     max_zero_vol: float = 0.05,              # <5% zero-volume periods
                     min_vol: float = 0.30,                   # 30% annualized minimum
                     max_vol: float = 3.0,                    # 300% max (avoid crazy memecoins)
                     target_size: int = 25) -> tuple:
    """
    Filter and select trading universe.
    """
    filtered = stats_df.copy()
    
    print("=" * 60)
    print("🔍 UNIVERSE SELECTION FILTERS")
    print("=" * 60)
    print(f"Starting coins: {len(filtered)}\n")
    
    # Filter: minimum volume
    filtered = filtered[filtered['avg_daily_volume_usd'] >= min_daily_volume]
    print(f"✅ After volume filter (>${min_daily_volume/1e6:.1f}M daily): {len(filtered)}")
    
    # Filter: data completeness
    filtered = filtered[filtered['data_completeness'] >= min_completeness]
    print(f"✅ After completeness filter (>{min_completeness:.0%}): {len(filtered)}")
    
    # Filter: not too many dead periods
    filtered = filtered[filtered['zero_volume_pct'] <= max_zero_vol]
    print(f"✅ After zero-volume filter (<{max_zero_vol:.0%}): {len(filtered)}")
    
    # Filter: volatility range (we need movement, but not insanity)
    filtered = filtered[
        (filtered['annualized_vol'] >= min_vol) & 
        (filtered['annualized_vol'] <= max_vol)
    ]
    print(f"✅ After volatility filter ({min_vol:.0%} - {max_vol:.0%}): {len(filtered)}")
    
    # Rank by composite score
    # Volume = liquidity, Volatility = opportunity, Sharpe = quality
    filtered['volume_rank'] = filtered['avg_daily_volume_usd'].rank(ascending=False, pct=True)
    filtered['vol_rank'] = filtered['annualized_vol'].rank(ascending=False, pct=True)
    filtered['sharpe_rank'] = filtered['sharpe_approx'].rank(ascending=False, pct=True)
    
    # Composite: 40% volume, 30% volatility, 30% sharpe
    filtered['composite_score'] = (
        0.40 * filtered['volume_rank'] +
        0.30 * filtered['vol_rank'] +
        0.30 * filtered['sharpe_rank']
    )
    
    filtered = filtered.sort_values('composite_score', ascending=False)
    
    # Take top N
    universe = filtered.head(target_size)['symbol'].tolist()
    
    print(f"\n{'='*60}")
    print(f"🎯 FINAL UNIVERSE: {len(universe)} coins")
    print(f"{'='*60}")
    for i, coin in enumerate(universe, 1):
        print(f"{i:2d}. {coin}")
    print(f"{'='*60}\n")
    
    return universe, filtered


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 UNIVERSE SELECTION ANALYSIS")
    print("=" * 60)
    print()
    
    # Load data
    data = load_all_data()
    
    if not data:
        print("❌ No data found! Run download_data.py first.")
        exit(1)
    
    # Compute stats
    stats = compute_universe_stats(data)
    
    # Create analysis directory
    os.makedirs('analysis', exist_ok=True)
    
    # Save full stats
    stats.to_csv('analysis/universe_stats_full.csv', index=False)
    print("💾 Saved full statistics to: analysis/universe_stats_full.csv\n")
    
    print("\n" + "=" * 60)
    print("📊 TOP 20 COINS BY VOLUME")
    print("=" * 60)
    top_20_display = stats.head(20)[['symbol', 'avg_daily_volume_usd', 'annualized_vol', 
                                      'total_return', 'sharpe_approx']].copy()
    top_20_display['avg_daily_volume_usd'] = top_20_display['avg_daily_volume_usd'] / 1e6
    top_20_display.columns = ['Symbol', 'Daily Vol ($M)', 'Ann. Vol', 'Total Return', 'Sharpe']
    print(top_20_display.to_string(index=False))
    
    # Select universe
    print("\n")
    universe, filtered_stats = select_universe(
        stats, 
        target_size=25,              # Top 25 coins
        min_daily_volume=1_000_000,  # $1M min
        min_vol=0.30,                # 30% min vol
        max_vol=3.0                  # 300% max vol
    )
    
    # Save universe
    os.makedirs('data', exist_ok=True)
    with open('data/universe.txt', 'w') as f:
        f.write('\n'.join(universe))
    
    print("💾 Saved universe to: data/universe.txt")
    
    # Save filtered stats
    filtered_stats.to_csv('analysis/universe_stats_filtered.csv', index=False)
    print("💾 Saved filtered stats to: analysis/universe_stats_filtered.csv\n")
    
    # Visualize
    print("📈 Creating visualization...\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top 25 by volume
    top_25 = filtered_stats.head(25)
    axes[0,0].barh(range(len(top_25)), top_25['avg_daily_volume_usd'] / 1e6)
    axes[0,0].set_yticks(range(len(top_25)))
    axes[0,0].set_yticklabels(top_25['symbol'])
    axes[0,0].set_xlabel('Avg Daily Volume ($M)')
    axes[0,0].set_title('Top 25 by Volume')
    axes[0,0].invert_yaxis()
    
    # Volatility
    axes[0,1].barh(range(len(top_25)), top_25['annualized_vol'] * 100)
    axes[0,1].set_yticks(range(len(top_25)))
    axes[0,1].set_yticklabels(top_25['symbol'])
    axes[0,1].set_xlabel('Annualized Volatility (%)')
    axes[0,1].set_title('Volatility Profile')
    axes[0,1].invert_yaxis()
    
    # Volume vs Volatility scatter
    axes[1,0].scatter(filtered_stats['avg_daily_volume_usd'] / 1e6, 
                       filtered_stats['annualized_vol'] * 100, 
                       s=100, alpha=0.6, c=filtered_stats['composite_score'], 
                       cmap='viridis')
    
    # Annotate selected universe (top 15 to avoid clutter)
    for symbol in universe[:15]:
        row = filtered_stats[filtered_stats['symbol'] == symbol].iloc[0]
        axes[1,0].annotate(symbol, 
                           (row['avg_daily_volume_usd']/1e6, row['annualized_vol']*100),
                           fontsize=8, fontweight='bold')
    
    axes[1,0].set_xlabel('Avg Daily Volume ($M)')
    axes[1,0].set_ylabel('Annualized Volatility (%)')
    axes[1,0].set_title('Volume vs Volatility (colored by composite score)')
    axes[1,0].set_xscale('log')
    
    # Returns
    axes[1,1].barh(range(len(top_25)), top_25['total_return'] * 100)
    axes[1,1].set_yticks(range(len(top_25)))
    axes[1,1].set_yticklabels(top_25['symbol'])
    axes[1,1].set_xlabel('Total Return (%)')
    axes[1,1].set_title('Historical Return (Full Period)')
    axes[1,1].axvline(0, color='black', linestyle='--', linewidth=0.5)
    axes[1,1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('analysis/universe_selection.png', dpi=150, bbox_inches='tight')
    print("✅ Saved visualization to: analysis/universe_selection.png")
    
    print("\n" + "=" * 60)
    print("✅ UNIVERSE SELECTION COMPLETE!")
    print("=" * 60)
    print(f"\n📋 Next steps:")
    print(f"   1. Review analysis/universe_selection.png")
    print(f"   2. Check data/universe.txt for selected coins")
    print(f"   3. Run backtest with these {len(universe)} coins")