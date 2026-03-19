import os
import sys
import matplotlib.pyplot as plt
from backtest.data_prep import prepare_backtest_data
from backtest.engine import MomentumBacktest

# Load universe and add PAXG + TRX
with open('data/universe.txt') as f:
    universe = [line.strip() for line in f if line.strip()]

for coin in ['PAXG', 'TRX']:
    if coin not in universe:
        # Check if data exists
        if os.path.exists(f'data/raw/{coin}_1h.csv'):
            universe.append(coin)
            print(f"Added {coin} to universe")
        else:
            print(f"WARNING: No data for {coin}")

print(f"Universe: {len(universe)} coins\n")

data = prepare_backtest_data(universe)

# Winning config with absolute momentum filter
params = {
    'initial_capital': 50_000_000,

    # Signal
    'lookback_windows': [48, 168, 336],      # Longer lookback (confirmed winner)
    'lookback_weights': [0.2, 0.3, 0.5],
    'skip_periods': 2,

    # Selection
    'num_holdings': 8,                        # Top 8 (sweep winner)
    'require_positive_momentum': True,        # THE KEY: only buy uptrending coins
    'rebalance_frequency': 24,                # Daily

    # Sizing
    'vol_lookback': 72,
    'target_vol': 0.50,
    'avg_correlation': 0.60,
    'max_position_weight': 0.25,
    'max_total_exposure': 0.90,

    # Risk
    'drawdown_reduce': -0.05,
    'drawdown_panic': -0.10,
    'drawdown_recovery': -0.03,
    'use_regime_filter': True,
    'regime_caution_threshold': -0.01,
    'regime_bear_threshold': -0.03,
    'regime_caution_scalar': 0.5,
    'regime_bear_scalar': 0.0,

    # Stops
    'position_stop_loss': -0.99,
    'use_stops': False,
    'cooldown_periods': 12,
    'trailing_stop': True,

    # Costs
    'use_limit_orders': True,
    'rebalance_threshold': 0.12,              # 12% drift threshold (sweep winner)
}

bt = MomentumBacktest(params)
results = bt.run(data)

m = results['metrics']
tl = results['trade_log']
eq = results['equity']
w = results['weights']

# Metrics
print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"  Net Return        : {m['total_return']:>10.2%}")
print(f"  Gross Return      : {m['total_return']+m['commission_pct']:>10.2%}")
print(f"  Annualized Return : {m['annualized_return']:>10.2%}")
print(f"  Sharpe Ratio      : {m['sharpe']:>10.2f}")
print(f"  Sortino Ratio     : {m['sortino']:>10.2f}")
print(f"  Max Drawdown      : {m['max_drawdown']:>10.2%}")
print(f"  Calmar Ratio      : {m['calmar']:>10.2f}")
print(f"  Win Rate          : {m['win_rate']:>10.2%}")
print(f"  Commission Drag   : {m['commission_pct']:>10.2%}")
print(f"  Final Equity      : ${m['final_equity']:>10,.2f}")
print(f"  Peak Equity       : ${m['peak_equity']:>10,.2f}")
print(f"{'='*60}")

# Trade diagnostics
if len(tl) > 0:
    print(f"\n  Trades: {len(tl)}")
    print(f"  Avg weight: {tl['total_weight'].mean():.1%}")
    print(f"  Commission: ${tl['commission'].sum():,.2f}")

# Coin holding frequency
print(f"\nCOIN HOLDING FREQUENCY:")
freq = (w > 0.01).mean().sort_values(ascending=False)
for coin, f_val in freq.items():
    if f_val > 0.005:
        print(f"  {coin:12s}: {f_val:>6.1%}")

# 10-day rolling window analysis
print(f"\n{'='*60}")
print(f"10-DAY WINDOW ANALYSIS (competition simulation)")
print(f"{'='*60}")
window = 240  # 10 days in hours
ten_day_returns = []
for start in range(0, len(eq) - window, 24):
    end = start + window
    ret = eq.iloc[end] / eq.iloc[start] - 1
    ten_day_returns.append(ret)

import numpy as np
ten_day = np.array(ten_day_returns)
print(f"  Windows tested    : {len(ten_day)}")
print(f"  Median 10-day ret : {np.median(ten_day):>7.2%}")
print(f"  Mean 10-day ret   : {np.mean(ten_day):>7.2%}")
print(f"  Best 10-day       : {np.max(ten_day):>7.2%}")
print(f"  Worst 10-day      : {np.min(ten_day):>7.2%}")
print(f"  % Positive        : {(ten_day > 0).mean():>7.1%}")
print(f"  % > -2%           : {(ten_day > -0.02).mean():>7.1%}")
print(f"  25th percentile   : {np.percentile(ten_day, 25):>7.2%}")
print(f"  75th percentile   : {np.percentile(ten_day, 75):>7.2%}")

# Plot
os.makedirs('analysis', exist_ok=True)

fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)

# 1. Equity
axes[0].plot(eq.index, eq.values, linewidth=1.2, color='blue')
axes[0].axhline(50000, color='gray', linestyle='--', alpha=0.5)
axes[0].set_ylabel('Portfolio ($)')
axes[0].set_title('Equity Curve')
axes[0].grid(True, alpha=0.3)

# 2. Drawdown
pk = eq.expanding().max()
dd = (eq - pk) / pk * 100
axes[1].fill_between(dd.index, dd.values, 0, color='red', alpha=0.4)
axes[1].set_ylabel('Drawdown (%)')
axes[1].set_title('Drawdown')
axes[1].grid(True, alpha=0.3)

# 3. What's held (stacked area)
top_coins = freq[freq > 0.01].index.tolist()[:10]
if top_coins:
    w_top = w[top_coins]
    axes[2].stackplot(w_top.index, *[w_top[c].values for c in top_coins],
                      labels=top_coins, alpha=0.7)
    axes[2].set_ylabel('Weight')
    axes[2].set_title('Portfolio Composition')
    axes[2].legend(loc='upper right', fontsize=8, ncol=3)
    axes[2].grid(True, alpha=0.3)

# 4. 10-day rolling return
rolling_10d = eq.pct_change(240)  # 10-day return at each point
axes[3].plot(rolling_10d.index, rolling_10d.values * 100, linewidth=0.8, color='green')
axes[3].axhline(0, color='gray', linestyle='--')
axes[3].set_ylabel('10-Day Return (%)')
axes[3].set_title('Rolling 10-Day Return (competition window)')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/final_backtest.png', dpi=150)
print(f"\nSaved: analysis/final_backtest.png")
plt.show(block=False)
plt.pause(2)
plt.close()