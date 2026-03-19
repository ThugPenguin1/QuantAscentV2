import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from backtest.data_prep import prepare_backtest_data
from backtest.engine import MomentumBacktest

with open('data/universe.txt') as f:
    universe = [line.strip() for line in f if line.strip()]
print(f"Universe: {len(universe)} coins")
data = prepare_backtest_data(universe)

# THE COMPETITION CONFIG
params = {
    'initial_capital': 50_000,

    # Signal: longer lookback (confirmed best)
    'lookback_windows': [48, 168, 336],
    'lookback_weights': [0.2, 0.3, 0.5],
    'skip_periods': 2,

    # Selection
    'num_holdings': 4,
    'require_positive_momentum': True,
    'rebalance_frequency': 48,            # Every 2 days (cuts trades in half)

    # Sizing
    'vol_lookback': 72,
    'target_vol': 0.50,
    'avg_correlation': 0.60,
    'max_position_weight': 0.25,
    'max_total_exposure': 0.90,

    # PAXG permanent hedge
    'paxg_floor': 0.20,                   # Always 20% in PAXG

    # Risk
    'drawdown_reduce': -0.05,
    'drawdown_panic': -0.10,
    'drawdown_recovery': -0.03,
    'use_regime_filter': True,
    'regime_caution_threshold': -0.01,
    'regime_bear_threshold': -0.03,
    'regime_caution_scalar': 0.5,
    'regime_bear_scalar': 0.0,

    # Stops off
    'position_stop_loss': -0.99,
    'use_stops': False,
    'cooldown_periods': 12,
    'trailing_stop': True,

    # Costs: high threshold to minimize trading
    'use_limit_orders': True,
    'rebalance_threshold': 0.04,
}

bt = MomentumBacktest(params)
results = bt.run(data)

m = results['metrics']
eq = results['equity']
tl = results['trade_log']
w = results['weights']
gross = m['total_return'] + m['commission_pct']

print(f"\n{'='*60}")
print(f"COMPETITION DEPLOYMENT CONFIG")
print(f"{'='*60}")
print(f"  Net Return        : {m['total_return']:>10.2%}")
print(f"  Gross Return      : {gross:>10.2%}")
print(f"  Sharpe            : {m['sharpe']:>10.2f}")
print(f"  Sortino           : {m['sortino']:>10.2f}")
print(f"  Max Drawdown      : {m['max_drawdown']:>10.2%}")
print(f"  Calmar            : {m['calmar']:>10.2f}")
print(f"  Commission        : {m['commission_pct']:>10.2%}")
print(f"  Trades            : {len(tl):>10}")
print(f"  Final Equity      : ${m['final_equity']:>10,.2f}")
print(f"  Peak Equity       : ${m['peak_equity']:>10,.2f}")

if len(tl) > 0:
    print(f"  Avg Weight        : {tl['total_weight'].mean():>10.1%}")
    print(f"  Avg Positions     : {tl['num_positions'].mean():>10.1f}")

# 10-day analysis
window = 240
ten_day = []
for start in range(0, len(eq) - window, 24):
    ret = eq.iloc[start + window] / eq.iloc[start] - 1
    ten_day.append(ret)
ten_day = np.array(ten_day)

print(f"\n  10-DAY COMPETITION WINDOW:")
print(f"    Median return   : {np.median(ten_day):>8.2%}")
print(f"    Mean return     : {np.mean(ten_day):>8.2%}")
print(f"    Best case       : {np.max(ten_day):>8.2%}")
print(f"    Worst case      : {np.min(ten_day):>8.2%}")
print(f"    % Positive      : {100*(ten_day>0).mean():>7.0f}%")
print(f"    % Above -2%     : {100*(ten_day>-0.02).mean():>7.0f}%")
print(f"    25th percentile : {np.percentile(ten_day, 25):>8.2%}")
print(f"    75th percentile : {np.percentile(ten_day, 75):>8.2%}")

# Holdings
print(f"\n  COIN HOLDING FREQUENCY:")
freq = (w > 0.01).mean().sort_values(ascending=False)
for coin, f_val in freq.items():
    if f_val > 0.005:
        print(f"    {coin:12s}: {f_val:>6.1%}")

# Monthly breakdown
print(f"\n  MONTHLY RETURNS:")
monthly = eq.resample('M').last().pct_change().dropna()
for date, ret in monthly.items():
    emoji = "+" if ret > 0 else "-"
    print(f"    {date.strftime('%Y-%m')}: {emoji}{abs(ret):.1%}")

print(f"{'='*60}")

# Plot
os.makedirs('analysis', exist_ok=True)
fig, axes = plt.subplots(3, 2, figsize=(18, 14))

# 1. Equity
axes[0, 0].plot(eq.index, eq.values, linewidth=1.2, color='blue')
axes[0, 0].axhline(50000, color='gray', linestyle='--', alpha=0.5)
axes[0, 0].set_ylabel('Portfolio ($)')
axes[0, 0].set_title(f'Equity Curve (Net: {m["total_return"]:.1%})')
axes[0, 0].grid(True, alpha=0.3)

# 2. Drawdown
pk = eq.expanding().max()
dd = (eq - pk) / pk * 100
axes[0, 1].fill_between(dd.index, dd.values, 0, color='red', alpha=0.4)
axes[0, 1].set_ylabel('Drawdown (%)')
axes[0, 1].set_title(f'Drawdown (Max: {m["max_drawdown"]:.1%})')
axes[0, 1].grid(True, alpha=0.3)

# 3. Portfolio composition
top_coins = freq[freq > 0.005].index.tolist()[:10]
if top_coins:
    w_top = w[top_coins].clip(lower=0)
    axes[1, 0].stackplot(w_top.index, *[w_top[c].values for c in top_coins],
                         labels=top_coins, alpha=0.7)
    axes[1, 0].set_ylabel('Weight')
    axes[1, 0].set_title('Portfolio Composition')
    axes[1, 0].legend(loc='upper right', fontsize=7, ncol=3)
    axes[1, 0].grid(True, alpha=0.3)

# 4. 10-day return distribution
axes[1, 1].hist(ten_day * 100, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].axvline(np.median(ten_day) * 100, color='green', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(ten_day):.2%}')
axes[1, 1].set_xlabel('10-Day Return (%)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('10-Day Return Distribution (Competition Windows)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 5. Rolling 10-day return
rolling_10d = eq.pct_change(240)
axes[2, 0].plot(rolling_10d.index, rolling_10d.values * 100, linewidth=0.8, color='green')
axes[2, 0].axhline(0, color='gray', linestyle='--')
axes[2, 0].set_ylabel('10-Day Return (%)')
axes[2, 0].set_title('Rolling 10-Day Return Over Time')
axes[2, 0].grid(True, alpha=0.3)

# 6. Regime filter
if results['regime_scalar'] is not None:
    rs = results['regime_scalar']
    axes[2, 1].plot(rs.index, rs.values, color='orange', linewidth=0.8)
    axes[2, 1].set_ylabel('Scalar')
    axes[2, 1].set_title('Regime Filter')
    axes[2, 1].set_ylim(-0.1, 1.1)
    axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/competition_config.png', dpi=150)
print(f"\nSaved: analysis/competition_config.png")
plt.show(block=False)
plt.pause(2)
plt.close()

# FINAL VERDICT
print(f"\n{'='*60}")
print(f"VERDICT")
print(f"{'='*60}")
all_pass = True

checks = [
    ("Gross return positive", gross > 0),
    ("Max drawdown < 10%", abs(m['max_drawdown']) < 0.10),
    ("Commission < 5%", m['commission_pct'] < 0.05),
    ("10-day median >= 0", np.median(ten_day) >= 0),
    ("10-day worst > -5%", np.min(ten_day) > -0.05),
    (">50% of 10-day windows positive", (ten_day > 0).mean() > 0.50),
    ("Equity never zero", eq.min() > 0),
]

for name, passed in checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  {status}: {name}")

print(f"\n  {'READY FOR DEPLOYMENT' if all_pass else 'REVIEW FAILURES ABOVE'}")
print(f"{'='*60}")