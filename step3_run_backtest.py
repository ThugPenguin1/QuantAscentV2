import os
import numpy as np
import matplotlib.pyplot as plt
from backtest.data_prep import prepare_backtest_data
from backtest.engine import MomentumBacktest

with open('data/universe.txt') as f:
    universe = [line.strip() for line in f if line.strip()]
print(f"Universe: {len(universe)} coins\n")

data = prepare_backtest_data(universe)

params = {
    'initial_capital': 50_000,

    # Momentum signal
    'lookback_windows': [24, 72, 168],
    'lookback_weights': [0.2, 0.3, 0.5],
    'skip_periods': 2,

    # Portfolio construction
    'num_holdings': 6,
    'rebalance_frequency': 24,               # CHANGED: daily (was 8h)

    # Volatility targeting
    'vol_lookback': 72,
    'target_vol': 0.50,
    'avg_correlation': 0.60,
    'max_position_weight': 0.25,
    'max_total_exposure': 0.90,

    # Risk management
    'drawdown_reduce': -0.07,
    'drawdown_panic': -0.15,
    'drawdown_recovery': -0.04,
    'position_stop_loss': -0.10,
    'cooldown_periods': 12,
    'trailing_stop': True,
    'use_regime_filter': True,

    # Costs
    'use_limit_orders': True,
    'rebalance_threshold': 0.05,              # CHANGED: 5% (was 4%)
}

bt = MomentumBacktest(params)
results = bt.run(data)

m = results['metrics']
print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"  Total Return      : {m['total_return']:>10.2%}")
print(f"  Annualized Return : {m['annualized_return']:>10.2%}")
print(f"  Sharpe Ratio      : {m['sharpe']:>10.2f}")
print(f"  Sortino Ratio     : {m['sortino']:>10.2f}")
print(f"  Max Drawdown      : {m['max_drawdown']:>10.2%}")
print(f"  Calmar Ratio      : {m['calmar']:>10.2f}")
print(f"  Win Rate          : {m['win_rate']:>10.2%}")
print(f"  Total Turnover    : {m['total_turnover']:>10.2f}")
print(f"  Commission Drag   : {m['commission_pct']:>10.2%}")
print(f"  Final Equity      : ${m['final_equity']:>10,.2f}")
print(f"  Peak Equity       : ${m['peak_equity']:>10,.2f}")
print(f"  Period            : {m['years']:.2f} years")
print(f"{'='*60}")

# Gross return estimate
gross = m['total_return'] + m['commission_pct']
print(f"\n  Gross return (before comm): ~{gross:.2%}")
print(f"  Commission as % of gross : ~{m['commission_pct']/max(gross,0.001)*100:.0f}%")

eq = results['equity']
w = results['weights']
print(f"\nSANITY:")
print(f"  Equity > 0 always : {'PASS' if eq.min() > 0 else 'FAIL'}")
print(f"  Starts at $50k    : {'PASS' if abs(eq.iloc[0] - 50000) < 1 else 'FAIL'}")
print(f"  Weights in [0,1]  : {'PASS' if w.max().max() <= 1.01 else 'FAIL'}")
print(f"  Weights >= 0      : {'PASS' if w.min().min() >= -0.001 else 'FAIL'}")

tl = results['trade_log']
if len(tl) > 0:
    print(f"\nTRADE DIAGNOSTICS:")
    print(f"  Total trades      : {len(tl)}")
    print(f"  Avg turnover      : {tl['turnover'].mean():.3f}")
    print(f"  Total commission  : ${tl['commission'].sum():,.2f}")
    print(f"  Avg positions     : {tl['num_positions'].mean():.1f}")
    print(f"  Avg total weight  : {tl['total_weight'].mean():.1%}")

    print(f"\n  EXPOSURE BREAKDOWN:")
    for s in [1.0, 0.5, 0.2, 0.0]:
        pct = (tl['combined_scalar'] == s).mean()
        if pct > 0:
            print(f"    Scalar={s:.1f}: {pct:.1%} of trades")
    print(f"    Avg combined scalar: {tl['combined_scalar'].mean():.2f}")

    full_trades = tl[tl['combined_scalar'] >= 1.0]
    if len(full_trades) > 0:
        print(f"    Avg weight when bull   : {full_trades['total_weight'].mean():.1%}")
    caution_trades = tl[tl['combined_scalar'] == 0.5]
    if len(caution_trades) > 0:
        print(f"    Avg weight when caution: {caution_trades['total_weight'].mean():.1%}")

    # Commission by regime
    bull_comm = tl[tl['combined_scalar'] >= 1.0]['commission'].sum()
    caution_comm = tl[tl['combined_scalar'] == 0.5]['commission'].sum()
    bear_comm = tl[tl['combined_scalar'] <= 0.0]['commission'].sum()
    print(f"\n  COMMISSION BY REGIME:")
    print(f"    Bull    : ${bull_comm:>8,.2f}")
    print(f"    Caution : ${caution_comm:>8,.2f}")
    print(f"    Bear    : ${bear_comm:>8,.2f}")

print(f"\nCOIN HOLDING FREQUENCY (top 10):")
holding_freq = (w > 0.01).mean().sort_values(ascending=False)
for coin, freq in list(holding_freq.items())[:10]:
    if freq > 0.01:
        print(f"  {coin:12s}: {freq:>6.1%}")

os.makedirs('analysis', exist_ok=True)
fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)

axes[0].plot(eq.index, eq.values, linewidth=1.2, color='blue')
axes[0].axhline(50000, color='gray', linestyle='--', alpha=0.5)
axes[0].set_ylabel('Portfolio Value ($)')
axes[0].set_title('Equity Curve')
axes[0].grid(True, alpha=0.3)

pk = eq.expanding().max()
dd = (eq - pk) / pk * 100
axes[1].fill_between(dd.index, dd.values, 0, color='red', alpha=0.4)
axes[1].set_ylabel('Drawdown (%)')
axes[1].set_title('Drawdown')
axes[1].grid(True, alpha=0.3)

if results['regime_scalar'] is not None:
    rs = results['regime_scalar']
    axes[2].plot(rs.index, rs.values, color='orange', linewidth=0.8)
    axes[2].set_ylabel('Scalar')
    axes[2].set_title('Regime Filter (1.0=bull, 0.5=caution, 0.0=cash)')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)

total_w = w.sum(axis=1)
n_pos = (w > 0.01).sum(axis=1)
axes[3].plot(n_pos.index, n_pos.values, alpha=0.5, label='# positions', color='blue')
ax2 = axes[3].twinx()
ax2.plot(total_w.index, total_w.values * 100, color='orange', alpha=0.5, label='Raw Weight %')
ax2.set_ylabel('Raw Weight (%)')
axes[3].set_ylabel('# Positions')
axes[3].set_title('Positions & Raw Exposure')
axes[3].legend(loc='upper left')
ax2.legend(loc='upper right')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/backtest_results.png', dpi=150)
print(f"\nSaved: analysis/backtest_results.png")
plt.show(block=False)
plt.pause(2)
plt.close()