import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from backtest.data_prep import prepare_backtest_data
from backtest.engine import MomentumBacktest

# Universe with PAXG and TRX
with open('data/universe.txt') as f:
    universe = [line.strip() for line in f if line.strip()]
print(f"Universe: {len(universe)} coins")

data = prepare_backtest_data(universe)

# Test 4 configs side by side
configs = {
    'current_cash_strategy': {
        'rebalance_threshold': 0.12,      # Blocks entries (current "winner")
        'paxg_floor': 0.0,                # No forced PAXG
        'num_holdings': 8,
        'regime_bear_scalar': 0.0,
    },
    'fix_entries_only': {
        'rebalance_threshold': 0.04,      # Let positions actually enter
        'paxg_floor': 0.0,
        'num_holdings': 6,
        'regime_bear_scalar': 0.0,
    },
    'fix_entries_plus_paxg': {
        'rebalance_threshold': 0.04,
        'paxg_floor': 0.15,               # Always hold 15% PAXG
        'num_holdings': 6,
        'regime_bear_scalar': 0.0,
    },
    'paxg_hedge_conservative': {
        'rebalance_threshold': 0.04,
        'paxg_floor': 0.20,               # Always hold 20% PAXG
        'num_holdings': 4,                 # Fewer momentum picks
        'regime_bear_scalar': 0.0,
    },
}

base_params = {
    'initial_capital': 50_000,
    'lookback_windows': [48, 168, 336],
    'lookback_weights': [0.2, 0.3, 0.5],
    'skip_periods': 2,
    'rebalance_frequency': 24,
    'require_positive_momentum': True,
    'vol_lookback': 72,
    'target_vol': 0.50,
    'avg_correlation': 0.60,
    'max_position_weight': 0.25,
    'max_total_exposure': 0.90,
    'drawdown_reduce': -0.05,
    'drawdown_panic': -0.10,
    'drawdown_recovery': -0.03,
    'use_regime_filter': True,
    'regime_caution_threshold': -0.01,
    'regime_bear_threshold': -0.03,
    'regime_caution_scalar': 0.5,
    'position_stop_loss': -0.99,
    'use_stops': False,
    'cooldown_periods': 12,
    'trailing_stop': True,
    'use_limit_orders': True,
}

print(f"\n{'='*100}")
print(f"COMPETITION READINESS TEST")
print(f"{'='*100}\n")

all_results = {}

for name, overrides in configs.items():
    print(f"\n--- {name} ---")

    p = base_params.copy()
    p['rebalance_threshold'] = overrides['rebalance_threshold']
    p['num_holdings'] = overrides['num_holdings']
    p['regime_bear_scalar'] = overrides['regime_bear_scalar']
    p['paxg_floor'] = overrides['paxg_floor']

    bt = MomentumBacktest(p)
    results = bt.run(data)

    m = results['metrics']
    eq = results['equity']
    tl = results['trade_log']

    avg_wt = tl['total_weight'].mean() if len(tl) > 0 else 0
    gross = m['total_return'] + m['commission_pct']

    # 10-day windows
    window = 240
    ten_day = []
    for start in range(0, len(eq) - window, 24):
        ret = eq.iloc[start + window] / eq.iloc[start] - 1
        ten_day.append(ret)
    ten_day = np.array(ten_day)

    print(f"  Net: {m['total_return']:.2%}  Gross: {gross:.2%}  "
          f"Sharpe: {m['sharpe']:.2f}  Sortino: {m['sortino']:.2f}  "
          f"MaxDD: {m['max_drawdown']:.2%}  Calmar: {m['calmar']:.2f}")
    print(f"  Comm: {m['commission_pct']:.2%}  Trades: {len(tl)}  AvgWt: {avg_wt:.1%}")
    print(f"  10-day: median={np.median(ten_day):.2%}  "
          f"best={np.max(ten_day):.2%}  worst={np.min(ten_day):.2%}  "
          f"%positive={100*(ten_day>0).mean():.0f}%")

    # Coin frequencies
    w = results['weights']
    freq = (w > 0.01).mean().sort_values(ascending=False)
    top_coins = [(c, f) for c, f in freq.items() if f > 0.005]
    if top_coins:
        coins_str = ', '.join([f"{c}:{f:.0%}" for c, f in top_coins[:8]])
        print(f"  Holdings: {coins_str}")

    all_results[name] = {
        'metrics': m,
        'equity': eq,
        'ten_day': ten_day,
        'weights': w,
        'trade_log': tl,
    }

# Comparison table
print(f"\n{'='*100}")
print(f"COMPARISON")
print(f"{'='*100}")
print(f"{'Config':<30} {'Net':>7} {'Gross':>7} {'Sharpe':>7} {'Sortino':>8} "
      f"{'MaxDD':>7} {'Calmar':>7} {'10d Med':>7} {'10d Best':>8} {'10d Worst':>9} {'%Pos':>5}")
print('-' * 100)

for name, res in all_results.items():
    m = res['metrics']
    td = res['ten_day']
    gross = m['total_return'] + m['commission_pct']
    print(f"{name:<30} {m['total_return']:>6.1%} {gross:>6.1%} {m['sharpe']:>7.2f} "
          f"{m['sortino']:>8.2f} {m['max_drawdown']:>6.1%} {m['calmar']:>7.2f} "
          f"{np.median(td):>6.2%} {np.max(td):>7.2%} {np.min(td):>8.2%} "
          f"{100*(td>0).mean():>4.0f}%")

# Plot all equity curves
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for i, (name, res) in enumerate(all_results.items()):
    eq = res['equity']
    axes[i].plot(eq.index, eq.values, linewidth=1.2)
    axes[i].axhline(50000, color='gray', linestyle='--', alpha=0.5)
    axes[i].set_title(f"{name}\nNet: {res['metrics']['total_return']:.1%}  "
                      f"MaxDD: {res['metrics']['max_drawdown']:.1%}")
    axes[i].set_ylabel('Portfolio ($)')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('analysis', exist_ok=True)
plt.savefig('analysis/competition_ready.png', dpi=150)
print(f"\nSaved: analysis/competition_ready.png")
plt.show(block=False)
plt.pause(2)
plt.close()