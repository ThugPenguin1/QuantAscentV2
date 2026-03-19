import os
import sys
import pandas as pd
import numpy as np
from backtest.data_prep import prepare_backtest_data
from backtest.engine import MomentumBacktest

with open('data/universe.txt') as f:
    universe = [line.strip() for line in f if line.strip()]

data = prepare_backtest_data(universe)

# Base params from sweep winner
base = {
    'initial_capital': 50_000,
    'skip_periods': 2,
    'vol_lookback': 72,
    'target_vol': 0.50,
    'avg_correlation': 0.60,
    'max_position_weight': 0.25,
    'max_total_exposure': 0.90,
    'cooldown_periods': 12,
    'trailing_stop': True,
    'use_limit_orders': True,
    'position_stop_loss': -0.99,
    'use_stops': False,
    'drawdown_reduce': -0.05,
    'drawdown_panic': -0.10,
    'drawdown_recovery': -0.03,
}

# Test configurations
configs = [
    # (name, rebal, holdings, thresh, regime, lookback, abs_momentum)
    
    # Sweep winner WITHOUT absolute momentum filter
    ("sweep_winner_no_filter", 24, 8, 0.12, 'soft_cash', 'longer', False),
    
    # Sweep winner WITH absolute momentum filter
    ("sweep_winner_abs_filter", 24, 8, 0.12, 'soft_cash', 'longer', True),
    
    # Variations with absolute filter
    ("abs_24h_6hold_longer", 24, 6, 0.08, 'binary', 'longer', True),
    ("abs_24h_4hold_longer", 24, 4, 0.08, 'binary', 'longer', True),
    ("abs_12h_6hold_longer", 12, 6, 0.08, 'binary', 'longer', True),
    ("abs_6h_6hold_longer", 6, 6, 0.12, 'binary', 'longer', True),
    ("abs_6h_8hold_longer", 6, 8, 0.12, 'binary', 'longer', True),
    
    # Absolute filter with default lookback
    ("abs_24h_6hold_default", 24, 6, 0.08, 'binary', 'default', True),
    ("abs_12h_6hold_default", 12, 6, 0.08, 'binary', 'default', True),
    
    # Absolute filter with shorter lookback (faster signals)
    ("abs_12h_6hold_shorter", 12, 6, 0.08, 'binary', 'shorter', True),
    
    # Absolute filter with NO regime filter (the abs filter IS the regime filter)
    ("abs_24h_6hold_no_regime", 24, 6, 0.08, 'off', 'longer', True),
    ("abs_12h_6hold_no_regime", 12, 6, 0.08, 'off', 'longer', True),
    ("abs_6h_6hold_no_regime", 6, 6, 0.12, 'off', 'longer', True),
    
    # Absolute filter, no regime, different lookbacks
    ("abs_24h_6hold_no_regime_default", 24, 6, 0.08, 'off', 'default', True),
    ("abs_24h_8hold_no_regime_longer", 24, 8, 0.08, 'off', 'longer', True),
]

def build_params(config):
    name, rebal, holdings, thresh, regime, lookback, abs_mom = config
    p = base.copy()
    p['rebalance_frequency'] = rebal
    p['num_holdings'] = holdings
    p['rebalance_threshold'] = thresh
    p['require_positive_momentum'] = abs_mom

    if lookback == 'default':
        p['lookback_windows'] = [24, 72, 168]
        p['lookback_weights'] = [0.2, 0.3, 0.5]
    elif lookback == 'longer':
        p['lookback_windows'] = [48, 168, 336]
        p['lookback_weights'] = [0.2, 0.3, 0.5]
    elif lookback == 'shorter':
        p['lookback_windows'] = [12, 48, 168]
        p['lookback_weights'] = [0.3, 0.3, 0.4]

    if regime == 'binary':
        p['use_regime_filter'] = True
        p['regime_caution_threshold'] = -0.02
        p['regime_bear_threshold'] = -0.02
        p['regime_caution_scalar'] = 0.0
        p['regime_bear_scalar'] = 0.0
    elif regime == 'soft_cash':
        p['use_regime_filter'] = True
        p['regime_caution_threshold'] = -0.01
        p['regime_bear_threshold'] = -0.03
        p['regime_caution_scalar'] = 0.5
        p['regime_bear_scalar'] = 0.0
    elif regime == 'off':
        p['use_regime_filter'] = False
        p['regime_caution_threshold'] = -0.01
        p['regime_bear_threshold'] = -0.05
        p['regime_caution_scalar'] = 0.5
        p['regime_bear_scalar'] = 0.0

    return p


print(f"{'='*100}")
print(f"ABSOLUTE MOMENTUM FILTER TEST")
print(f"{'='*100}")
print(f"  Testing {len(configs)} configurations\n")

results = []

for i, config in enumerate(configs):
    name = config[0]
    print(f"[{i+1}/{len(configs)}] {name}...")

    params = build_params(config)

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        bt = MomentumBacktest(params)
        res = bt.run(data)
        sys.stdout = old_stdout

        m = res['metrics']
        tl = res['trade_log']

        avg_wt = tl['total_weight'].mean() if len(tl) > 0 else 0
        n_trades = len(tl)
        total_comm = m['total_commission']
        gross = m['total_return'] + m['commission_pct']

        print(f"   Net={m['total_return']:>7.1%}  Gross={gross:>6.1%}  "
              f"Sharpe={m['sharpe']:>6.2f}  MaxDD={m['max_drawdown']:>6.1%}  "
              f"Comm={m['commission_pct']:>5.1%}  Trades={n_trades}  AvgWt={avg_wt:.0%}")

        results.append({
            'name': name,
            'abs_filter': config[6],
            'rebal': config[1],
            'holdings': config[2],
            'thresh': config[3],
            'regime': config[4],
            'lookback': config[5],
            'net_return': m['total_return'],
            'gross_return': gross,
            'sharpe': m['sharpe'],
            'sortino': m['sortino'],
            'max_dd': m['max_drawdown'],
            'calmar': m['calmar'],
            'commission': m['commission_pct'],
            'trades': n_trades,
            'avg_weight': avg_wt,
            'final_equity': m['final_equity'],
        })
    except Exception as e:
        sys.stdout = old_stdout
        print(f"   FAILED: {e}")

# Results table
df = pd.DataFrame(results)
df = df.sort_values('net_return', ascending=False)

print(f"\n{'='*120}")
print(f"RESULTS RANKED BY NET RETURN")
print(f"{'='*120}")
print(f"{'Name':<35} {'AbsFilt':>7} {'Net':>7} {'Gross':>7} {'Sharpe':>7} "
      f"{'MaxDD':>7} {'Calmar':>7} {'Comm':>6} {'Trades':>6} {'AvgWt':>6}")
print('-' * 120)

for _, row in df.iterrows():
    filt = "YES" if row['abs_filter'] else "NO"
    print(f"{row['name']:<35} {filt:>7} {row['net_return']:>6.1%} {row['gross_return']:>6.1%} "
          f"{row['sharpe']:>7.2f} {row['max_dd']:>6.1%} {row['calmar']:>7.2f} "
          f"{row['commission']:>5.1%} {row['trades']:>6.0f} {row['avg_weight']:>5.0%}")

# Comparison: with vs without absolute filter
print(f"\n{'='*80}")
print(f"IMPACT OF ABSOLUTE MOMENTUM FILTER")
print(f"{'='*80}")

with_filter = df[df['abs_filter'] == True]
without_filter = df[df['abs_filter'] == False]

if len(without_filter) > 0:
    print(f"\n  WITHOUT absolute filter (best):")
    best_no = without_filter.iloc[0]
    print(f"    Net: {best_no['net_return']:.1%}  Gross: {best_no['gross_return']:.1%}  "
          f"MaxDD: {best_no['max_dd']:.1%}  Commission: {best_no['commission']:.1%}")

if len(with_filter) > 0:
    print(f"\n  WITH absolute filter (best):")
    best_yes = with_filter.sort_values('net_return', ascending=False).iloc[0]
    print(f"    Net: {best_yes['net_return']:.1%}  Gross: {best_yes['gross_return']:.1%}  "
          f"MaxDD: {best_yes['max_dd']:.1%}  Commission: {best_yes['commission']:.1%}")
    print(f"    Config: {best_yes['name']}")

# Save
df.to_csv('analysis/absolute_momentum_test.csv', index=False)
print(f"\nSaved to analysis/absolute_momentum_test.csv")