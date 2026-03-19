import os
import sys
import time
import itertools
import pandas as pd
import numpy as np
from datetime import datetime
from backtest.data_prep import prepare_backtest_data
from backtest.engine import MomentumBacktest

with open('data/universe.txt') as f:
    universe = [line.strip() for line in f if line.strip()]

print(f"Loading data for {len(universe)} coins...")
data = prepare_backtest_data(universe)
print(f"Data ready: {len(data['close'])} timestamps\n")

# Targeted grid: include 6h and 12h rebalancing,
# include the 0.2 bear scalar that previously worked,
# use longer lookback (confirmed winner)
param_grid = {
    'rebalance_frequency': [6, 12, 24],
    'num_holdings':        [4, 6, 8],
    'rebalance_threshold': [0.05, 0.08, 0.12],
    'regime_config':       [
        # (caution_thresh, bear_thresh, caution_scalar, bear_scalar, name)
        (-0.01, -0.03, 0.5, 0.2, 'soft'),       # Previous +10% config style
        (-0.01, -0.03, 0.5, 0.0, 'soft_cash'),   # Same but cash in bear
        (-0.02, -0.02, 0.0, 0.0, 'binary'),      # In or out
        ('off', 'off', 1.0, 1.0, 'off'),          # No regime filter
    ],
    'lookback_config':     [
        ([24, 72, 168], [0.2, 0.3, 0.5], 'default'),     # 1d/3d/7d
        ([48, 168, 336], [0.2, 0.3, 0.5], 'longer'),     # 2d/7d/14d (confirmed winner)
        ([12, 48, 168], [0.3, 0.3, 0.4], 'shorter'),     # 12h/2d/7d (faster signal)
    ],
}
# Total: 3 × 3 × 3 × 4 × 3 = 324 runs
# At ~10-20 sec each = 54-108 minutes

FIXED = {
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
    'position_stop_loss': -0.99,  # Disabled for speed
    'use_stops': False,
    # Drawdown: use the setting from the +10% run
    'drawdown_reduce': -0.05,
    'drawdown_panic': -0.10,
    'drawdown_recovery': -0.03,
}


def build_params(rebal, holdings, thresh, regime_cfg, lb_cfg):
    p = FIXED.copy()
    p['rebalance_frequency'] = rebal
    p['num_holdings'] = holdings
    p['rebalance_threshold'] = thresh

    # Lookback
    p['lookback_windows'] = lb_cfg[0]
    p['lookback_weights'] = lb_cfg[1]

    # Regime
    if regime_cfg[4] == 'off':
        p['use_regime_filter'] = False
        p['regime_caution_threshold'] = -0.01
        p['regime_bear_threshold'] = -0.05
        p['regime_caution_scalar'] = 0.5
        p['regime_bear_scalar'] = 0.0
    else:
        p['use_regime_filter'] = True
        p['regime_caution_threshold'] = regime_cfg[0]
        p['regime_bear_threshold'] = regime_cfg[1]
        p['regime_caution_scalar'] = regime_cfg[2]
        p['regime_bear_scalar'] = regime_cfg[3]

    return p


def run_single(params):
    try:
        bt = MomentumBacktest(params)
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        results = bt.run(data)
        sys.stdout = old_stdout

        m = results['metrics']
        tl = results['trade_log']

        if len(tl) > 0:
            avg_weight = tl['total_weight'].mean()
            avg_positions = tl['num_positions'].mean()
            num_trades = len(tl)
        else:
            avg_weight = 0
            avg_positions = 0
            num_trades = 0

        return {
            'total_return': m['total_return'],
            'annualized_return': m['annualized_return'],
            'sharpe': m['sharpe'],
            'sortino': m['sortino'],
            'max_drawdown': m['max_drawdown'],
            'calmar': m['calmar'],
            'commission_pct': m['commission_pct'],
            'total_commission': m['total_commission'],
            'final_equity': m['final_equity'],
            'peak_equity': m['peak_equity'],
            'num_trades': num_trades,
            'avg_weight': avg_weight,
            'avg_positions': avg_positions,
            'gross_return': m['total_return'] + m['commission_pct'],
        }
    except Exception as e:
        sys.stdout = sys.__stdout__
        print(f"  FAILED: {e}")
        return None


# Generate combinations
combos = []
for rebal in param_grid['rebalance_frequency']:
    for hold in param_grid['num_holdings']:
        for thresh in param_grid['rebalance_threshold']:
            for regime in param_grid['regime_config']:
                for lb in param_grid['lookback_config']:
                    combos.append((rebal, hold, thresh, regime, lb))

print(f"{'='*70}")
print(f"TARGETED PARAMETER SWEEP")
print(f"{'='*70}")
print(f"  Total combinations: {len(combos)}")
print(f"  Started at: {datetime.now().strftime('%H:%M:%S')}")
print(f"{'='*70}\n")

all_results = []
start_time = time.time()

for i, (rebal, hold, thresh, regime, lb) in enumerate(combos):
    elapsed = time.time() - start_time
    eta = f"ETA: {(elapsed/(i+1))*(len(combos)-i-1)/60:.0f}min" if i > 0 else ""

    regime_name = regime[4]
    lb_name = lb[2]

    print(f"  [{i+1}/{len(combos)}] "
          f"rebal={rebal}h hold={hold} thresh={thresh} "
          f"regime={regime_name} lb={lb_name}  {eta}")

    params = build_params(rebal, hold, thresh, regime, lb)
    result = run_single(params)

    if result is not None:
        row = {
            'rebalance_frequency': rebal,
            'num_holdings': hold,
            'rebalance_threshold': thresh,
            'regime_mode': regime_name,
            'lookback_config': lb_name,
            **result,
        }
        all_results.append(row)

elapsed_total = (time.time() - start_time) / 60
print(f"\n  Completed {len(all_results)}/{len(combos)} runs in {elapsed_total:.1f} minutes")

# Analyze
os.makedirs('analysis', exist_ok=True)
df = pd.DataFrame(all_results)
df.to_csv('analysis/targeted_sweep_results.csv', index=False)

# Composite score
for col, lo, hi in [('total_return', -0.5, 1.0), ('sharpe', -3, 4),
                     ('sortino', -3, 5), ('calmar', -2, 4)]:
    clipped = df[col].clip(lower=lo, upper=hi)
    df[f'score_{col}'] = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-10)

df['composite'] = (0.25 * df['score_total_return'] +
                   0.25 * df['score_sharpe'] +
                   0.25 * df['score_sortino'] +
                   0.25 * df['score_calmar'])

df = df.sort_values('composite', ascending=False)

# Print top 20
print(f"\n{'='*110}")
print(f"TOP 20 PARAMETER COMBINATIONS")
print(f"{'='*110}")
print(f"{'Rank':>4} {'Return':>8} {'Gross':>7} {'Sharpe':>7} {'Sortino':>8} "
      f"{'MaxDD':>8} {'Calmar':>7} {'Comm%':>7} {'Trades':>6} {'AvgWt':>6} "
      f"{'Rebal':>5} {'Hold':>4} {'Thresh':>6} {'Regime':>10} {'LB':>8} {'Score':>6}")
print('-' * 110)

for rank, (_, row) in enumerate(df.head(20).iterrows(), 1):
    print(f"{rank:>4} "
          f"{row['total_return']:>7.1%} "
          f"{row['gross_return']:>6.1%} "
          f"{row['sharpe']:>7.2f} "
          f"{row['sortino']:>8.2f} "
          f"{row['max_drawdown']:>7.1%} "
          f"{row['calmar']:>7.2f} "
          f"{row['commission_pct']:>6.1%} "
          f"{row['num_trades']:>6.0f} "
          f"{row['avg_weight']:>5.0%} "
          f"{row['rebalance_frequency']:>5.0f}h "
          f"{row['num_holdings']:>4.0f} "
          f"{row['rebalance_threshold']:>5.0%} "
          f"{row['regime_mode']:>10} "
          f"{row['lookback_config']:>8} "
          f"{row['composite']:>6.3f}")

# Show positive-return runs (if any)
positive = df[df['total_return'] > 0]
print(f"\n  POSITIVE RETURN RUNS: {len(positive)} out of {len(df)}")
if len(positive) > 0:
    print(f"\n  Best positive runs:")
    for _, row in positive.head(10).iterrows():
        print(f"    Return={row['total_return']:.1%} Sharpe={row['sharpe']:.2f} "
              f"MaxDD={row['max_drawdown']:.1%} Comm={row['commission_pct']:.1%} "
              f"rebal={row['rebalance_frequency']:.0f}h hold={row['num_holdings']:.0f} "
              f"regime={row['regime_mode']} lb={row['lookback_config']}")

# Parameter importance
print(f"\n{'='*80}")
print(f"PARAMETER IMPORTANCE")
print(f"{'='*80}")

for param in ['rebalance_frequency', 'num_holdings', 'rebalance_threshold',
              'regime_mode', 'lookback_config']:
    print(f"\n  {param}:")
    group = df.groupby(param).agg({
        'composite': 'mean',
        'total_return': 'mean',
        'commission_pct': 'mean',
        'sharpe': 'mean',
        'gross_return': 'mean',
    }).sort_values('composite', ascending=False)

    for val, row in group.iterrows():
        print(f"    {str(val):>12}: score={row['composite']:.3f}  "
              f"net={row['total_return']:>6.1%}  "
              f"gross={row['gross_return']:>6.1%}  "
              f"comm={row['commission_pct']:>5.1%}  "
              f"sharpe={row['sharpe']:>5.2f}")

# Best combo
best = df.iloc[0]
print(f"\n{'='*80}")
print(f"BEST COMBINATION")
print(f"{'='*80}")
print(f"  rebalance_frequency : {best['rebalance_frequency']:.0f}h")
print(f"  num_holdings        : {best['num_holdings']:.0f}")
print(f"  rebalance_threshold : {best['rebalance_threshold']:.0%}")
print(f"  regime_mode         : {best['regime_mode']}")
print(f"  lookback_config     : {best['lookback_config']}")
print(f"\n  Net Return    : {best['total_return']:.1%}")
print(f"  Gross Return  : {best['gross_return']:.1%}")
print(f"  Sharpe        : {best['sharpe']:.2f}")
print(f"  Sortino       : {best['sortino']:.2f}")
print(f"  Max DD        : {best['max_drawdown']:.1%}")
print(f"  Calmar        : {best['calmar']:.2f}")
print(f"  Commission    : {best['commission_pct']:.1%}")
print(f"  Avg Weight    : {best['avg_weight']:.0%}")
print(f"{'='*80}")