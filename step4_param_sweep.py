import os
import sys
import time
import itertools
import pandas as pd
import numpy as np
from datetime import datetime
from backtest.data_prep import prepare_backtest_data
from backtest.engine import MomentumBacktest

# ── Load data once ──
with open('data/universe.txt') as f:
    universe = [line.strip() for line in f if line.strip()]

print(f"Loading data for {len(universe)} coins...")
data = prepare_backtest_data(universe)
print(f"Data ready: {len(data['close'])} timestamps\n")

# ──────────────────────────────────────────────────────────────
# PARAMETER GRID
# ──────────────────────────────────────────────────────────────
# Each key maps to a list of values to test.
# Total combinations = product of all list lengths.
# Keep it manageable: aim for 50-150 runs (each takes ~10-30 seconds).

param_grid = {
    'rebalance_frequency': [24, 48],
    'num_holdings':        [4, 6, 8],
    'rebalance_threshold': [0.05, 0.08],
    'regime_mode':         ['binary', 'three_tier', 'off'],
    'drawdown_mode':       ['binary', 'three_tier'],
    'position_stop_loss':  [None],
    'lookback_config':     ['default', 'longer'],
}
# Total: 2 × 3 × 2 × 3 × 2 × 1 × 2 = 144 runs
# At ~20 seconds each = ~48 minutes

# Fixed parameters (not swept)
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
}


def build_params(combo):
    """Convert a grid combination into a full params dict."""
    p = FIXED.copy()

    p['rebalance_frequency'] = combo['rebalance_frequency']
    p['num_holdings'] = combo['num_holdings']
    p['rebalance_threshold'] = combo['rebalance_threshold']

    # Lookback configuration
    if combo['lookback_config'] == 'default':
        p['lookback_windows'] = [24, 72, 168]
        p['lookback_weights'] = [0.2, 0.3, 0.5]
    elif combo['lookback_config'] == 'longer':
        p['lookback_windows'] = [48, 168, 336]    # 2d, 7d, 14d
        p['lookback_weights'] = [0.2, 0.3, 0.5]

    # Regime filter mode
    p['use_regime_filter'] = True
    if combo['regime_mode'] == 'binary':
        # Bull or Cash, no middle ground
        p['regime_bear_threshold'] = -0.02
        p['regime_caution_threshold'] = -0.02  # Same as bear = no caution tier
        p['regime_bear_scalar'] = 0.0
        p['regime_caution_scalar'] = 0.0       # Caution = same as bear = cash
    elif combo['regime_mode'] == 'three_tier':
        p['regime_bear_threshold'] = -0.05
        p['regime_caution_threshold'] = -0.01
        p['regime_bear_scalar'] = 0.0
        p['regime_caution_scalar'] = 0.5
    elif combo['regime_mode'] == 'off':
        p['use_regime_filter'] = False
        p['regime_bear_threshold'] = -0.05
        p['regime_caution_threshold'] = -0.01
        p['regime_bear_scalar'] = 0.0
        p['regime_caution_scalar'] = 0.5

    # Drawdown circuit breaker mode
    if combo['drawdown_mode'] == 'binary':
        # In or Out, no middle ground
        p['drawdown_reduce'] = -0.10
        p['drawdown_panic'] = -0.10   # Same threshold = no reduce tier
        p['drawdown_recovery'] = -0.05
    elif combo['drawdown_mode'] == 'three_tier':
        p['drawdown_reduce'] = -0.07
        p['drawdown_panic'] = -0.15
        p['drawdown_recovery'] = -0.04

    # Stop loss
    if combo['position_stop_loss'] is not None:
        p['position_stop_loss'] = combo['position_stop_loss']
        p['use_stops'] = True
    else:
        p['position_stop_loss'] = -0.99  # Effectively disabled
        p['use_stops'] = False

    return p


def run_single(params, run_id, total):
    """Run one backtest silently and return metrics."""
    try:
        bt = MomentumBacktest(params)

        # Suppress print output during sweep
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        results = bt.run(data)
        sys.stdout = old_stdout

        m = results['metrics']

        # Compute additional diagnostics from trade log
        tl = results['trade_log']
        if len(tl) > 0:
            avg_weight = tl['total_weight'].mean()
            avg_positions = tl['num_positions'].mean()
            num_trades = len(tl)
            pct_full = (tl['combined_scalar'] >= 1.0).mean()
            pct_cash = (tl['combined_scalar'] <= 0.0).mean()
        else:
            avg_weight = 0
            avg_positions = 0
            num_trades = 0
            pct_full = 0
            pct_cash = 0

        return {
            'total_return': m['total_return'],
            'annualized_return': m['annualized_return'],
            'sharpe': m['sharpe'],
            'sortino': m['sortino'],
            'max_drawdown': m['max_drawdown'],
            'calmar': m['calmar'],
            'win_rate': m['win_rate'],
            'commission_pct': m['commission_pct'],
            'total_commission': m['total_commission'],
            'total_turnover': m['total_turnover'],
            'final_equity': m['final_equity'],
            'peak_equity': m['peak_equity'],
            'num_trades': num_trades,
            'avg_weight': avg_weight,
            'avg_positions': avg_positions,
            'pct_full_exposure': pct_full,
            'pct_cash': pct_cash,
            'gross_return': m['total_return'] + m['commission_pct'],
        }
    except Exception as e:
        sys.stdout = sys.__stdout__
        print(f"  Run {run_id} FAILED: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# GENERATE ALL COMBINATIONS
# ──────────────────────────────────────────────────────────────
keys = list(param_grid.keys())
values = list(param_grid.values())
combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"{'='*70}")
print(f"PARAMETER SWEEP")
print(f"{'='*70}")
print(f"  Parameters being tested:")
for k, v in param_grid.items():
    print(f"    {k}: {v}")
print(f"\n  Total combinations: {len(combos)}")
print(f"  Estimated time: {len(combos) * 15 / 60:.0f} - {len(combos) * 30 / 60:.0f} minutes")
print(f"  Started at: {datetime.now().strftime('%H:%M:%S')}")
print(f"{'='*70}\n")

# ──────────────────────────────────────────────────────────────
# RUN ALL COMBINATIONS
# ──────────────────────────────────────────────────────────────
all_results = []
start_time = time.time()

for i, combo in enumerate(combos):
    elapsed = time.time() - start_time
    if i > 0:
        per_run = elapsed / i
        remaining = per_run * (len(combos) - i)
        eta = f"ETA: {remaining/60:.0f}min"
    else:
        eta = ""

    print(f"  [{i+1}/{len(combos)}] "
          f"rebal={combo['rebalance_frequency']}h "
          f"hold={combo['num_holdings']} "
          f"thresh={combo['rebalance_threshold']} "
          f"regime={combo['regime_mode']} "
          f"dd={combo['drawdown_mode']} "
          f"stop={combo['position_stop_loss']} "
          f"lb={combo['lookback_config']}  "
          f"{eta}")

    params = build_params(combo)
    result = run_single(params, i+1, len(combos))

    if result is not None:
        # Store both the combo params and the results
        row = {**combo, **result}
        all_results.append(row)

print(f"\n  Completed {len(all_results)}/{len(combos)} runs in {(time.time()-start_time)/60:.1f} minutes")

# ──────────────────────────────────────────────────────────────
# SAVE AND ANALYZE RESULTS
# ──────────────────────────────────────────────────────────────
os.makedirs('analysis', exist_ok=True)
df = pd.DataFrame(all_results)
df.to_csv('analysis/param_sweep_results.csv', index=False)
print(f"\n  Saved all results to: analysis/param_sweep_results.csv")

# ── Compute composite score (mimicking competition scoring) ──
# Normalize each metric to [0, 1] range
df['score_return'] = df['total_return'].clip(lower=-0.5, upper=1.0)
df['score_return'] = (df['score_return'] - df['score_return'].min()) / \
                      (df['score_return'].max() - df['score_return'].min() + 1e-10)

df['score_sharpe'] = df['sharpe'].clip(lower=-2, upper=4)
df['score_sharpe'] = (df['score_sharpe'] - df['score_sharpe'].min()) / \
                      (df['score_sharpe'].max() - df['score_sharpe'].min() + 1e-10)

df['score_sortino'] = df['sortino'].clip(lower=-2, upper=5)
df['score_sortino'] = (df['score_sortino'] - df['score_sortino'].min()) / \
                       (df['score_sortino'].max() - df['score_sortino'].min() + 1e-10)

df['score_calmar'] = df['calmar'].clip(lower=-2, upper=4)
df['score_calmar'] = (df['score_calmar'] - df['score_calmar'].min()) / \
                      (df['score_calmar'].max() - df['score_calmar'].min() + 1e-10)

# Composite: balanced across all four metrics
df['composite'] = (0.25 * df['score_return'] +
                   0.25 * df['score_sharpe'] +
                   0.25 * df['score_sortino'] +
                   0.25 * df['score_calmar'])

df = df.sort_values('composite', ascending=False)

# ── Print top 20 ──
print(f"\n{'='*120}")
print(f"TOP 20 PARAMETER COMBINATIONS (by composite score)")
print(f"{'='*120}")
print(f"{'Rank':>4} {'Return':>8} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} "
      f"{'Calmar':>7} {'Comm%':>7} {'Trades':>6} {'AvgWt':>6} "
      f"{'Rebal':>5} {'Hold':>4} {'Thresh':>6} {'Regime':>10} {'DD':>10} "
      f"{'Stop':>6} {'LB':>8} {'Score':>6}")
print('-' * 120)

for rank, (_, row) in enumerate(df.head(20).iterrows(), 1):
    stop_str = f"{row['position_stop_loss']:.0%}" if row['position_stop_loss'] is not None else "off"
    print(f"{rank:>4} "
          f"{row['total_return']:>7.1%} "
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
          f"{row['drawdown_mode']:>10} "
          f"{stop_str:>6} "
          f"{row['lookback_config']:>8} "
          f"{row['composite']:>6.3f}")

# ── Print worst 5 (for contrast) ──
print(f"\nBOTTOM 5 (worst):")
for rank, (_, row) in enumerate(df.tail(5).iterrows(), 1):
    stop_str = f"{row['position_stop_loss']:.0%}" if row['position_stop_loss'] is not None else "off"
    print(f"  {row['total_return']:>7.1%} "
          f"Sharpe={row['sharpe']:>5.2f} "
          f"MaxDD={row['max_drawdown']:>6.1%} "
          f"Comm={row['commission_pct']:>5.1%} "
          f"regime={row['regime_mode']} "
          f"dd={row['drawdown_mode']} "
          f"rebal={row['rebalance_frequency']:.0f}h")

# ── Parameter importance: which parameter matters most? ──
print(f"\n{'='*80}")
print(f"PARAMETER IMPORTANCE (average composite score by parameter value)")
print(f"{'='*80}")

for param in keys:
    print(f"\n  {param}:")
    group = df.groupby(param)['composite'].mean().sort_values(ascending=False)
    for val, score in group.items():
        # Also show avg return and avg commission for context
        subset = df[df[param] == val]
        avg_ret = subset['total_return'].mean()
        avg_comm = subset['commission_pct'].mean()
        avg_sharpe = subset['sharpe'].mean()
        print(f"    {str(val):>15}: score={score:.3f}  "
              f"ret={avg_ret:>6.1%}  comm={avg_comm:>5.1%}  sharpe={avg_sharpe:>5.2f}")

# ── Best combo summary ──
best = df.iloc[0]
print(f"\n{'='*80}")
print(f"RECOMMENDED PARAMETERS (highest composite score)")
print(f"{'='*80}")
print(f"  rebalance_frequency : {best['rebalance_frequency']:.0f}h")
print(f"  num_holdings        : {best['num_holdings']:.0f}")
print(f"  rebalance_threshold : {best['rebalance_threshold']:.0%}")
print(f"  regime_mode         : {best['regime_mode']}")
print(f"  drawdown_mode       : {best['drawdown_mode']}")
print(f"  position_stop_loss  : {best['position_stop_loss']}")
print(f"  lookback_config     : {best['lookback_config']}")
print(f"\n  Expected performance:")
print(f"    Return    : {best['total_return']:.1%}")
print(f"    Sharpe    : {best['sharpe']:.2f}")
print(f"    Sortino   : {best['sortino']:.2f}")
print(f"    Max DD    : {best['max_drawdown']:.1%}")
print(f"    Calmar    : {best['calmar']:.2f}")
print(f"    Commission: {best['commission_pct']:.1%}")
print(f"{'='*80}")