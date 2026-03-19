import pandas as pd
import numpy as np
from backtest.signals import compute_momentum_scores, rank_momentum, select_top_n, select_top_n_positive
from backtest.sizing import compute_coin_volatility, inverse_volatility_weights, apply_rebalance_threshold
from backtest.risk import RiskManager


class MomentumBacktest:

    def __init__(self, params):
        self.p = params
        self.results = {}

    def run(self, data):
        close_df = data['close']
        returns_df = data['returns']
        p = self.p

        print(f"\n{'='*70}")
        print(f"BACKTEST")
        print(f"{'='*70}")
        print(f"  {len(close_df)} timestamps | {len(close_df.columns)} coins")
        print(f"  {close_df.index[0]} to {close_df.index[-1]}")

        print("\n[1] Momentum scores...")
        mom = compute_momentum_scores(
            close_df, returns_df,
            lookback_windows=p['lookback_windows'],
            lookback_weights=p['lookback_weights'],
            skip_periods=p['skip_periods']
        )
        print(f"    First valid signal: {mom.first_valid_index()}")

        print("[2] Ranking...")
        ranks = rank_momentum(mom)

        # THE KEY CHANGE: use absolute momentum filter if enabled
        if p.get('require_positive_momentum', True):
            selected = select_top_n_positive(mom, ranks, n=p['num_holdings'])
            avg_selected = selected.sum(axis=1).mean()
            print(f"    Absolute momentum filter ON: avg {avg_selected:.1f} coins selected per period")
            
            # Show how often we'd hold 0, 1-3, 4-6 coins
            counts = selected.sum(axis=1)
            pct_zero = (counts == 0).mean()
            pct_few = ((counts > 0) & (counts <= 3)).mean()
            pct_full = (counts > 3).mean()
            print(f"    Zero holdings: {pct_zero:.1%} | 1-3 coins: {pct_few:.1%} | 4+ coins: {pct_full:.1%}")
        else:
            selected = select_top_n(ranks, n=p['num_holdings'])
            print(f"    Absolute momentum filter OFF: always {p['num_holdings']} coins")

        rebal = p['rebalance_frequency']
        rebal_mask = pd.Series(False, index=close_df.index)
        rebal_mask.iloc[::rebal] = True
        print(f"[3] Rebalance every {rebal}h ({rebal_mask.sum()} rebalances)")

        print("[4] Volatility estimation...")
        coin_vol = compute_coin_volatility(returns_df, vol_lookback=p['vol_lookback'])

        print("[5] Inverse-vol sizing...")
        raw_weights = inverse_volatility_weights(
            selected, coin_vol,
            target_vol=p['target_vol'],
            max_position_weight=p['max_position_weight'],
            max_total_exposure=p['max_total_exposure'],
            avg_correlation=p.get('avg_correlation', 0.60),
            rebalance_mask=rebal_mask,
        )

        print("[6] Rebalance threshold filter...")
        practical = apply_rebalance_threshold(raw_weights, threshold=p['rebalance_threshold'])

        print("[7] Risk management...")
        risk = RiskManager(
            drawdown_reduce=p['drawdown_reduce'],
            drawdown_panic=p['drawdown_panic'],
            drawdown_recovery=p['drawdown_recovery'],
            position_stop_loss=p['position_stop_loss'],
            cooldown_periods=p.get('cooldown_periods', 12),
            use_trailing_stop=p.get('trailing_stop', True),
            regime_caution_threshold=p.get('regime_caution_threshold', -0.01),
            regime_bear_threshold=p.get('regime_bear_threshold', -0.05),
            regime_caution_scalar=p.get('regime_caution_scalar', 0.5),
            regime_bear_scalar=p.get('regime_bear_scalar', 0.0),
        )

        regime_scalar = None
        if p.get('use_regime_filter', True):
            regime_scalar = risk.compute_regime_filter(mom)
            pct_bull = (regime_scalar >= 1.0).mean()
            pct_caution = ((regime_scalar < 1.0) & (regime_scalar > 0.0)).mean()
            pct_bear = (regime_scalar <= 0.0).mean()
            print(f"    Bull: {pct_bull:.1%} | Caution: {pct_caution:.1%} | Bear: {pct_bear:.1%}")

        if p.get('use_stops', False) and p.get('position_stop_loss', -0.99) > -0.50:
            practical = risk.apply_stop_losses(practical, close_df)

        print("[8] Simulating...")
        equity, trade_log = self._simulate(practical, returns_df, regime_scalar)

        metrics = self._metrics(equity, trade_log)

        self.results = {
            'equity': equity,
            'weights': practical,
            'momentum_scores': mom,
            'ranks': ranks,
            'selected': selected,
            'coin_vol': coin_vol,
            'regime_scalar': regime_scalar,
            'trade_log': trade_log,
            'metrics': metrics,
        }

        print(f"\n{'='*70}")
        print(f"COMPLETE")
        print(f"{'='*70}")
        return self.results

    def _simulate(self, weights, returns_df, regime_scalar):
        capital = self.p['initial_capital']
        fee = 0.0005 if self.p['use_limit_orders'] else 0.001
        paxg_floor = self.p.get('paxg_floor', 0.0)
        has_paxg = 'PAXG' in weights.columns

        equity_values = [capital]
        trade_log = []
        prev_w = pd.Series(0.0, index=weights.columns)

        peak = capital
        reduced_mode = False
        use_regime = regime_scalar is not None

        for i in range(1, len(weights)):
            idx = weights.index[i]
            ret = returns_df.iloc[i].fillna(0.0)

            # 1. Return from current holdings
            port_ret = (prev_w * ret).sum()
            capital *= (1.0 + port_ret)

            if capital > peak:
                peak = capital

            # 2. Drawdown
            dd = (capital - peak) / peak if peak > 0 else 0.0

            if dd <= self.p['drawdown_panic']:
                dd_scalar = 0.2
                reduced_mode = True
            elif dd <= self.p['drawdown_reduce']:
                dd_scalar = 0.5
                reduced_mode = True
            elif reduced_mode and dd >= self.p['drawdown_recovery']:
                dd_scalar = 1.0
                reduced_mode = False
            elif reduced_mode:
                dd_scalar = 0.5
            else:
                dd_scalar = 1.0

            # 3. Regime
            r_scalar = 1.0
            if use_regime and i < len(regime_scalar):
                r_scalar = regime_scalar.iloc[i]

            combined = min(dd_scalar, r_scalar)

            # 4. Target weights from signals
            if combined <= 0.0:
                target_w = pd.Series(0.0, index=weights.columns)
            else:
                target_w = weights.iloc[i] * combined

            # 5. PAXG FLOOR: always hold minimum PAXG regardless of signals
            #    PAXG is a hedge — hold it in ALL regimes including bear
            if has_paxg and paxg_floor > 0:
                if target_w['PAXG'] < paxg_floor:
                    target_w['PAXG'] = paxg_floor

                # If total exceeds max, scale down non-PAXG positions
                total = target_w.sum()
                max_exp = self.p['max_total_exposure']
                if total > max_exp:
                    non_paxg = target_w.drop('PAXG')
                    non_paxg_total = non_paxg.sum()
                    if non_paxg_total > 0:
                        scale = (max_exp - paxg_floor) / non_paxg_total
                        scale = max(scale, 0.0)
                        target_w[non_paxg.index] = non_paxg * scale
                    target_w['PAXG'] = paxg_floor

            # 6. Commission
            w_change = (target_w - prev_w).abs().sum()
            commission = w_change * fee * capital
            capital -= commission

            if capital <= 0:
                capital = 0.0
                prev_w *= 0.0
                equity_values.append(0.0)
                continue

            prev_w = target_w.copy()
            equity_values.append(capital)

            # 7. Log
            if w_change > 0.001:
                trade_log.append({
                    'timestamp': idx,
                    'turnover': w_change,
                    'commission': commission,
                    'equity': capital,
                    'dd_scalar': dd_scalar,
                    'regime_scalar': r_scalar,
                    'combined_scalar': combined,
                    'drawdown': dd,
                    'num_positions': int((target_w > 0.001).sum()),
                    'total_weight': target_w.sum(),
                    'port_return': port_ret,
                })

        equity = pd.Series(equity_values, index=weights.index[:len(equity_values)])
        trade_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
        return equity, trade_df
    def _metrics(self, equity, trade_log):
        ret = equity.pct_change().dropna()
        total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
        hours = len(equity)
        years = hours / (24.0 * 365.0)
        ann_return = (1.0 + total_return) ** (1.0 / max(years, 0.01)) - 1.0

        sharpe = (ret.mean() / ret.std()) * np.sqrt(24 * 365) if ret.std() > 0 else 0.0

        downside = np.minimum(ret, 0.0)
        dd_dev = np.sqrt((downside ** 2).mean())
        sortino = (ret.mean() / dd_dev) * np.sqrt(24 * 365) if dd_dev > 0 else 0.0

        pk = equity.expanding().max()
        drawdown = (equity - pk) / pk
        max_dd = drawdown.min()
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

        if len(trade_log) > 0:
            total_comm = trade_log['commission'].sum()
            total_turnover = trade_log['turnover'].sum()
        else:
            total_comm = 0.0
            total_turnover = 0.0

        return {
            'total_return': total_return,
            'annualized_return': ann_return,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd,
            'calmar': calmar,
            'total_turnover': total_turnover,
            'total_commission': total_comm,
            'commission_pct': total_comm / self.p['initial_capital'],
            'win_rate': (ret > 0).mean(),
            'final_equity': equity.iloc[-1],
            'peak_equity': equity.max(),
            'hours': hours,
            'years': years,
        }