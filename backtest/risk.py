import pandas as pd
import numpy as np


class RiskManager:

    def __init__(self,
                 drawdown_reduce=-0.07,
                 drawdown_panic=-0.15,
                 drawdown_recovery=-0.04,
                 position_stop_loss=-0.10,
                 cooldown_periods=12,
                 use_trailing_stop=True,
                 regime_caution_threshold=-0.01,
                 regime_bear_threshold=-0.05,
                 regime_caution_scalar=0.5,
                 regime_bear_scalar=0.0):
        self.dd_reduce = drawdown_reduce
        self.dd_panic = drawdown_panic
        self.dd_recovery = drawdown_recovery
        self.stop_loss = position_stop_loss
        self.cooldown = cooldown_periods
        self.use_trailing = use_trailing_stop
        self.regime_caution_thresh = regime_caution_threshold
        self.regime_bear_thresh = regime_bear_threshold
        self.regime_caution_scalar = regime_caution_scalar
        self.regime_bear_scalar = regime_bear_scalar

    def compute_regime_filter(self, momentum_scores):
        avg = momentum_scores.mean(axis=1)
        smoothed = avg.rolling(window=48, min_periods=24).mean()

        scalar = pd.Series(1.0, index=smoothed.index)
        scalar[smoothed < self.regime_caution_thresh] = self.regime_caution_scalar
        scalar[smoothed < self.regime_bear_thresh] = self.regime_bear_scalar
        return scalar

    def apply_stop_losses(self, weights, close_df):
        w = weights.values.copy().astype(np.float64)
        px = close_df.values.astype(np.float64)
        n_times, n_coins = w.shape

        ref_price = np.full(n_coins, np.nan)
        cooldown_end = np.full(n_coins, -1, dtype=np.int64)

        for i in range(n_times):
            for j in range(n_coins):
                if i < cooldown_end[j]:
                    w[i, j] = 0.0
                    continue

                price = px[i, j]
                wt = w[i, j]

                if wt > 1e-8:
                    if np.isnan(ref_price[j]):
                        ref_price[j] = price
                    else:
                        if self.use_trailing:
                            if price > ref_price[j]:
                                ref_price[j] = price
                        drop = price / ref_price[j] - 1.0
                        if drop <= self.stop_loss:
                            w[i, j] = 0.0
                            ref_price[j] = np.nan
                            cooldown_end[j] = i + self.cooldown
                else:
                    ref_price[j] = np.nan

        return pd.DataFrame(w, index=weights.index, columns=weights.columns)