import pandas as pd
import numpy as np


def compute_coin_volatility(returns_df, vol_lookback=72):
    hourly_vol = returns_df.rolling(
        window=vol_lookback,
        min_periods=max(vol_lookback // 2, 20)
    ).std()
    return hourly_vol * np.sqrt(24 * 365)


def inverse_volatility_weights(selected, coin_vol,
                               target_vol=0.50,
                               max_position_weight=0.25,
                               max_total_exposure=0.90,
                               avg_correlation=0.60,
                               rebalance_mask=None):
    """
    Inverse-vol sizing. If rebalance_mask is provided, only compute
    weights at rebalance timestamps and forward-fill the rest.
    This is 20-50x faster than computing every hour.
    """
    weights = pd.DataFrame(0.0, index=selected.index, columns=selected.columns)

    if rebalance_mask is not None:
        compute_indices = selected.index[rebalance_mask]
    else:
        compute_indices = selected.index

    for idx in compute_indices:
        active = selected.columns[selected.loc[idx]]
        if len(active) == 0:
            continue

        vols = coin_vol.loc[idx, active].dropna()
        vols = vols[vols > 0.05]
        if len(vols) == 0:
            continue

        inv_vol = 1.0 / vols
        raw_w = inv_vol / inv_vol.sum()

        indiv_var = (raw_w ** 2 * vols ** 2).sum()
        cross_var = avg_correlation * ((raw_w * vols).sum() ** 2)
        port_var = (1.0 - avg_correlation) * indiv_var + cross_var
        port_vol = np.sqrt(max(port_var, 1e-10))

        scalar = target_vol / port_vol
        scaled_w = raw_w * scalar
        scaled_w = scaled_w.clip(upper=max_position_weight)

        total = scaled_w.sum()
        if total > max_total_exposure:
            scaled_w *= max_total_exposure / total

        weights.loc[idx, scaled_w.index] = scaled_w.values

    # Forward-fill from rebalance points
    if rebalance_mask is not None:
        weights = weights.ffill()
        weights = weights.fillna(0.0)

    return weights


def apply_rebalance_threshold(target_weights, threshold=0.03):
    actual = target_weights.copy()
    current = target_weights.iloc[0].copy()

    for i in range(1, len(target_weights)):
        target = target_weights.iloc[i]
        drift = (target - current).abs()

        new = current.copy()
        trade = drift > threshold
        new[trade] = target[trade]

        exits = (target == 0) & (current > 0)
        new[exits] = 0.0

        actual.iloc[i] = new
        current = new.copy()

    return actual