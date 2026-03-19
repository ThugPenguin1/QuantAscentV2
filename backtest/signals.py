import pandas as pd
import numpy as np


def compute_momentum_scores(close_df, returns_df,
                            lookback_windows=[24, 72, 168],
                            lookback_weights=[0.2, 0.3, 0.5],
                            skip_periods=2):
    """
    Blended momentum score for each coin at each timestamp.
    Positive score = coin is trending UP.
    Negative score = coin is trending DOWN.
    """
    scores = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)

    for window, weight in zip(lookback_windows, lookback_weights):
        if skip_periods > 0:
            end_price = close_df.shift(skip_periods)
            start_price = close_df.shift(window + skip_periods)
        else:
            end_price = close_df
            start_price = close_df.shift(window)

        window_return = (end_price / start_price) - 1.0
        scores += weight * window_return

    return scores


def rank_momentum(scores):
    """Cross-sectional rank. Rank 1 = highest momentum."""
    return scores.rank(axis=1, ascending=False, method='average')


def select_top_n(ranks, n=6):
    """Boolean DataFrame: True if coin is in top N."""
    return ranks.le(n)


def select_top_n_positive(scores, ranks, n=6):
    """
    THE FIX: Select top N coins, BUT only if their momentum is positive.
    
    If 3 coins have positive momentum and n=6, select only those 3.
    If 0 coins have positive momentum, select nothing (all cash).
    
    This prevents buying coins that are going DOWN just because
    they're "the least bad."
    
    Chan: "Time-series momentum asks 'is this asset trending?'
    Cross-sectional asks 'which assets are trending most?'
    The combination is more powerful than either alone."
    """
    # Must be in top N by rank
    in_top_n = ranks.le(n)
    
    # Must have positive momentum score
    is_positive = scores > 0
    
    # Both conditions must be true
    selected = in_top_n & is_positive
    
    return selected