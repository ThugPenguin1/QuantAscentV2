# Roostoo Momentum + PAXG Hedge Bot

This repo contains:
- Backtesting framework for cross-sectional crypto momentum
- Live bot for Roostoo mock exchange
- Risk overlays including:
  - absolute momentum filter
  - regime filter
  - inverse volatility sizing
  - PAXG hedge
  - drawdown circuit breaker

## Strategy Summary
- Long-only
- Select top positive-momentum coins
- Allocate by inverse volatility
- Maintain permanent PAXG hedge
- Reduce or eliminate exposure in bearish regimes
- Rebalance every 12 hours

## Setup

```bash
pip install -r requirements.txt