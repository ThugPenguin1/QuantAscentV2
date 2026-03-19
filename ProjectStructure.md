# Create project structure
"""
momentum_strategy/
├── data/
│   ├── raw/           # Raw OHLCV downloads
│   ├── processed/     # Cleaned, aligned data
├── backtest/
│   ├── engine.py      # Backtest framework
│   ├── signals.py     # Momentum signal construction
│   ├── sizing.py      # Volatility targeting & Kelly
│   ├── risk.py        # Drawdown circuit breaker, regime filter
│   ├── costs.py       # Commission modeling
│   └── run_backtest.py
├── analysis/
│   ├── universe_selection.py
│   └── results_analysis.py
├── live/
│   └── bot.py         # For later
├── config.py
└── requirements.txt
"""

"""momentum_strategy/
├── config.py                 # Coin lists, fees, precision, helpers
├── data/
│   ├── raw/                  # 67 CSV files (already downloaded)
│   ├── universe.txt          # Output of universe selection
│   └── universe_stats.csv    # Stats for review
├── backtest/
│   ├── __init__.py
│   ├── data_prep.py          # Load, align, validate data
│   ├── signals.py            # Momentum scoring and ranking
│   ├── sizing.py             # Inverse vol weights, vol targeting
│   ├── risk.py               # Regime filter, drawdown breaker, stops
│   ├── engine.py             # Main backtest orchestrator + simulator
│   └── costs.py              # Commission modeling
├── analysis/
│   ├── universe_selection.py # Filter and rank coins
│   ├── run_backtest.py       # Run backtest with params
│   ├── sensitivity.py        # Parameter sensitivity (later)
│   └── walk_forward.py       # Walk-forward optimization (later)
└── live/
    └── bot.py                # Live trading (later)"""

""" ROOSTOO COINS = ['OPENUSDT', 'TRUMPUSDT', 'TONUSDT', 'SUSDT', 'SOLUSDT', 'OMNIUSDT', 'CAKEUSDT', 'ARBUSDT', 'AVNTUSDT', 'PAXGUSDT', 'EDENUSDT', 'HEMIUSDT', 'FETUSDT', 'LINKUSDT', 'FORMUSDT', 'FLOKIUSDT', 'BONKUSDT', 'FILUSDT', 'BTCUSDT', 'TAOUSDT', 'UNIUSDT', 'PEPEUSDT', 'PUMPUSDT', 'HBARUSDT', 'XRPUSDT', 'AAVEUSDT', 'WLFIUSDT', 'EIGENUSDT', 'LINEAUSDT', '1000CHEEMSUSDT', 'BIOUSDT', 'LISTAUSDT', 'AVAXUSDT', 'MIRAUSDT', 'XLMUSDT', 'SUIUSDT', 'NEARUSDT', 'SEIUSDT', 'PENGUUSDT', 'ETHUSDT', 'PENDLEUSDT', 'PLUMEUSDT', 'WIFUSDT', 'ICPUSDT', 'BNBUSDT', 'VIRTUALUSDT', 'APTUSDT', 'SHIBUSDT', 'POLUSDT', 'ZECUSDT', 'DOGEUSDT', 'CRVUSDT', 'ASTERUSDT', 'TRXUSDT', 'BMTUSDT', 'ZENUSDT', 'ONDOUSDT', 'LTCUSDT', 'STOUSDT', 'SOMIUSDT', 'WLDUSDT', 'XPLUSDT', 'CFXUSDT', 'DOTUSDT', 'TUTUSDT', 'ADAUSDT', 'ENAUSDT'] 
"""