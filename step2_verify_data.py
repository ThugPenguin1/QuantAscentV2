import pandas as pd
from backtest.data_prep import prepare_backtest_data

with open('data/universe.txt') as f:
    universe = [line.strip() for line in f if line.strip()]

print(f"Universe: {len(universe)} coins")
print(f"Coins: {universe}\n")

data = prepare_backtest_data(universe)

close = data['close']
returns = data['returns']

print(f"\nVERIFICATION:")
print(f"  Timestamps     : {len(close)}")
print(f"  Coins           : {len(close.columns)}")
print(f"  Date range      : {close.index[0]} to {close.index[-1]}")
print(f"  NaN in close    : {close.isna().sum().sum()}")
print(f"  NaN in returns  : {returns.iloc[1:].isna().sum().sum()}")

print(f"\nPer-coin data coverage:")
for coin in universe:
    filepath = f'data/raw/{coin}_1h.csv'
    try:
        raw = pd.read_csv(filepath)
        raw_len = len(raw)
    except:
        raw_len = 0
    pct = len(close) / raw_len * 100 if raw_len > 0 else 0
    status = "OK" if pct > 90 else "LOW"
    print(f"  {coin:12s}: {raw_len:>5d} raw -> {len(close):>5d} aligned ({pct:>5.1f}%) {status}")

# Quick sanity: are prices reasonable?
print(f"\nPrice sanity check (latest):")
latest = close.iloc[-1]
for coin in ['BTC', 'ETH', 'SOL']:
    if coin in latest.index:
        print(f"  {coin}: ${latest[coin]:,.2f}")