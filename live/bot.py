from telegram_notifier import send_telegram
import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from roostoo_api import RoostooAPI

load_dotenv()

API_KEY = os.getenv("ROOSTOO_API_KEY")
API_SECRET = os.getenv("ROOSTOO_API_SECRET")
BASE_URL = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")

PARAMS = {
    'lookback_windows': [48, 168, 336],
    'lookback_weights': [0.2, 0.3, 0.5],
    'skip_periods': 2,
    'num_holdings': 4,
    'rebalance_hours': 12,
    'rebalance_threshold': 0.03,
    'paxg_floor': 0.20,
    'max_total_exposure': 0.85,
    'max_position_weight': 0.25,
    'regime_caution_threshold': -0.01,
    'regime_bear_threshold': -0.03,
    'vol_lookback': 72,
    'target_vol': 0.50,
    'avg_correlation': 0.60,
}

UNIVERSE = [
    "BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "SUI", "PEPE",
    "ZEC", "ADA", "LINK", "AVAX", "TRUMP", "LTC", "UNI",
    "BONK", "TAO", "NEAR", "AAVE", "PENGU", "VIRTUAL",
    "PAXG", "TRX", "ENA", "ARB", "WLD", "WIF",
]

PAIR_PRECISION = {
    "BTC/USD": (2, 5), "ETH/USD": (2, 4), "SOL/USD": (2, 3),
    "XRP/USD": (4, 1), "BNB/USD": (2, 3), "DOGE/USD": (5, 0),
    "SUI/USD": (4, 1), "PEPE/USD": (8, 0), "ZEC/USD": (2, 3),
    "ADA/USD": (4, 1), "LINK/USD": (2, 2), "AVAX/USD": (2, 2),
    "TRUMP/USD": (3, 3), "LTC/USD": (2, 3), "UNI/USD": (3, 2),
    "BONK/USD": (8, 0), "TAO/USD": (1, 4), "NEAR/USD": (3, 1),
    "AAVE/USD": (2, 3), "PENGU/USD": (6, 0), "VIRTUAL/USD": (4, 1),
    "PAXG/USD": (2, 4), "TRX/USD": (4, 1), "ENA/USD": (4, 2),
    "ARB/USD": (4, 1), "WLD/USD": (4, 1), "WIF/USD": (3, 2),
}

LOG_DIR = "logs"
PRICE_POLL_INTERVAL = 300  # 5 minutes


class PriceHistory:
    def __init__(self, universe):
        self.universe = universe
        self.prices = {coin: [] for coin in universe}

    def bootstrap_from_binance(self, days=20):
        import ccxt
        print(f"[BOOTSTRAP] Downloading {days} days of hourly data from Binance...")
        exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})

        for coin in self.universe:
            pair = f"{coin}/USDT"
            try:
                since = exchange.parse8601(
                    (datetime.utcnow() - timedelta(days=days)).isoformat()
                )
                candles = []
                while True:
                    batch = exchange.fetch_ohlcv(pair, '1h', since=since, limit=1000)
                    if not batch:
                        break
                    candles.extend(batch)
                    since = batch[-1][0] + 1
                    if len(batch) < 1000:
                        break
                    time.sleep(0.1)

                for c in candles:
                    ts = datetime.utcfromtimestamp(c[0] / 1000)
                    self.prices[coin].append((ts, c[4]))

                print(f"  {coin}: {len(candles)} candles")
            except Exception as e:
                print(f"  {coin}: FAILED - {e}")
            time.sleep(0.2)

        print(f"[BOOTSTRAP] Done. {sum(len(v) for v in self.prices.values())} total candles.")

    def add_live_prices(self, ticker_data):
        now = datetime.utcnow()
        for coin in self.universe:
            pair = f"{coin}/USD"
            if pair in ticker_data:
                price = ticker_data[pair].get("LastPrice")
                if price and price > 0:
                    self.prices[coin].append((now, float(price)))

    def get_dataframe(self):
        frames = {}
        for coin, history in self.prices.items():
            if not history:
                continue
            df = pd.DataFrame(history, columns=["timestamp", "close"])
            df = df.set_index("timestamp")
            df = df[~df.index.duplicated(keep="last")]
            df = df.resample("1h").last().ffill()
            frames[coin] = df["close"]

        if not frames:
            return None

        return pd.DataFrame(frames).ffill().dropna()

    def hours_available(self):
        lengths = [len(v) for v in self.prices.values() if v]
        return min(lengths) if lengths else 0


def compute_signals(price_df, params):
    if len(price_df) < max(params['lookback_windows']) + params['skip_periods'] + 10:
        return None, None, 1.0

    returns_df = price_df.pct_change()

    scores = pd.Series(0.0, index=price_df.columns)
    for window, weight in zip(params['lookback_windows'], params['lookback_weights']):
        skip = params['skip_periods']
        end_price = price_df.iloc[-(skip + 1)] if skip > 0 else price_df.iloc[-1]
        start_price = price_df.iloc[-(window + skip + 1)]
        window_ret = (end_price / start_price) - 1.0
        scores += weight * window_ret

    positive = scores[scores > 0].sort_values(ascending=False)
    selected = positive.head(params['num_holdings']).index.tolist()
    selected = [c for c in selected if c != 'PAXG']

    avg_mom = scores.mean()
    if avg_mom < params['regime_bear_threshold']:
        regime_scalar = 0.0
    elif avg_mom < params['regime_caution_threshold']:
        regime_scalar = 0.5
    else:
        regime_scalar = 1.0

    if regime_scalar <= 0.0:
        return {}, selected, regime_scalar

    if not selected:
        return {}, selected, regime_scalar

    vol = returns_df[selected].tail(params['vol_lookback']).std() * np.sqrt(24 * 365)
    vol = vol[vol > 0.05].dropna()
    valid_selected = [c for c in selected if c in vol.index]

    if not valid_selected:
        return {}, selected, regime_scalar

    vol = vol[valid_selected]
    inv_vol = 1.0 / vol
    raw_w = inv_vol / inv_vol.sum()
    raw_w *= regime_scalar

    avg_corr = params['avg_correlation']
    indiv_var = (raw_w ** 2 * vol ** 2).sum()
    cross_var = avg_corr * ((raw_w * vol).sum() ** 2)
    port_vol = np.sqrt((1 - avg_corr) * indiv_var + cross_var)

    if port_vol > 0:
        scalar = params['target_vol'] / port_vol
        raw_w *= scalar

    raw_w = raw_w.clip(upper=params['max_position_weight'])

    max_momentum_weight = params['max_total_exposure'] - params['paxg_floor']
    if raw_w.sum() > max_momentum_weight:
        raw_w *= max_momentum_weight / raw_w.sum()

    weights = {coin: float(w) for coin, w in raw_w.items() if w > 0.005}
    return weights, selected, regime_scalar


class Portfolio:
    def __init__(self, api):
        self.api = api
        self.peak_value = 0

    def get_state(self):
        wallet = self.api.balance()
        if not wallet:
            return None, None, None

        ticker = self.api.ticker()
        if not ticker:
            return None, None, None

        usd = float(wallet.get("USD", {}).get("Free", 0))
        usd += float(wallet.get("USD", {}).get("Lock", 0))

        holdings = {}
        total_value = usd

        for coin in UNIVERSE:
            pair = f"{coin}/USD"
            free = float(wallet.get(coin, {}).get("Free", 0))
            lock = float(wallet.get(coin, {}).get("Lock", 0))
            amount = free + lock

            if amount > 0 and pair in ticker:
                price = float(ticker[pair].get("LastPrice", 0))
                value = amount * price
                total_value += value
                holdings[coin] = {
                    "amount": amount,
                    "free": free,
                    "price": price,
                    "value": value,
                }

        if total_value > self.peak_value:
            self.peak_value = total_value

        weights = {}
        for coin, h in holdings.items():
            weights[coin] = h["value"] / total_value if total_value > 0 else 0

        return total_value, weights, holdings

    def get_drawdown(self, total_value):
        if self.peak_value > 0:
            return (total_value - self.peak_value) / self.peak_value
        return 0.0


def execute_rebalance(api, current_weights, target_weights, total_value, ticker_data):
    trades = []

    full_target = dict(target_weights)
    full_target["PAXG"] = max(full_target.get("PAXG", 0), PARAMS["paxg_floor"])

    for coin in set(list(current_weights.keys()) + list(full_target.keys())):
        current_w = current_weights.get(coin, 0)
        target_w = full_target.get(coin, 0)
        drift = abs(target_w - current_w)

        if drift < PARAMS["rebalance_threshold"] and target_w > 0:
            continue

        if current_w > 0 and target_w == 0:
            pass
        elif drift < 0.01:
            continue

        trade_value = (target_w - current_w) * total_value
        pair = f"{coin}/USD"

        if pair not in ticker_data:
            continue

        price = float(ticker_data[pair].get("LastPrice", 0))
        if price <= 0:
            continue

        _, amt_prec = PAIR_PRECISION.get(pair, (4, 2))
        amount = abs(trade_value) / price
        amount = round(amount, amt_prec)

        if amount * price < 1.0:
            continue

        if trade_value < 0:
            trades.append(("SELL", coin, pair, amount, price))
        elif trade_value > 0:
            trades.append(("BUY", coin, pair, amount, price))

    trades.sort(key=lambda x: 0 if x[0] == "SELL" else 1)

    executed = []
    for side, coin, pair, amount, price in trades:
        print(f"  {side} {amount} {coin} (~${amount * price:,.0f})")
        result = api.place_order(pair, side, amount, order_type="MARKET")
        if result:
            executed.append({
                "time": datetime.utcnow().isoformat(),
                "side": side,
                "coin": coin,
                "amount": amount,
                "price": price,
                "value": amount * price,
                "success": result.get("Success", False),
                "order_id": result.get("OrderDetail", {}).get("OrderID"),
                "status": result.get("OrderDetail", {}).get("Status"),
            })
        time.sleep(0.5)

    return executed


class Logger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.trade_file = os.path.join(LOG_DIR, "trades.jsonl")
        self.state_file = os.path.join(LOG_DIR, "portfolio_state.jsonl")
        self.signal_file = os.path.join(LOG_DIR, "signals.jsonl")

    def log_trade(self, trades):
        with open(self.trade_file, "a") as f:
            for t in trades:
                f.write(json.dumps(t) + "\n")

    def log_state(self, state):
        with open(self.state_file, "a") as f:
            f.write(json.dumps(state) + "\n")

    def log_signal(self, signal):
        with open(self.signal_file, "a") as f:
            f.write(json.dumps(signal) + "\n")


def main():
    print("=" * 60)
    print("MOMENTUM + PAXG HEDGE BOT")
    print(f"Started: {datetime.utcnow().isoformat()}")
    print("=" * 60)

    send_telegram(
        f"🚀 Bot started\n"
        f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"Strategy: Momentum + PAXG Hedge\n"
        f"Rebalance: every {PARAMS['rebalance_hours']}h"
    )

    if not API_KEY or not API_SECRET:
        print("Missing API credentials in .env")
        return

    api = RoostooAPI(API_KEY, API_SECRET, BASE_URL)
    portfolio = Portfolio(api)
    history = PriceHistory(UNIVERSE)
    logger = Logger()

    print("\n[1] Checking API connection...")
    st = api.server_time()
    if st:
        print(f"  Server time: {st}")
    else:
        print("  FAILED to connect.")
        return

    print("\n[2] Getting initial balance...")
    total_value, current_weights, holdings = portfolio.get_state()
    if total_value is None:
        print("  FAILED to get balance.")
        return
    print(f"  Total value: ${total_value:,.2f}")
    print(f"  Current positions: {list(current_weights.keys()) if current_weights else 'USD only'}")

    print("\n[3] Bootstrapping price history from Binance...")
    history.bootstrap_from_binance(days=20)
    print(f"  Hours available: {history.hours_available()}")

    print("\n[4] Checking PAXG hedge...")
    paxg_weight = current_weights.get("PAXG", 0)
    if paxg_weight < PARAMS["paxg_floor"] * 0.5:
        print(f"  PAXG weight: {paxg_weight:.1%} < target {PARAMS['paxg_floor']:.0%}")
        print("  Buying initial PAXG hedge...")
        ticker = api.ticker("PAXG/USD")
        if ticker and "PAXG/USD" in ticker:
            paxg_price = float(ticker["PAXG/USD"]["LastPrice"])
            buy_value = total_value * PARAMS["paxg_floor"]
            buy_amount = round(buy_value / paxg_price, 4)
            print(f"  Buying {buy_amount} PAXG @ ${paxg_price:,.2f} (~${buy_value:,.0f})")
            result = api.place_order("PAXG/USD", "BUY", buy_amount, order_type="MARKET")
            if result and result.get("Success"):
                print("  PAXG hedge established!")
                logger.log_trade([{
                    "time": datetime.utcnow().isoformat(),
                    "side": "BUY",
                    "coin": "PAXG",
                    "amount": buy_amount,
                    "price": paxg_price,
                    "value": buy_value,
                    "success": True,
                    "reason": "initial_hedge",
                }])
            time.sleep(2)
    else:
        print(f"  PAXG weight: {paxg_weight:.1%} — OK")

    print("\n[5] Entering main loop...")
    print(f"  Rebalance every {PARAMS['rebalance_hours']} hours")
    print(f"  Price poll every {PRICE_POLL_INTERVAL} seconds")

    last_rebalance = datetime.utcnow() - timedelta(hours=PARAMS["rebalance_hours"])

    while True:
        try:
            now = datetime.utcnow()

            ticker = api.ticker()
            if ticker:
                history.add_live_prices(ticker)

            hours_since = (now - last_rebalance).total_seconds() / 3600

            if hours_since >= PARAMS["rebalance_hours"]:
                print(f"\n{'='*60}")
                print(f"REBALANCE at {now.strftime('%Y-%m-%d %H:%M UTC')}")
                print(f"{'='*60}")

                total_value, current_weights, holdings = portfolio.get_state()
                if total_value is None:
                    print("  Failed to get portfolio state. Skipping.")
                    last_rebalance = now
                    continue

                dd = portfolio.get_drawdown(total_value)
                print(f"  Portfolio: ${total_value:,.2f}  DD: {dd:.1%}")
                print(f"  Current: {json.dumps({k: f'{v:.1%}' for k, v in current_weights.items()})}")

                if dd <= -0.10:
                    dd_scalar = 0.2
                elif dd <= -0.05:
                    dd_scalar = 0.5
                else:
                    dd_scalar = 1.0

                price_df = history.get_dataframe()
                if price_df is None or len(price_df) < 200:
                    print("  Not enough price data. Skipping.")
                    last_rebalance = now
                    continue

                momentum_weights, selected, regime_scalar = compute_signals(price_df, PARAMS)
                combined = min(dd_scalar, regime_scalar)

                print(f"  Regime: {regime_scalar:.1f}  DD scalar: {dd_scalar:.1f}  Combined: {combined:.1f}")
                print(f"  Selected coins: {selected}")

                if momentum_weights is None:
                    momentum_weights = {}

                target_weights = {}
                if combined <= 0:
                    target_weights["PAXG"] = PARAMS["paxg_floor"]
                    print("  BEAR MODE: Going to PAXG only")
                else:
                    for coin, w in momentum_weights.items():
                        target_weights[coin] = w * combined
                    target_weights["PAXG"] = max(target_weights.get("PAXG", 0), PARAMS["paxg_floor"])

                print(f"  Target: {json.dumps({k: f'{v:.1%}' for k, v in target_weights.items()})}")

                send_telegram(
                    f"🔄 REBALANCE\n"
                    f"Time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                    f"Portfolio: ${total_value:,.2f}\n"
                    f"Drawdown: {dd:.2%}\n"
                    f"Regime: {regime_scalar:.1f}\n"
                    f"DD scalar: {dd_scalar:.1f}\n"
                    f"Selected: {', '.join(selected) if selected else 'None'}\n"
                    f"Target: {json.dumps({k: f'{v:.1%}' for k, v in target_weights.items()})}"
                )

                logger.log_signal({
                    "time": now.isoformat(),
                    "regime_scalar": regime_scalar,
                    "dd_scalar": dd_scalar,
                    "combined": combined,
                    "selected": selected,
                    "target_weights": {k: round(v, 4) for k, v in target_weights.items()},
                    "total_value": total_value,
                    "drawdown": dd,
                })

                trades = execute_rebalance(api, current_weights, target_weights, total_value, ticker)

                if trades:
                    logger.log_trade(trades)
                    print(f"  Executed {len(trades)} trades")

                    trade_lines = []
                    for t in trades:
                        trade_lines.append(
                            f"{t['side']} {t['coin']} "
                            f"amt={t['amount']} "
                            f"~${t['value']:,.0f}"
                        )

                    send_telegram(
                        f"✅ Trades Executed ({len(trades)})\n"
                        + "\n".join(trade_lines)
                        + f"\nPortfolio: ${total_value:,.2f}"
                    )
                else:
                    print("  No trades needed")
                    send_telegram(
                        f"ℹ️ No trades needed\n"
                        f"Time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                        f"Portfolio: ${total_value:,.2f}\n"
                        f"Selected: {', '.join(selected) if selected else 'None'}"
                    )

                time.sleep(3)
                total_value2, weights2, _ = portfolio.get_state()
                logger.log_state({
                    "time": now.isoformat(),
                    "total_value": total_value2,
                    "weights": {k: round(v, 4) for k, v in (weights2 or {}).items()},
                    "drawdown": portfolio.get_drawdown(total_value2) if total_value2 else None,
                    "num_trades": len(trades),
                })

                last_rebalance = now

            else:
                total_value, current_weights, _ = portfolio.get_state()
                if total_value:
                    dd = portfolio.get_drawdown(total_value)

                    if current_weights:
                        top = sorted(current_weights.items(), key=lambda x: x[1], reverse=True)[:5]
                        top_str = ", ".join([f"{coin}:{w:.1%}" for coin, w in top if w > 0.001])
                    else:
                        top_str = "USD only"

                    hours_to_rebal = max(0, PARAMS["rebalance_hours"] - hours_since)

                    print(
                        f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"Value=${total_value:,.2f} | "
                        f"DD={dd:.2%} | "
                        f"Next rebalance in {hours_to_rebal:.1f}h | "
                        f"Holdings: {top_str}"
                    )

        except KeyboardInterrupt:
            print("\n\nBot stopped by user.")
            send_telegram("🛑 Bot stopped manually.")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            tb = traceback.format_exc()
            print(tb)

            send_telegram(
                f"❌ BOT ERROR\n"
                f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"Error: {str(e)}"
            )

        time.sleep(PRICE_POLL_INTERVAL)


if __name__ == "__main__":
    main()