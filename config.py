# config.py

# ============================================================
# ROOSTOO TRADING PAIRS (67 coins)
# Format: "COIN/USD" - as used by Roostoo API
# ============================================================

ROOSTOO_PAIRS = [
    "1000CHEEMS/USD", "AAVE/USD", "ADA/USD", "APT/USD", "ARB/USD",
    "ASTER/USD", "AVAX/USD", "AVNT/USD", "BIO/USD", "BMT/USD",
    "BNB/USD", "BONK/USD", "BTC/USD", "CAKE/USD", "CFX/USD",
    "CRV/USD", "DOGE/USD", "DOT/USD", "EDEN/USD", "EIGEN/USD",
    "ENA/USD", "ETH/USD", "FET/USD", "FIL/USD", "FLOKI/USD",
    "FORM/USD", "HBAR/USD", "HEMI/USD", "ICP/USD", "LINEA/USD",
    "LINK/USD", "LISTA/USD", "LTC/USD", "MIRA/USD", "NEAR/USD",
    "OMNI/USD", "ONDO/USD", "OPEN/USD", "PAXG/USD", "PENDLE/USD",
    "PENGU/USD", "PEPE/USD", "PLUME/USD", "POL/USD", "PUMP/USD",
    "S/USD", "SEI/USD", "SHIB/USD", "SOL/USD", "SOMI/USD",
    "STO/USD", "SUI/USD", "TAO/USD", "TON/USD", "TRUMP/USD",
    "TRX/USD", "TUT/USD", "UNI/USD", "VIRTUAL/USD", "WIF/USD",
    "WLD/USD", "WLFI/USD", "XLM/USD", "XPL/USD", "XRP/USD",
    "ZEC/USD", "ZEN/USD"
]

# Base coin symbols (for fetching data from Binance)
ROOSTOO_COINS = [
    "1000CHEEMS", "AAVE", "ADA", "APT", "ARB", "ASTER", "AVAX", "AVNT",
    "BIO", "BMT", "BNB", "BONK", "BTC", "CAKE", "CFX", "CRV", "DOGE",
    "DOT", "EDEN", "EIGEN", "ENA", "ETH", "FET", "FIL", "FLOKI", "FORM",
    "HBAR", "HEMI", "ICP", "LINEA", "LINK", "LISTA", "LTC", "MIRA",
    "NEAR", "OMNI", "ONDO", "OPEN", "PAXG", "PENDLE", "PENGU", "PEPE",
    "PLUME", "POL", "PUMP", "S", "SEI", "SHIB", "SOL", "SOMI", "STO",
    "SUI", "TAO", "TON", "TRUMP", "TRX", "TUT", "UNI", "VIRTUAL", "WIF",
    "WLD", "WLFI", "XLM", "XPL", "XRP", "ZEC", "ZEN"
]

# Precision info for order placement
PAIR_PRECISION = {
    "1000CHEEMS/USD": {"price": 6, "amount": 0},
    "AAVE/USD": {"price": 2, "amount": 3},
    "ADA/USD": {"price": 4, "amount": 1},
    "APT/USD": {"price": 3, "amount": 2},
    "ARB/USD": {"price": 4, "amount": 1},
    "ASTER/USD": {"price": 3, "amount": 2},
    "AVAX/USD": {"price": 2, "amount": 2},
    "AVNT/USD": {"price": 4, "amount": 1},
    "BIO/USD": {"price": 4, "amount": 1},
    "BMT/USD": {"price": 5, "amount": 1},
    "BNB/USD": {"price": 2, "amount": 3},
    "BONK/USD": {"price": 8, "amount": 0},
    "BTC/USD": {"price": 2, "amount": 5},
    "CAKE/USD": {"price": 3, "amount": 2},
    "CFX/USD": {"price": 5, "amount": 0},
    "CRV/USD": {"price": 4, "amount": 1},
    "DOGE/USD": {"price": 5, "amount": 0},
    "DOT/USD": {"price": 3, "amount": 2},
    "EDEN/USD": {"price": 4, "amount": 1},
    "EIGEN/USD": {"price": 3, "amount": 2},
    "ENA/USD": {"price": 4, "amount": 2},
    "ETH/USD": {"price": 2, "amount": 4},
    "FET/USD": {"price": 4, "amount": 1},
    "FIL/USD": {"price": 3, "amount": 2},
    "FLOKI/USD": {"price": 8, "amount": 0},
    "FORM/USD": {"price": 4, "amount": 1},
    "HBAR/USD": {"price": 5, "amount": 0},
    "HEMI/USD": {"price": 5, "amount": 1},
    "ICP/USD": {"price": 3, "amount": 2},
    "LINEA/USD": {"price": 5, "amount": 0},
    "LINK/USD": {"price": 2, "amount": 2},
    "LISTA/USD": {"price": 4, "amount": 1},
    "LTC/USD": {"price": 2, "amount": 3},
    "MIRA/USD": {"price": 4, "amount": 1},
    "NEAR/USD": {"price": 3, "amount": 1},
    "OMNI/USD": {"price": 2, "amount": 2},
    "ONDO/USD": {"price": 4, "amount": 1},
    "OPEN/USD": {"price": 4, "amount": 1},
    "PAXG/USD": {"price": 2, "amount": 4},
    "PENDLE/USD": {"price": 3, "amount": 1},
    "PENGU/USD": {"price": 6, "amount": 0},
    "PEPE/USD": {"price": 8, "amount": 0},
    "PLUME/USD": {"price": 5, "amount": 0},
    "POL/USD": {"price": 4, "amount": 1},
    "PUMP/USD": {"price": 6, "amount": 0},
    "S/USD": {"price": 5, "amount": 1},
    "SEI/USD": {"price": 4, "amount": 1},
    "SHIB/USD": {"price": 8, "amount": 0},
    "SOL/USD": {"price": 2, "amount": 3},
    "SOMI/USD": {"price": 4, "amount": 1},
    "STO/USD": {"price": 4, "amount": 1},
    "SUI/USD": {"price": 4, "amount": 1},
    "TAO/USD": {"price": 1, "amount": 4},
    "TON/USD": {"price": 3, "amount": 2},
    "TRUMP/USD": {"price": 3, "amount": 3},
    "TRX/USD": {"price": 4, "amount": 1},
    "TUT/USD": {"price": 5, "amount": 0},
    "UNI/USD": {"price": 3, "amount": 2},
    "VIRTUAL/USD": {"price": 4, "amount": 1},
    "WIF/USD": {"price": 3, "amount": 2},
    "WLD/USD": {"price": 4, "amount": 1},
    "WLFI/USD": {"price": 4, "amount": 1},
    "XLM/USD": {"price": 4, "amount": 0},
    "XPL/USD": {"price": 4, "amount": 1},
    "XRP/USD": {"price": 4, "amount": 1},
    "ZEC/USD": {"price": 2, "amount": 3},
    "ZEN/USD": {"price": 3, "amount": 2},
}

# ============================================================
# TRADING SETTINGS
# ============================================================

INITIAL_CAPITAL = 50_000
MAKER_FEE = 0.0005   # 0.05% limit orders
TAKER_FEE = 0.001    # 0.1% market orders
MIN_ORDER_VALUE = 1  # $1 minimum order

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_binance_symbol(roostoo_pair):
    """Convert Roostoo pair to Binance symbol for data fetching."""
    # "BTC/USD" -> "BTCUSDT"
    coin = roostoo_pair.replace("/USD", "")
    return f"{coin}USDT"

def get_roostoo_pair(coin):
    """Convert coin to Roostoo pair format."""
    # "BTC" -> "BTC/USD"
    return f"{coin}/USD"

def round_price(pair, price):
    """Round price to correct precision for Roostoo."""
    precision = PAIR_PRECISION.get(pair, {}).get("price", 4)
    return round(price, precision)

def round_amount(pair, amount):
    """Round amount to correct precision for Roostoo."""
    precision = PAIR_PRECISION.get(pair, {}).get("amount", 2)
    return round(amount, precision)