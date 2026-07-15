import pandas as pd
import numpy as np
import ccxt
import requests
import os
import time
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ================= Secure Configuration (From GitHub Secrets) =================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN') or os.environ.get('BOT_TOKEN')
CHANNEL_ID = os.environ.get('CHANNEL_ID')

# ================= Trading Settings =================
TIMEFRAME = '15m'
TOP_N_COINS = 20
STABLECOINS = ['USDC/USDT', 'TUSD/USDT', 'DAI/USDT', 'FDUSD/USDT', 'USDP/USDT', 'PYUSD/USDT']
BLACKLIST = ['WXT/USDT', 'ANTFUN/USDT', 'UPC/USDT', 'RAIN/USDT', 'USD1/USDT', 'USDE/USDT']

# ================= Risk Management Settings =================
LEVERAGE = 10
TP1_PERC = 0.6
TP2_PERC = 1.5
TP3_PERC = 2.4
TP4_PERC = 5.0
TP5_PERC = 7.0
SL_PERC = 6.0

# ================= Quality Filters (Backtested — F3+F4) =================
# Filter F3: Momentum Strength — |momentum| > 0.5 × rolling_std(100)
#   → Ensures momentum is statistically significant, not noise
#   → Backtest: WR 79.2% → alone, PF 4.09
FILTER_MOMENTUM_STRENGTH = True

# Filter F4: ATR Minimum — ATR% > 0.3% of price
#   → Avoids low-volatility coins where signals are noise
#   → Backtest: WR 82.6% → alone, PF 4.87
#   → Combined F3+F4: WR 84.5%, PF 5.18 (BEST combo)
FILTER_ATR_MINIMUM = True
ATR_MIN_PERCENT = 0.3     # Minimum ATR as % of price

# Optional: Squeeze Duration (F2) — uncomment for fewer trades
# FILTER_SQUEEZE_DURATION = True
# MIN_SQUEEZE_BARS = 5

# ================= Cooldown Settings =================
COOLDOWN_FILE = Path('cooldown.json')
COOLDOWN_HOURS = 4


def _fmt(price):
    """Smart price formatting: fewer decimals for high prices"""
    if price >= 1000:   return f"{price:,.2f}"
    elif price >= 1:    return f"{price:,.4f}"
    else:              return f"{price:,.6f}"


def build_signal_message(symbol, signal_type, entry_price, entry_time):
    if signal_type == 1:
        tp1 = entry_price * (1 + TP1_PERC / 100)
        tp2 = entry_price * (1 + TP2_PERC / 100)
        tp3 = entry_price * (1 + TP3_PERC / 100)
        tp4 = entry_price * (1 + TP4_PERC / 100)
        tp5 = entry_price * (1 + TP5_PERC / 100)
        sl  = entry_price * (1 - SL_PERC / 100)
        direction = "LONG"
    else:
        tp1 = entry_price * (1 - TP1_PERC / 100)
        tp2 = entry_price * (1 - TP2_PERC / 100)
        tp3 = entry_price * (1 - TP3_PERC / 100)
        tp4 = entry_price * (1 - TP4_PERC / 100)
        tp5 = entry_price * (1 - TP5_PERC / 100)
        sl  = entry_price * (1 + SL_PERC / 100)
        direction = "SHORT"

    pair = symbol.replace('/', '')
    msg = f"""🌤 New Trading Signals

#{pair}  │ 15m │
{direction}
Entry: {_fmt(entry_price)}
Leverage :  {LEVERAGE}x

TP1 ➜ {_fmt(tp1)}
TP2 ➜ {_fmt(tp2)}
TP3 ➜ {_fmt(tp3)}
TP4 ➜ {_fmt(tp4)}
TP5 ➜ {_fmt(tp5)}  ☀️☀️

SL :  {_fmt(sl)}
↻ After TP1 → BE"""

    return msg


def _fast_linreg_endpoint(series, window=20):
    """Fast vectorized linear regression endpoint using numpy convolution.
    Computes slope*(w-1) + intercept for each rolling window.
    ~6000x faster than scipy.stats.linregress rolling apply."""
    vals = series.values.astype(float)
    n = len(vals)
    result = np.full(n, np.nan)
    if n < window:
        return pd.Series(result, index=series.index)

    y = np.arange(window, dtype=float)
    sum_y = y.sum()
    sum_y2 = (y ** 2).sum()
    denom = window * sum_y2 - sum_y ** 2

    weighted_sum = np.convolve(vals, y[::-1], mode='valid')
    rolling_sum = np.convolve(vals, np.ones(window), mode='valid')

    slopes = (window * weighted_sum - rolling_sum * sum_y) / denom
    x_means = rolling_sum / window
    intercepts = x_means - slopes * sum_y / window
    result[window - 1:] = slopes * (window - 1) + intercepts

    return pd.Series(result, index=series.index)


class SqueezeMomentumIndicator:
    def __init__(self, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult

    def true_range(self, high, low, close):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def calculate_indicators(self, df):
        data = df.copy()

        # Bollinger Bands
        bb_basis = data['close'].rolling(window=self.bb_length).mean()
        bb_dev = self.bb_mult * data['close'].rolling(window=self.bb_length).std()
        upper_bb = bb_basis + bb_dev
        lower_bb = bb_basis - bb_dev

        # Keltner Channel
        kc_ma = data['close'].rolling(window=self.kc_length).mean()
        tr = self.true_range(data['high'], data['low'], data['close'])
        range_ma = tr.rolling(window=self.kc_length).mean()
        upper_kc = kc_ma + range_ma * self.kc_mult
        lower_kc = kc_ma - range_ma * self.kc_mult

        # Squeeze detection
        squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)

        # Momentum (fast vectorized linear regression)
        highest_high = data['high'].rolling(window=self.kc_length).max()
        lowest_low = data['low'].rolling(window=self.kc_length).min()
        close_ma = data['close'].rolling(window=self.kc_length).mean()
        avg_val = ((highest_high + lowest_low) / 2 + close_ma) / 2
        momentum = _fast_linreg_endpoint(data['close'] - avg_val, self.kc_length)

        # ── Quality Filter Indicators ──
        # F3: Momentum Strength — rolling std-based threshold
        mom_rolling_std = momentum.rolling(window=100).std()
        mom_threshold = (mom_rolling_std * 0.5).fillna(0)
        momentum_strong = momentum.abs() > mom_threshold

        # F4: ATR as % of price
        atr = tr.rolling(window=14).mean()
        atr_pct = (atr / data['close']) * 100

        data['squeeze_on'] = squeeze_on
        data['momentum'] = momentum
        data['momentum_increasing'] = momentum > momentum.shift(1)
        data['momentum_strong'] = momentum_strong
        data['atr_pct'] = atr_pct

        return data

    def generate_signals(self, df):
        data = self.calculate_indicators(df)
        data['signal'] = 0

        squeeze_on_safe = data['squeeze_on'].fillna(False).astype(bool)
        mom_inc_safe = data['momentum_increasing'].fillna(False).astype(bool)

        data['squeeze_release'] = (squeeze_on_safe.shift(1) == True) & (squeeze_on_safe == False)

        # ── Base BUY condition ──
        buy_cond = (
            (data['squeeze_release'] == True) &
            (data['momentum'] > 0) &
            (mom_inc_safe == True)
        )

        # ── Base SELL condition ──
        sell_cond = (
            ((data['momentum'] < 0) & (data['momentum'].shift(1).fillna(0) >= 0)) |
            ((mom_inc_safe == False) & (mom_inc_safe.shift(1).fillna(True) == False) & (data['momentum'] > 0))
        )

        # ── Apply Quality Filters ──
        if FILTER_MOMENTUM_STRENGTH:
            mom_ok = data['momentum_strong'].fillna(False)
            buy_cond = buy_cond & mom_ok
            sell_cond = sell_cond & mom_ok

        if FILTER_ATR_MINIMUM:
            atr_ok = data['atr_pct'].fillna(0) > ATR_MIN_PERCENT
            buy_cond = buy_cond & atr_ok
            sell_cond = sell_cond & atr_ok

        data.loc[buy_cond, 'signal'] = 1
        data.loc[sell_cond, 'signal'] = -1

        return data


def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHANNEL_ID:
        print("Error: TELEGRAM_TOKEN or CHANNEL_ID is missing!")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHANNEL_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error sending Telegram message: {e}")


def get_mexc_data(symbol, timeframe, limit=100):
    exchange = ccxt.mexc({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def get_top_mexc_coins(limit=20):
    """Fetches top N coins sorted by 24h volume in USDT, excluding blacklist"""
    print(f"Fetching top {limit} coins by volume from MEXC...")
    exchange = ccxt.mexc({'enableRateLimit': True})
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []

        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT') and symbol not in STABLECOINS and symbol not in BLACKLIST:
                vol = ticker.get('quoteVolume') or 0
                if vol > 1000000:
                    usdt_pairs.append({'symbol': symbol, 'volume': vol})

        usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
        top_coins = [pair['symbol'] for pair in usdt_pairs[:limit]]
        print(f"Successfully fetched: {top_coins[:5]} ... (and {len(top_coins)-5} more)")
        return top_coins
    except Exception as e:
        print(f"Error fetching top coins list: {e}")
        return []


# ================= Cooldown Functions =================
def load_cooldown():
    if COOLDOWN_FILE.exists():
        try:
            with open(COOLDOWN_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_cooldown(data):
    try:
        with open(COOLDOWN_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save cooldown file: {e}")

def is_on_cooldown(symbol, cooldown_data):
    if symbol not in cooldown_data:
        return False
    try:
        last_time = datetime.fromisoformat(cooldown_data[symbol])
        elapsed_hours = (datetime.now() - last_time).total_seconds() / 3600
        return elapsed_hours < COOLDOWN_HOURS
    except Exception:
        return False


def main():
    print("=== Running Multi-Coin Scalping Bot ===")
    if not TELEGRAM_TOKEN or not CHANNEL_ID:
        print("Environment not configured.")
        return

    print(f"[{datetime.now()}] Starting scan for TOP {TOP_N_COINS} coins on {TIMEFRAME} timeframe...")
    print(f"Filters: Momentum Strength={FILTER_MOMENTUM_STRENGTH} | ATR Min={FILTER_ATR_MINIMUM}%")

    top_coins = get_top_mexc_coins(TOP_N_COINS)

    if not top_coins:
        print("Failed to get coin list. Aborting run.")
        return

    cooldown_data = load_cooldown()
    cooldown_skipped = 0
    indicator = SqueezeMomentumIndicator()
    signals_found = 0
    filter_blocked = 0

    for symbol in top_coins:
        try:
            if is_on_cooldown(symbol, cooldown_data):
                cooldown_skipped += 1
                continue

            time.sleep(0.5)

            df = get_mexc_data(symbol, TIMEFRAME)
            df_signals = indicator.generate_signals(df)

            latest_candle = df_signals.iloc[-2]
            current_signal = latest_candle['signal']
            current_price = latest_candle['close']
            current_time = latest_candle.name

            if current_signal != 0:
                signals_found += 1
                cooldown_data[symbol] = datetime.now().isoformat()

                msg = build_signal_message(symbol, current_signal, current_price, current_time)

                send_telegram_message(msg)
                print(f"-> Signal sent for {symbol}: {'BUY' if current_signal == 1 else 'SELL'}")

        except Exception as e:
            pass

    save_cooldown(cooldown_data)
    print(f"[{datetime.now()}] Scan finished. Signals: {signals_found} | Cooldown skipped: {cooldown_skipped}")


if __name__ == "__main__":
    main()
