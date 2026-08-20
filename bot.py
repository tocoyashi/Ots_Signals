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

# ================= Secure Configuration =================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN') or os.environ.get('BOT_TOKEN')
CHANNEL_ID = os.environ.get('CHANNEL_ID')

# ================= Trading Settings =================
TIMEFRAME = '15m'
TOP_N_COINS = 20
STABLECOINS = ['USDC/USDT', 'TUSD/USDT', 'DAI/USDT', 'FDUSD/USDT', 'USDP/USDT', 'PYUSD/USDT']
BLACKLIST = ['WXT/USDT', 'ANTFUN/USDT', 'UPC/USDT', 'RAIN/USDT', 'USD1/USDT', 'USDE/USDT']

LEVERAGE = 10
TP1_PERC = 0.8
TP2_PERC = 1.6
TP3_PERC = 2.8
TP4_PERC = 4.5
TP5_PERC = 7.0

# 🔒 SL ثابت 6% (كما في النسخة الأصلية)
SL_PERC = 6.0

# ================= Advanced Filters (لزيادة الدقة) =================
TREND_FILTER = True
TREND_EMA_PERIOD = 50

SQUEEZE_DURATION_FILTER = True
MIN_SQUEEZE_BARS = 3

FILTER_MOMENTUM_STRENGTH = True
FILTER_ATR_MINIMUM = True
ATR_MIN_PERCENT = 0.3

RSI_FILTER = True
RSI_PERIOD = 14
RSI_LONG_MAX = 65
RSI_SHORT_MIN = 35

VOLUME_FILTER = True
VOL_MIN_RATIO = 1.2

ADX_FILTER = True
ADX_PERIOD = 14
ADX_MIN = 22

# ================= Cooldown =================
COOLDOWN_FILE = Path('cooldown.json')
COOLDOWN_HOURS = 6


def _fmt(price):
    if price >= 1000:   return f"{price:,.2f}"
    elif price >= 1:    return f"{price:,.4f}"
    else:              return f"{price:,.6f}"


def calculate_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    return adx, plus_di, minus_di


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_signal_message(symbol, signal_type, entry_price):
    pair = symbol.replace('/', '')
    if signal_type == 1:
        direction = "LONG"
        tp1 = entry_price * (1 + TP1_PERC / 100)
        tp2 = entry_price * (1 + TP2_PERC / 100)
        tp3 = entry_price * (1 + TP3_PERC / 100)
        tp4 = entry_price * (1 + TP4_PERC / 100)
        tp5 = entry_price * (1 + TP5_PERC / 100)
        sl  = entry_price * (1 - SL_PERC / 100)
    else:
        direction = "SHORT"
        tp1 = entry_price * (1 - TP1_PERC / 100)
        tp2 = entry_price * (1 - TP2_PERC / 100)
        tp3 = entry_price * (1 - TP3_PERC / 100)
        tp4 = entry_price * (1 - TP4_PERC / 100)
        tp5 = entry_price * (1 - TP5_PERC / 100)
        sl  = entry_price * (1 + SL_PERC / 100)
    
    return f"""🎯 Precision Signal Alert

#{pair}  │ {TIMEFRAME} │
Direction: {direction}
Leverage: {LEVERAGE}x
Entry: {_fmt(entry_price)}

🎯 Targets:
TP1 ➜ {_fmt(tp1)} ({TP1_PERC}%)
TP2 ➜ {_fmt(tp2)} ({TP2_PERC}%)
TP3 ➜ {_fmt(tp3)} ({TP3_PERC}%)
TP4 ➜ {_fmt(tp4)} ({TP4_PERC}%)
TP5 ➜ {_fmt(tp5)} ({TP5_PERC}%)

🛡 Stop Loss: {_fmt(sl)} ({SL_PERC}%)
⚖️ After TP1 → Move to Breakeven

━━━━━━━━━━━━━━━
✅ Filters Passed:
• Squeeze Release + Duration ≥{MIN_SQUEEZE_BARS}
• Momentum Strength
• Trend Confirmation (EMA{TREND_EMA_PERIOD})
• Volume Spike >{VOL_MIN_RATIO}x
• RSI Zone Check
• ADX > {ADX_MIN}

L E A K E D  B Y: @BULLS_SIGNALS"""


def _fast_linreg_endpoint(series, window=20):
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
        
        bb_basis = data['close'].rolling(window=self.bb_length).mean()
        bb_dev = self.bb_mult * data['close'].rolling(window=self.bb_length).std()
        upper_bb = bb_basis + bb_dev
        lower_bb = bb_basis - bb_dev
        
        kc_ma = data['close'].rolling(window=self.kc_length).mean()
        tr = self.true_range(data['high'], data['low'], data['close'])
        range_ma = tr.rolling(window=self.kc_length).mean()
        upper_kc = kc_ma + range_ma * self.kc_mult
        lower_kc = kc_ma - range_ma * self.kc_mult
        
        squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        
        highest_high = data['high'].rolling(window=self.kc_length).max()
        lowest_low = data['low'].rolling(window=self.kc_length).min()
        close_ma = data['close'].rolling(window=self.kc_length).mean()
        avg_val = ((highest_high + lowest_low) / 2 + close_ma) / 2
        momentum = _fast_linreg_endpoint(data['close'] - avg_val, self.kc_length)
        
        # ── Advanced Filters ──
        squeeze_groups = (squeeze_on != squeeze_on.shift()).cumsum()
        squeeze_duration = squeeze_on.groupby(squeeze_groups).cumsum()
        
        mom_rolling_std = momentum.rolling(window=100).std()
        mom_threshold = (mom_rolling_std * 0.5).fillna(0)
        momentum_strong = momentum.abs() > mom_threshold
        
        atr = tr.rolling(window=14).mean()
        atr_pct = (atr / data['close']) * 100
        
        rsi = calculate_rsi(data['close'], RSI_PERIOD)
        vol_sma = data['volume'].rolling(window=20).mean()
        ema_trend = data['close'].ewm(span=TREND_EMA_PERIOD, adjust=False).mean()
        adx, plus_di, minus_di = calculate_adx(data, ADX_PERIOD)
        
        data['squeeze_on'] = squeeze_on
        data['squeeze_duration'] = squeeze_duration
        data['momentum'] = momentum
        data['momentum_increasing'] = momentum > momentum.shift(1)
        data['momentum_strong'] = momentum_strong
        data['atr'] = atr
        data['atr_pct'] = atr_pct
        data['rsi'] = rsi
        data['vol_sma'] = vol_sma
        data['ema_trend'] = ema_trend
        data['adx'] = adx
        return data

    def generate_signals(self, df):
        data = self.calculate_indicators(df)
        data['signal'] = 0
        
        squeeze_on_safe = data['squeeze_on'].fillna(False).astype(bool)
        mom_inc_safe = data['momentum_increasing'].fillna(False).astype(bool)
        
        # Squeeze Release + Duration
        squeeze_release = (squeeze_on_safe.shift(1) == True) & (squeeze_on_safe == False)
        if SQUEEZE_DURATION_FILTER:
            valid_duration = data['squeeze_duration'].shift(1) >= MIN_SQUEEZE_BARS
            squeeze_release = squeeze_release & valid_duration
        
        # BUY: Release + Momentum صاعد + قوي
        buy_cond = (
            (squeeze_release == True) &
            (data['momentum'] > 0) &
            (mom_inc_safe == True) &
            (data['momentum_strong'].fillna(False) == True)
        )
        
        # SELL: Release + Momentum هابط + قوي
        sell_cond = (
            (squeeze_release == True) &
            (data['momentum'] < 0) &
            (mom_inc_safe == False) &
            (data['momentum_strong'].fillna(False) == True)
        )
        
        # Apply Filters
        if TREND_FILTER:
            buy_cond = buy_cond & (data['close'] > data['ema_trend'])
            sell_cond = sell_cond & (data['close'] < data['ema_trend'])
        
        if FILTER_ATR_MINIMUM:
            atr_ok = data['atr_pct'].fillna(0) > ATR_MIN_PERCENT
            buy_cond = buy_cond & atr_ok
            sell_cond = sell_cond & atr_ok
        
        if RSI_FILTER:
            buy_cond = buy_cond & (data['rsi'].fillna(50) < RSI_LONG_MAX)
            sell_cond = sell_cond & (data['rsi'].fillna(50) > RSI_SHORT_MIN)
        
        if VOLUME_FILTER:
            vol_ok = data['volume'] > data['vol_sma'] * VOL_MIN_RATIO
            buy_cond = buy_cond & vol_ok
            sell_cond = sell_cond & vol_ok
        
        if ADX_FILTER:
            adx_ok = data['adx'].fillna(0) > ADX_MIN
            buy_cond = buy_cond & adx_ok
            sell_cond = sell_cond & adx_ok
        
        data.loc[buy_cond, 'signal'] = 1
        data.loc[sell_cond, 'signal'] = -1
        return data


def get_mexc_data(symbol, timeframe, limit=150):
    exchange = ccxt.mexc({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def get_top_mexc_coins(limit=20):
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
        top = [p['symbol'] for p in usdt_pairs[:limit]]
        print(f"Top coins: {top[:5]} ... ({len(top)} total)")
        return top
    except Exception as e:
        print(f"Error fetching coins: {e}")
        return []


def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHANNEL_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': CHANNEL_ID, 'text': message, 'parse_mode': 'HTML'}
    try:
        requests.post(url, json=payload, timeout=30)
    except Exception as e:
        print(f"Telegram error: {e}")


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
        print(f"Cooldown save error: {e}")

def is_on_cooldown(symbol, cooldown_data):
    if symbol not in cooldown_data:
        return False
    try:
        last_time = datetime.fromisoformat(cooldown_data[symbol])
        return (datetime.now() - last_time).total_seconds() / 3600 < COOLDOWN_HOURS
    except Exception:
        return False


def main():
    print("=" * 60)
    print(" PRECISION SQUEEZE BOT — Fixed SL 6% Edition")
    print(f" Timeframe: {TIMEFRAME} | Leverage: {LEVERAGE}x | SL: {SL_PERC}%")
    print("=" * 60)
    
    if not TELEGRAM_TOKEN or not CHANNEL_ID:
        print("Environment not configured.")
        return

    top_coins = get_top_mexc_coins(TOP_N_COINS)
    if not top_coins:
        return

    cooldown_data = load_cooldown()
    indicator = SqueezeMomentumIndicator()
    stats = {'sent': 0, 'cooldown': 0, 'no_signal': 0}

    for symbol in top_coins:
        try:
            if is_on_cooldown(symbol, cooldown_data):
                stats['cooldown'] += 1
                continue

            time.sleep(0.5)
            df = get_mexc_data(symbol, TIMEFRAME, limit=150)
            df_signals = indicator.generate_signals(df)
            latest = df_signals.iloc[-2]
            current_signal = latest['signal']
            
            if current_signal == 0:
                stats['no_signal'] += 1
                continue
            
            entry_price = latest['close']
            cooldown_data[symbol] = datetime.now().isoformat()
            stats['sent'] += 1

            msg = build_signal_message(symbol, current_signal, entry_price)
            send_telegram_message(msg)
            
            dir_str = "LONG" if current_signal == 1 else "SHORT"
            print(f"  🎯 {symbol} {dir_str} | SL: {SL_PERC}%")

        except Exception as e:
            print(f"  ⚠️ {symbol}: {e}")

    save_cooldown(cooldown_data)
    print(f"\n Done. Sent: {stats['sent']} | Cooldown: {stats['cooldown']} | No signal: {stats['no_signal']}")


if __name__ == "__main__":
    main()
