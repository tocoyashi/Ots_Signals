import os
import logging
import requests
import pandas as pd
import numpy as np
import ccxt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OTS-Bot")

# ═══════════════════════════════════════════
#  CONFIG FROM GITHUB SECRETS
# ═══════════════════════════════════════════
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHANNEL_ID = os.environ.get("CHANNEL_ID")
EXCHANGE_NAME = os.environ.get("EXCHANGE_NAME", "bybit")
SYMBOL = os.environ.get("SYMBOL", "BTC/USDT")
TIMEFRAME = "4h"
LOOKBACK = 1000

# RSI SYSTEM
RSI_LENGTH = 14
RSI_UPPER = 65.0
RSI_LOWER = 35.0
RSI_COOLDOWN = 20
RSI_MAX_DISTANCE = 2.2
RSI_MAX_CANDLE = 2.0

# KDJ SYSTEM
KDJ_PERIOD = 9
KDJ_LOOKBACK = 100
KDJ_HMA_LEN = 50
KDJ_MIN_CANDLE = 0.5
KDJ_MAX_CANDLE = 2.0
KDJ_MAX_DIST_EMA = 2.0
KDJ_COOLDOWN = 20

# S/R SYSTEM
SR_PIVOT_LEN = 10
SR_MIN_DISTANCE = 0.5
SR_MAX_DIST_EMA = 2.0
SR_MAX_CANDLE = 2.0
SR_COOLDOWN = 15

# OTS SYSTEM
OTS_ALPHA = 20
OTS_DELTA = 17

# VOLUME SYSTEM
VOL_LENGTH = 14
VOL_MULTIPLIER = 3.4
VOL_COOLDOWN = 30
VOL_DISTANCE_PERCENT = 3.0

# PRESSURE SYSTEM
PRESSURE_RSI_PERIOD = 50
PRESSURE_RATE = 45
PRESSURE_THRESHOLD = 500
PRESSURE_EMA_DIST = 5.0

# TP/SL
SL_PCT = 3.5
TP1_PCT = 1.0
TP2_PCT = 3.0
TP3_PCT = 6.0


# ═══════════════════════════════════════════
#  INDICATOR FUNCTIONS
# ═══════════════════════════════════════════

def rma(s, length):
    return s.ewm(alpha=1 / length, adjust=False).mean()

def ema(s, length):
    return s.ewm(span=length, adjust=False).mean()

def hma(s, length):
    half = length // 2
    sqrt_len = int(np.sqrt(length))
    return ema(2 * ema(s, half) - ema(s, length), sqrt_len)

def rsi_calc(close, length=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    up_r = rma(up, length)
    down_r = rma(down, length)
    val = np.where(down_r == 0, 100.0, np.where(up_r == 0, 0.0, 100.0 - 100.0 / (1.0 + up_r / down_r)))
    return pd.Series(val, index=close.index)

def bcwsma(series, length, m=1):
    result = pd.Series(np.nan, index=series.index)
    for i in range(len(series)):
        if i == 0:
            result.iloc[i] = series.iloc[i]
        else:
            prev = result.iloc[i - 1]
            if pd.isna(prev):
                result.iloc[i] = series.iloc[i]
            else:
                result.iloc[i] = (m * series.iloc[i] + (length - m) * prev) / length
    return result

def apply_cooldown(signals, cooldown):
    result = signals.copy()
    last_idx = -999999
    for i in range(len(result)):
        if result.iloc[i] and (i - last_idx) >= cooldown:
            last_idx = i
        else:
            result.iloc[i] = False
    return result

def pivothigh(high, left, right):
    result = pd.Series(np.nan, index=high.index)
    for i in range(left, len(high) - right):
        window = high.iloc[i - left: i + right + 1]
        if high.iloc[i] == window.max():
            result.iloc[i + right] = high.iloc[i]
    return result

def pivotlow(low, left, right):
    result = pd.Series(np.nan, index=low.index)
    for i in range(left, len(low) - right):
        window = low.iloc[i - left: i + right + 1]
        if low.iloc[i] == window.min():
            result.iloc[i + right] = low.iloc[i]
    return result


# ═══════════════════════════════════════════
#  SYSTEM 1: RSI + EMA 150 + EMA 500
# ═══════════════════════════════════════════

def rsi_system(df):
    rsi = df["rsi"]
    close = df["close"]
    e150 = df["ema150"]
    e500 = df["ema500"]

    is_up = rsi > RSI_UPPER
    is_down = rsi < RSI_LOWER
    buy_cross = is_up & ~is_up.shift(1).fillna(False)
    sell_cross = is_down & ~is_down.shift(1).fillna(False)

    e150_above = e150 > e500
    e150_below = e150 < e500
    dist = ((close - e150) / e150).abs() * 100
    near = dist <= RSI_MAX_DISTANCE
    candle = ((close - df["open"]) / df["open"]).abs() * 100
    valid_c = candle <= RSI_MAX_CANDLE

    buy = buy_cross & (close > e150) & near & valid_c & e150_above
    sell = sell_cross & (close < e150) & near & valid_c & e150_below
    return apply_cooldown(buy, RSI_COOLDOWN), apply_cooldown(sell, RSI_COOLDOWN), "RSI"


# ═══════════════════════════════════════════
#  SYSTEM 2: KDJ + HMA 50
# ═══════════════════════════════════════════

def kdj_system(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    e200 = df["ema200"]

    df["hma50"] = hma(close, KDJ_HMA_LEN)
    hma50 = df["hma50"]

    kdj_h = high.rolling(KDJ_LOOKBACK).max()
    kdj_l = low.rolling(KDJ_LOOKBACK).min()
    rsv = 100.0 * ((close - kdj_l) / (kdj_h - kdj_l)).fillna(50)
    pK = bcwsma(rsv, KDJ_PERIOD)
    pD = bcwsma(pK, KDJ_PERIOD)

    cross_up = (pK > pD) & (pK.shift(1) <= pD.shift(1).fillna(0))
    cross_down = (pK < pD) & (pK.shift(1) >= pD.shift(1).fillna(0))

    hma_up = (close > hma50) & (close.shift(1) <= hma50.shift(1).fillna(close))
    hma_down = (close < hma50) & (close.shift(1) >= hma50.shift(1).fillna(close))

    candle = ((close - df["open"]) / close).abs() * 100
    valid_c = (candle >= KDJ_MIN_CANDLE) & (candle <= KDJ_MAX_CANDLE)
    dist = ((close - e200) / close).abs() * 100
    valid_d = dist <= KDJ_MAX_DIST_EMA

    buy = cross_up & (close > e200) & hma_up & valid_c & valid_d
    sell = cross_down & (close < e200) & hma_down & valid_c & valid_d
    return apply_cooldown(buy, KDJ_COOLDOWN), apply_cooldown(sell, KDJ_COOLDOWN), "KDJ"


# ═══════════════════════════════════════════
#  SYSTEM 3: S/R Filter + EMA 200
# ═══════════════════════════════════════════

def sr_system(df):
    rsi = df["rsi"]
    close = df["close"]
    e200 = df["ema200"]

    is_up = rsi > 70.0
    is_down = rsi < 30.0
    buy_base = is_up & ~is_up.shift(1).fillna(False)
    sell_base = is_down & ~is_down.shift(1).fillna(False)

    dist = ((close - e200) / e200).abs() * 100
    near = dist <= SR_MAX_DIST_EMA
    candle = ((close - df["open"]) / df["open"]).abs() * 100
    valid_c = candle <= SR_MAX_CANDLE

    buy = buy_base & (close > e200) & near & valid_c
    sell = sell_base & (close < e200) & near & valid_c
    return apply_cooldown(buy, SR_COOLDOWN), apply_cooldown(sell, SR_COOLDOWN), "S/R"


# ═══════════════════════════════════════════
#  SYSTEM 4: OTS Exhaustion (Lelec)
# ═══════════════════════════════════════════

def ots_system(df):
    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    e250 = df["ema250"]
    n = len(close)

    result = np.zeros(n, dtype=int)
    bindex = sindex = 0

    for i in range(4, n):
        if close[i] > close[i - 4]: bindex += 1
        if close[i] < close[i - 4]: sindex += 1

        if sindex > OTS_DELTA and close[i] > open_[i]:
            if low[i] <= np.min(low[i - OTS_ALPHA + 1: i + 1]):
                result[i] = 1; sindex = 0
        if bindex > OTS_DELTA and close[i] < open_[i]:
            if high[i] >= np.max(high[i - OTS_ALPHA + 1: i + 1]):
                result[i] = -1; bindex = 0

    lelec = pd.Series(result, index=df.index)
    buy = (lelec == 1) & (df["close"] > e250)
    sell = (lelec == -1) & (df["close"] < e250)
    return buy, sell, "OTS"


# ═══════════════════════════════════════════
#  SYSTEM 5: Volume Spike
# ═══════════════════════════════════════════

def vol_system(df):
    close = df["close"]
    open_ = df["open"]
    vol = df["volume"]
    e250 = df["ema250"]

    bull = np.where(close > open_, vol, 0)
    bear = np.where(open_ > close, vol, 0)
    bull_s = pd.Series(bull, index=df.index)
    bear_s = pd.Series(bear, index=df.index)

    bullma = ema(bull_s, VOL_LENGTH)
    bearma = ema(bear_s, VOL_LENGTH)

    buy = (bull_s > bullma * VOL_MULTIPLIER) & (bull_s.shift(1) <= (bullma.shift(1) * VOL_MULTIPLIER).fillna(0))
    sell = (bear_s > bearma * VOL_MULTIPLIER) & (bear_s.shift(1) <= (bearma.shift(1) * VOL_MULTIPLIER).fillna(0))

    dp = ((close - e250) / e250) * 100
    buy = buy & (dp >= 0) & (dp <= VOL_DISTANCE_PERCENT)
    sell = sell & (dp <= 0) & (dp >= -VOL_DISTANCE_PERCENT)
    return apply_cooldown(buy, VOL_COOLDOWN), apply_cooldown(sell, VOL_COOLDOWN), "VOL"


# ═══════════════════════════════════════════
#  SYSTEM 6: Pressure Tracker
# ═══════════════════════════════════════════

def pressure_system(df):
    close = df["close"]
    r1 = rsi_calc(close, PRESSURE_RSI_PERIOD)
    r2 = rsi_calc(r1, PRESSURE_RSI_PERIOD)
    r3 = rsi_calc(r2, PRESSURE_RSI_PERIOD)
    rsim = r1 * 5 - r2 * 3 + r3

    diff = (rsim - rsim.shift(1).fillna(0)) * PRESSURE_RATE
    rsim_up = pd.Series(0.0, index=df.index)
    rsim_down = pd.Series(0.0, index=df.index)

    for i in range(1, len(df)):
        d = diff.iloc[i]
        if d > 0:
            if rsim_up.iloc[i - 1] != 0 or d > PRESSURE_THRESHOLD:
                rsim_up.iloc[i] = rsim_up.iloc[i - 1] + d
                rsim_down.iloc[i] = 0
            else:
                rsim_up.iloc[i] = 0
                rsim_down.iloc[i] = rsim_down.iloc[i - 1]
        else:
            if rsim_down.iloc[i - 1] != 0 or d < -PRESSURE_THRESHOLD:
                rsim_up.iloc[i] = 0
                rsim_down.iloc[i] = rsim_down.iloc[i - 1] + d
            else:
                rsim_up.iloc[i] = rsim_up.iloc[i - 1]
                rsim_down.iloc[i] = 0

    dp = ((close - df["ema250"]) / df["ema250"]) * 100
    is_up_rsi = df["rsi"] > RSI_UPPER
    is_down_rsi = df["rsi"] < RSI_LOWER

    buy = (rsim_up > 0) & (rsim_up.shift(1).fillna(0) == 0) & (diff > PRESSURE_THRESHOLD) & (dp < -PRESSURE_EMA_DIST) & ~is_down_rsi
    sell = (rsim_down < 0) & (rsim_down.shift(1).fillna(0) == 0) & (diff < -PRESSURE_THRESHOLD) & (dp > PRESSURE_EMA_DIST) & ~is_up_rsi
    return buy, sell, "Pressure"


# ═══════════════════════════════════════════
#  TELEGRAM
# ═══════════════════════════════════════════

def send_telegram(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHANNEL_ID, "text": text, "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.json().get("ok"):
            return True
        logger.error(f"TG error: {r.json()}")
    except Exception as e:
        logger.error(f"TG send error: {e}")
    return False


def send_signal(sig_type, source, price, rsi_val, bar_time):
    if sig_type == "BUY":
        emoji = "🟢"
        direction = "شراء / BUY"
        sl = round(price * (1 - SL_PCT / 100), 4)
        tp1 = round(price * (1 + TP1_PCT / 100), 4)
        tp2 = round(price * (1 + TP2_PCT / 100), 4)
        tp3 = round(price * (1 + TP3_PCT / 100), 4)
        sl_txt = f"📉 وقف الخسارة: <b>{sl}</b>"
    else:
        emoji = "🔴"
        direction = "بيع / SELL"
        sl = round(price * (1 + SL_PCT / 100), 4)
        tp1 = round(price * (1 - TP1_PCT / 100), 4)
        tp2 = round(price * (1 - TP2_PCT / 100), 4)
        tp3 = round(price * (1 - TP3_PCT / 100), 4)
        sl_txt = f"📈 وقف الخسارة: <b>{sl}</b>"

    msg = (
        f"{emoji} <b>OTS 4H Trading System V1.2</b> {emoji}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>الزوج:</b> {SYMBOL}\n"
        f"⏰ <b>الإطار:</b> 4H\n"
        f"🔔 <b>الإشارة:</b> <b>{direction}</b>\n"
        f"🧩 <b>النظام:</b> {source}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 <b>سعر الدخول:</b> <b>{price}</b>\n"
        f"{sl_txt}\n"
        f"🎯 هدف 1 (50%): <b>{tp1}</b>\n"
        f"🎯 هدف 2 (40%): <b>{tp2}</b>\n"
        f"🎯 هدف 3 (10%): <b>{tp3}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📏 <b>RSI:</b> {rsi_val:.1f}\n"
        f"🕐 <b>الوقت:</b> {bar_time}\n"
    )
    return send_telegram(msg)


# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════

def analyze():
    logger.info(f"Starting OTS 4H Scan for {SYMBOL}...")

    exchange = ccxt.__dict__.get(EXCHANGE_NAME, ccxt.bybit)()
    exchange.enableRateLimit = True

    raw = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LOOKBACK)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    logger.info(f"Fetched {len(df)} candles")

    # Compute base indicators
    df["rsi"] = rsi_calc(df["close"], RSI_LENGTH)
    df["ema150"] = ema(df["close"], 150)
    df["ema200"] = ema(df["close"], 200)
    df["ema250"] = ema(df["close"], 250)
    df["ema500"] = ema(df["close"], 500)

    # Run all systems
    systems = [
        rsi_system(df),
        kdj_system(df),
        sr_system(df),
        ots_system(df),
        vol_system(df),
        pressure_system(df),
    ]

    last = df.iloc[-1]
    last_time = str(df.index[-1])
    price = last["close"]
    rsi_val = last["rsi"]

    found = False
    for buy_col, sell_col, name in systems:
        if buy_col.iloc[-1]:
            logger.info(f"BUY signal from {name}")
            send_signal("BUY", name, round(price, 4), rsi_val, last_time)
            found = True
        if sell_col.iloc[-1]:
            logger.info(f"SELL signal from {name}")
            send_signal("SELL", name, round(price, 4), rsi_val, last_time)
            found = True

    if not found:
        logger.info("No signals found this round.")

    logger.info("Scan complete.")


if __name__ == "__main__":
    analyze()