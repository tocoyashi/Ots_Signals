"""
OTS 4H Trading System V1.2 — Backtest Engine
Tests all 7 systems + Scalp on 4 pairs over 4 months.
Sends results to Telegram.
"""

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta, timezone

pd.set_option('future.no_silent_downcasting', True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OTS-Backtest")

# ═══════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHANNEL_ID = os.environ.get("CHANNEL_ID", "")
EXCHANGE_NAME = os.environ.get("EXCHANGE_NAME", "mexc")
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
TIMEFRAME = "4h"
INITIAL_BALANCE = 10000.0
RISK_PER_TRADE_PCT = 2.0  # 2% of balance per trade
BACKTEST_MONTHS = 4

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

# SCALP SYSTEM
SCALP_COOLDOWN = 10
SCALP_EMA_FAR = 1.5
SCALP_MAX_EMA200 = 3.0
SCALP_SL_PCT = 2.0
SCALP_TP_PCT = 0.85

# TP/SL (Normal signals)
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


# ═══════════════════════════════════════════
#  SYSTEMS (same as bot.py)
# ═══════════════════════════════════════════

def rsi_system(df):
    rsi = df["rsi"]
    close = df["close"]
    e150 = df["ema150"]
    e500 = df["ema500"]
    is_up = rsi > RSI_UPPER
    is_down = rsi < RSI_LOWER
    buy_cross = is_up & ~is_up.shift(1).fillna(False).infer_objects(copy=False)
    sell_cross = is_down & ~is_down.shift(1).fillna(False).infer_objects(copy=False)
    e150_above = e150 > e500
    e150_below = e150 < e500
    dist = ((close - e150) / e150).abs() * 100
    near = dist <= RSI_MAX_DISTANCE
    candle = ((close - df["open"]) / df["open"]).abs() * 100
    valid_c = candle <= RSI_MAX_CANDLE
    buy = buy_cross & (close > e150) & near & valid_c & e150_above
    sell = sell_cross & (close < e150) & near & valid_c & e150_below
    return apply_cooldown(buy, RSI_COOLDOWN), apply_cooldown(sell, RSI_COOLDOWN), "RSI"


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


def sr_system(df):
    rsi = df["rsi"]
    close = df["close"]
    e200 = df["ema200"]
    is_up = rsi > 70.0
    is_down = rsi < 30.0
    buy_base = is_up & ~is_up.shift(1).fillna(False).infer_objects(copy=False)
    sell_base = is_down & ~is_down.shift(1).fillna(False).infer_objects(copy=False)
    dist = ((close - e200) / e200).abs() * 100
    near = dist <= SR_MAX_DIST_EMA
    candle = ((close - df["open"]) / df["open"]).abs() * 100
    valid_c = candle <= SR_MAX_CANDLE
    buy = buy_base & (close > e200) & near & valid_c
    sell = sell_base & (close < e200) & near & valid_c
    return apply_cooldown(buy, SR_COOLDOWN), apply_cooldown(sell, SR_COOLDOWN), "S/R"


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


def scalp_system(df):
    rsi = df["rsi"]
    close = df["close"]
    e150 = df["ema150"]
    e200 = df["ema200"]
    is_up = rsi > RSI_UPPER
    is_down = rsi < RSI_LOWER
    buy_cross = is_up & ~is_up.shift(1).fillna(False).infer_objects(copy=False)
    sell_cross = is_down & ~is_down.shift(1).fillna(False).infer_objects(copy=False)
    ema_dist = ((e150 - e200) / e200).abs() * 100
    is_far = ema_dist >= SCALP_EMA_FAR
    tiny_dist = ((close - e200) / e200).abs() * 100
    near = tiny_dist <= SCALP_MAX_EMA200
    buy = buy_cross & (close > e150) & ~is_far & near
    sell = sell_cross & (close < e150) & ~is_far & near
    return apply_cooldown(buy, SCALP_COOLDOWN), apply_cooldown(sell, SCALP_COOLDOWN), "Scalp"


# ═══════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════

def fetch_data(exchange, symbol, months=BACKTEST_MONTHS):
    """Fetch 4H data — need extra for EMA500 warmup (~2000 bars = ~333 days)"""
    since = int((datetime.now(timezone.utc) - timedelta(days=365 * 2)).timestamp() * 1000)
    logger.info(f"  Fetching {symbol} ...")
    all_candles = []
    FETCH_LIMIT = 500  # MEXC returns max 500 per request
    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=FETCH_LIMIT)
        except Exception as e:
            logger.error(f"    Fetch error: {e}")
            break
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < FETCH_LIMIT:
            break
        time.sleep(0.5)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    # Keep only last BACKTEST_MONTHS for trading
    cutoff = df.index[-1] - pd.DateOffset(months=BACKTEST_MONTHS)
    df_full = df.copy()  # Full data for indicator warmup
    df_trade = df[df.index >= cutoff].copy()

    return df_full, df_trade


def prepare_indicators(df_full):
    """Calculate all indicators on full data"""
    df = df_full.copy()
    df["rsi"] = rsi_calc(df["close"], RSI_LENGTH)
    df["ema150"] = ema(df["close"], 150)
    df["ema200"] = ema(df["close"], 200)
    df["ema250"] = ema(df["close"], 250)
    df["ema500"] = ema(df["close"], 500)
    return df


def run_backtest_on_symbol(df_full, df_trade, symbol):
    """Run backtest for one symbol. Returns list of trade dicts."""
    df = prepare_indicators(df_full)

    # Get all signals
    all_systems = [
        rsi_system(df), kdj_system(df), sr_system(df),
        ots_system(df), vol_system(df), pressure_system(df),
    ]
    scalp_buy, scalp_sell, _ = scalp_system(df)

    # Build signal timeline (only in trade window)
    trade_start = df_trade.index[0]
    trade_end = df_trade.index[-1]

    trades = []

    for buy_col, sell_col, sys_name in all_systems:
        for idx in df.index:
            if idx < trade_start or idx > trade_end:
                continue
            # Only take signal at this exact bar
            if buy_col.loc[idx]:
                trades.append({
                    "time": idx, "symbol": symbol, "system": sys_name,
                    "type": "LONG", "entry": df.loc[idx, "close"],
                    "is_scalp": False
                })
            if sell_col.loc[idx]:
                trades.append({
                    "time": idx, "symbol": symbol, "system": sys_name,
                    "type": "SHORT", "entry": df.loc[idx, "close"],
                    "is_scalp": False
                })

    # Scalp signals
    for idx in df.index:
        if idx < trade_start or idx > trade_end:
            continue
        if scalp_buy.loc[idx]:
            trades.append({
                "time": idx, "symbol": symbol, "system": "Scalp",
                "type": "LONG", "entry": df.loc[idx, "close"],
                "is_scalp": True
            })
        if scalp_sell.loc[idx]:
            trades.append({
                "time": idx, "symbol": symbol, "system": "Scalp",
                "type": "SHORT", "entry": df.loc[idx, "close"],
                "is_scalp": True
            })

    return trades, df


def simulate_trades(trades, df):
    """Simulate trade outcomes using future bars"""
    results = []
    for t in trades:
        entry_price = t["entry"]
        is_long = t["type"] == "LONG"
        is_scalp = t["is_scalp"]

        if is_scalp:
            sl_pct = SCALP_SL_PCT
            tp1_pct = SCALP_TP_PCT
            # Scalp: simple TP1/SL, no TP2/TP3
            tp_levels = [(tp1_pct, 1.0)]  # (pct, weight)
        else:
            sl_pct = SL_PCT
            tp_levels = [
                (TP1_PCT, 0.50),   # 50% at TP1
                (TP2_PCT, 0.40),   # 40% at TP2
                (TP3_PCT, 0.10),   # 10% at TP3
            ]

        # Find the bar index in df
        try:
            bar_idx = df.index.get_loc(t["time"])
        except KeyError:
            continue

        # Look ahead for exit (max 50 bars = ~8 days)
        exit_price = None
        exit_bar = None
        exit_reason = ""

        if is_scalp:
            for j in range(bar_idx + 1, min(bar_idx + 50, len(df))):
                h = df.iloc[j]["high"]
                l = df.iloc[j]["low"]
                if is_long:
                    if l <= entry_price * (1 - sl_pct / 100):
                        exit_price = entry_price * (1 - sl_pct / 100)
                        exit_reason = "SL"; exit_bar = j; break
                    if h >= entry_price * (1 + tp1_pct / 100):
                        exit_price = entry_price * (1 + tp1_pct / 100)
                        exit_reason = "TP1"; exit_bar = j; break
                else:
                    if h >= entry_price * (1 + sl_pct / 100):
                        exit_price = entry_price * (1 + sl_pct / 100)
                        exit_reason = "SL"; exit_bar = j; break
                    if l <= entry_price * (1 - tp1_pct / 100):
                        exit_price = entry_price * (1 - tp1_pct / 100)
                        exit_reason = "TP1"; exit_bar = j; break
        else:
            # Multi-TP: track closed portions ACROSS bars
            remaining_weight = 1.0
            weighted_exit = 0.0
            tp_idx = 0  # which TP level we're waiting for
            closed_completely = False

            for j in range(bar_idx + 1, min(bar_idx + 50, len(df))):
                h = df.iloc[j]["high"]
                l = df.iloc[j]["low"]

                if is_long:
                    # Check SL first (on remaining portion)
                    if l <= entry_price * (1 - sl_pct / 100):
                        weighted_exit += remaining_weight * (entry_price * (1 - sl_pct / 100))
                        exit_price = weighted_exit
                        exit_reason = "SL"; exit_bar = j
                        closed_completely = True; break

                    # Check current TP level
                    if tp_idx < len(tp_levels):
                        tp_pct, weight = tp_levels[tp_idx]
                        if h >= entry_price * (1 + tp_pct / 100):
                            weighted_exit += weight * (entry_price * (1 + tp_pct / 100))
                            remaining_weight -= weight
                            tp_idx += 1
                            if remaining_weight <= 0.01:
                                exit_price = weighted_exit
                                exit_reason = f"TP{tp_idx}"
                                exit_bar = j
                                closed_completely = True; break
                else:
                    if h >= entry_price * (1 + sl_pct / 100):
                        weighted_exit += remaining_weight * (entry_price * (1 + sl_pct / 100))
                        exit_price = weighted_exit
                        exit_reason = "SL"; exit_bar = j
                        closed_completely = True; break

                    if tp_idx < len(tp_levels):
                        tp_pct, weight = tp_levels[tp_idx]
                        if l <= entry_price * (1 - tp_pct / 100):
                            weighted_exit += weight * (entry_price * (1 - tp_pct / 100))
                            remaining_weight -= weight
                            tp_idx += 1
                            if remaining_weight <= 0.01:
                                exit_price = weighted_exit
                                exit_reason = f"TP{tp_idx}"
                                exit_bar = j
                                closed_completely = True; break

            if not closed_completely and remaining_weight > 0.01:
                # Close remaining at last bar's close
                last_j = min(bar_idx + 50, len(df)) - 1
                weighted_exit += remaining_weight * df.iloc[last_j]["close"]
                exit_price = weighted_exit
                exit_reason = "TIMEOUT"
                exit_bar = last_j

        # If no exit found within max bars, close at last available close
        if exit_price is None:
            last_j = min(bar_idx + 50, len(df)) - 1
            exit_price = df.iloc[last_j]["close"]
            exit_reason = "TIMEOUT"
            exit_bar = last_j

        # Calculate PnL
        if is_long:
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        # Risk amount based on SL
        risk_amount = INITIAL_BALANCE * (RISK_PER_TRADE_PCT / 100)
        if is_scalp:
            risk_amount = INITIAL_BALANCE * (1.0 / 100)  # 1% risk for scalp

        # Position size (how many $ worth)
        if is_scalp:
            pos_size = risk_amount / (sl_pct / 100)
        else:
            pos_size = risk_amount / (SL_PCT / 100)

        pnl_dollar = pos_size * (pnl_pct / 100)

        # Bars held
        bars_held = exit_bar - bar_idx if exit_bar else 0

        results.append({
            "time": t["time"],
            "symbol": t["symbol"],
            "system": t["system"],
            "type": t["type"],
            "is_scalp": t["is_scalp"],
            "entry": entry_price,
            "exit": exit_price,
            "pnl_pct": round(pnl_pct, 3),
            "pnl_dollar": round(pnl_dollar, 2),
            "exit_reason": exit_reason,
            "bars_held": bars_held,
        })

    return results


def calculate_stats(trades):
    """Calculate comprehensive backtest statistics"""
    if not trades:
        return None

    df = pd.DataFrame(trades)
    df = df.sort_values("time").reset_index(drop=True)

    total_trades = len(df)
    wins = df[df["pnl_dollar"] > 0]
    losses = df[df["pnl_dollar"] <= 0]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    total_pnl = df["pnl_dollar"].sum()
    avg_win = wins["pnl_dollar"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl_dollar"].mean() if len(losses) > 0 else 0
    avg_trade = df["pnl_dollar"].mean()

    # Profit Factor
    gross_profit = wins["pnl_dollar"].sum()
    gross_loss = abs(losses["pnl_dollar"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown (running balance)
    running_balance = INITIAL_BALANCE + df["pnl_dollar"].cumsum()
    peak = running_balance.cummax()
    drawdown = (running_balance - peak) / peak * 100
    max_dd = drawdown.min()
    max_dd_pct = abs(max_dd)

    # Final balance
    final_balance = INITIAL_BALANCE + total_pnl
    return_pct = (total_pnl / INITIAL_BALANCE) * 100

    # Avg bars held
    avg_bars = df["bars_held"].mean()

    # Exit reason breakdown
    exit_counts = df["exit_reason"].value_counts().to_dict()

    # Per-system stats
    system_stats = {}
    for sys_name in df["system"].unique():
        sub = df[df["system"] == sys_name]
        sub_wins = sub[sub["pnl_dollar"] > 0]
        sub_profit = sub["pnl_dollar"].sum()
        sub_wr = (len(sub_wins) / len(sub) * 100) if len(sub) > 0 else 0
        system_stats[sys_name] = {
            "trades": len(sub),
            "wins": len(sub_wins),
            "pnl": round(sub_profit, 2),
            "win_rate": round(sub_wr, 1),
            "avg_pnl": round(sub["pnl_dollar"].mean(), 2),
        }

    # Per-symbol stats
    symbol_stats = {}
    for sym in df["symbol"].unique():
        sub = df[df["symbol"] == sym]
        sub_wins = sub[sub["pnl_dollar"] > 0]
        sub_profit = sub["pnl_dollar"].sum()
        sub_wr = (len(sub_wins) / len(sub) * 100) if len(sub) > 0 else 0
        symbol_stats[sym] = {
            "trades": len(sub),
            "pnl": round(sub_profit, 2),
            "win_rate": round(sub_wr, 1),
        }

    # Scalp vs Normal
    scalp_trades = df[df["is_scalp"] == True]
    normal_trades = df[df["is_scalp"] == False]

    return {
        "total_trades": total_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_trade": round(avg_trade, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "final_balance": round(final_balance, 2),
        "return_pct": round(return_pct, 2),
        "avg_bars": round(avg_bars, 1),
        "exit_counts": exit_counts,
        "system_stats": system_stats,
        "symbol_stats": symbol_stats,
        "scalp_trades": len(scalp_trades),
        "scalp_pnl": round(scalp_trades["pnl_dollar"].sum(), 2) if len(scalp_trades) > 0 else 0,
        "scalp_wr": round((len(scalp_trades[scalp_trades["pnl_dollar"] > 0]) / len(scalp_trades) * 100), 1) if len(scalp_trades) > 0 else 0,
        "normal_trades": len(normal_trades),
        "normal_pnl": round(normal_trades["pnl_dollar"].sum(), 2) if len(normal_trades) > 0 else 0,
        "normal_wr": round((len(normal_trades[normal_trades["pnl_dollar"] > 0]) / len(normal_trades) * 100), 1) if len(normal_trades) > 0 else 0,
    }


def send_telegram(text):
    if not BOT_TOKEN or not CHANNEL_ID:
        logger.warning("BOT_TOKEN or CHANNEL_ID not set — skipping Telegram")
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHANNEL_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.json().get("ok"):
            return True
        logger.error(f"TG error: {r.json()}")
    except Exception as e:
        logger.error(f"TG send error: {e}")
    return False


def build_telegram_message(stats, backtest_period):
    """Build the Telegram result message"""
    # Main stats
    msg = (
        f"📊 <b>OTS 4H Trading System V1.2 — Backtest Report</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 Period: {backtest_period}\n"
        f" pairs: {', '.join(SYMBOLS)}\n"
        f"Timeframe: 4H\n"
        f"Capital: ${INITIAL_BALANCE:,.0f} | Risk: {RISK_PER_TRADE_PCT}%/trade\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    msg += (
        f"📈 <b>OVERALL RESULTS</b>\n"
        f"Total Trades: <b>{stats['total_trades']}</b>\n"
        f"  Normal: {stats['normal_trades']} | Scalp: {stats['scalp_trades']}\n"
        f"Wins: {stats['win_count']} | Losses: {stats['loss_count']}\n"
        f"Win Rate: <b>{stats['win_rate']}%</b>\n"
        f"  Normal WR: {stats['normal_wr']}% | Scalp WR: {stats['scalp_wr']}%\n"
        f"Profit Factor: <b>{stats['profit_factor']}</b>\n\n"
    )

    msg += (
        f"💰 <b>PROFIT & LOSS</b>\n"
        f"Total PnL: <b>${stats['total_pnl']:,.2f}</b>\n"
        f"  Normal PnL: ${stats['normal_pnl']:,.2f}\n"
        f"  Scalp PnL: ${stats['scalp_pnl']:,.2f}\n"
        f"Gross Profit: ${stats['gross_profit']:,.2f}\n"
        f"Gross Loss: ${stats['gross_loss']:,.2f}\n"
        f"Avg Win: ${stats['avg_win']:,.2f} | Avg Loss: ${stats['avg_loss']:,.2f}\n"
        f"Avg Trade: ${stats['avg_trade']:,.2f}\n\n"
    )

    msg += (
        f"📉 <b>RISK METRICS</b>\n"
        f"Max Drawdown: <b>{stats['max_dd_pct']}%</b>\n"
        f"Return: <b>{stats['return_pct']}%</b>\n"
        f"Final Balance: <b>${stats['final_balance']:,.2f}</b>\n"
        f"Avg Bars Held: {stats['avg_bars']}\n\n"
    )

    # System breakdown
    msg += f"🧩 <b>SYSTEM BREAKDOWN</b>\n"
    for sys_name, ss in sorted(stats["system_stats"].items(), key=lambda x: x[1]["pnl"], reverse=True):
        emoji = "🟢" if ss["pnl"] > 0 else "🔻"
        msg += (
            f"  {emoji} {sys_name}: {ss['trades']} trades | "
            f"WR {ss['win_rate']}% | PnL ${ss['pnl']:,.2f}\n"
        )

    # Symbol breakdown
    msg += f"\n📊 <b>PAIR BREAKDOWN</b>\n"
    for sym, ss in sorted(stats["symbol_stats"].items(), key=lambda x: x[1]["pnl"], reverse=True):
        emoji = "🟢" if ss["pnl"] > 0 else "🔻"
        msg += (
            f"  {emoji} {sym}: {ss['trades']} trades | "
            f"WR {ss['win_rate']}% | PnL ${ss['pnl']:,.2f}\n"
        )

    # Exit reasons
    msg += f"\n🏁 <b>EXIT REASONS</b>\n"
    for reason, count in stats["exit_counts"].items():
        msg += f"  {reason}: {count}\n"

    return msg


def main():
    start_time = time.time()
    logger.info(f"=== OTS 4H Backtest — {BACKTEST_MONTHS} months ===")
    logger.info(f"Pairs: {SYMBOLS}")
    logger.info(f"Capital: ${INITIAL_BALANCE:,.0f}")

    exchange = ccxt.__dict__.get(EXCHANGE_NAME, ccxt.mexc)()
    exchange.enableRateLimit = True

    all_trades = []

    for symbol in SYMBOLS:
        logger.info(f"\n--- Processing {symbol} ---")
        try:
            df_full, df_trade = fetch_data(exchange, symbol)
        except Exception as e:
            logger.error(f"  Failed to fetch {symbol}: {e}")
            continue

        logger.info(f"  Full data: {len(df_full)} bars | Trade window: {len(df_trade)} bars")
        logger.info(f"  Trade window: {df_trade.index[0]} to {df_trade.index[-1]}")

        trades, df = run_backtest_on_symbol(df_full, df_trade, symbol)
        logger.info(f"  Signals found: {len(trades)}")

        results = simulate_trades(trades, df)
        all_trades.extend(results)
        logger.info(f"  Simulated trades: {len(results)}")

    if not all_trades:
        logger.error("No trades generated!")
        send_telegram("⚠️ OTS Backtest: No trades generated. Check data or parameters.")
        return

    stats = calculate_stats(all_trades)
    elapsed = time.time() - start_time

    # Build period string
    df_trades = pd.DataFrame(all_trades)
    period_start = df_trades["time"].min().strftime("%Y-%m-%d")
    period_end = df_trades["time"].max().strftime("%Y-%m-%d")
    backtest_period = f"{period_start} → {period_end}"

    # Print results
    logger.info(f"\n{'='*50}")
    logger.info(f"BACKTEST RESULTS — {backtest_period}")
    logger.info(f"{'='*50}")
    logger.info(f"Total Trades: {stats['total_trades']}")
    logger.info(f"Win Rate: {stats['win_rate']}%")
    logger.info(f"Profit Factor: {stats['profit_factor']}")
    logger.info(f"Total PnL: ${stats['total_pnl']:,.2f}")
    logger.info(f"Return: {stats['return_pct']}%")
    logger.info(f"Max Drawdown: {stats['max_dd_pct']}%")
    logger.info(f"Final Balance: ${stats['final_balance']:,.2f}")
    logger.info(f"\nPer System:")
    for sys_name, ss in sorted(stats["system_stats"].items(), key=lambda x: x[1]["pnl"], reverse=True):
        logger.info(f"  {sys_name}: {ss['trades']} trades, WR {ss['win_rate']}%, PnL ${ss['pnl']:,.2f}")
    logger.info(f"\nPer Pair:")
    for sym, ss in sorted(stats["symbol_stats"].items(), key=lambda x: x[1]["pnl"], reverse=True):
        logger.info(f"  {sym}: {ss['trades']} trades, WR {ss['win_rate']}%, PnL ${ss['pnl']:,.2f}")
    logger.info(f"\nScalp: {stats['scalp_trades']} trades, WR {stats['scalp_wr']}%, PnL ${stats['scalp_pnl']:,.2f}")
    logger.info(f"Normal: {stats['normal_trades']} trades, WR {stats['normal_wr']}%, PnL ${stats['normal_pnl']:,.2f}")
    logger.info(f"\nElapsed: {elapsed:.1f}s")

    # Send to Telegram
    msg = build_telegram_message(stats, backtest_period)
    msg += f"\n⏱ Completed in {elapsed:.0f}s"

    sent = send_telegram(msg)
    if sent:
        logger.info("Results sent to Telegram!")
    else:
        logger.warning("Failed to send to Telegram. Check BOT_TOKEN and CHANNEL_ID.")
        # Print message so user can see it anyway
        print("\n" + "="*50)
        print("TELEGRAM MESSAGE (not sent):")
        print("="*50)
        print(msg)


if __name__ == "__main__":
    main()