"""
OTS Precision Bot V2.0 — Backtest Engine
VOL Core + Confluence (RSI/S/R/Pressure) + Daily Trend + ADX Filter
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
logger = logging.getLogger("OTS-Precision-BT")

# ═══════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHANNEL_ID = os.environ.get("CHANNEL_ID", "")
EXCHANGE_NAME = os.environ.get("EXCHANGE_NAME", "mexc")
SYMBOLS = ["BTC/USDT", "ADA/USDT", "XRP/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT"]
TIMEFRAME = "1h"
INITIAL_BALANCE = 10000.0
RISK_PER_TRADE_PCT = 2.0
BACKTEST_MONTHS = 4

# VOL SYSTEM (Core)
VOL_LENGTH = 14
VOL_MULTIPLIER = 2.5
VOL_COOLDOWN = 10
VOL_DISTANCE_PERCENT = 7.0

# RSI SYSTEM (Confluence)
RSI_LENGTH = 14
RSI_UPPER = 65.0
RSI_LOWER = 35.0
RSI_MAX_DISTANCE = 2.2
RSI_MAX_CANDLE = 2.0

# S/R SYSTEM (Confluence)
SR_MAX_DIST_EMA = 2.0
SR_MAX_CANDLE = 2.0

# PRESSURE SYSTEM (Confluence)
PRESSURE_RSI_PERIOD = 50
PRESSURE_RATE = 45
PRESSURE_THRESHOLD = 500
PRESSURE_EMA_DIST = 5.0

# FILTERS
ADX_LENGTH = 14
ADX_MIN = 10.0

# DAILY TREND
DAILY_EMA_FAST = 50
DAILY_EMA_SLOW = 200

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

def rsi_calc(close, length=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    up_r = rma(up, length)
    down_r = rma(down, length)
    val = np.where(down_r == 0, 100.0, np.where(up_r == 0, 0.0, 100.0 - 100.0 / (1.0 + up_r / down_r)))
    return pd.Series(val, index=close.index)

def adx_calc(high, low, close, length=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = rma(tr, length)
    plus_di = 100 * rma(plus_dm, length) / atr
    minus_di = 100 * rma(minus_dm, length) / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = rma(dx, length)
    return adx, plus_di, minus_di

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
#  SIGNAL SYSTEMS
# ═══════════════════════════════════════════

def vol_system(df):
    close, open_, vol, e250 = df["close"], df["open"], df["volume"], df["ema250"]
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
    return apply_cooldown(buy, VOL_COOLDOWN), apply_cooldown(sell, VOL_COOLDOWN)


def rsi_confluence(df):
    """RSI direction confluence"""
    rsi = df["rsi"]
    buy = (rsi > 50.0)   # RSI bullish
    sell = (rsi < 50.0)  # RSI bearish
    return buy, sell


def sr_confluence(df):
    """S/R extreme confluence"""
    rsi = df["rsi"]
    buy = (rsi > 60.0)   # RSI strongly bullish
    sell = (rsi < 40.0)  # RSI strongly bearish
    return buy, sell


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
    return buy, sell


# ═══════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════

def fetch_data(exchange, symbol, timeframe):
    since = int((datetime.now(timezone.utc) - timedelta(days=730)).timestamp() * 1000)
    logger.info(f"  Fetching {symbol} {timeframe} ...")
    all_candles = []
    FETCH_LIMIT = 500
    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=FETCH_LIMIT)
        except Exception as e:
            logger.error(f"    Fetch error: {e}")
            break
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < FETCH_LIMIT:
            break
        time.sleep(0.3)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()


def prepare_1h(df_full):
    df = df_full.copy()
    df["rsi"] = rsi_calc(df["close"], RSI_LENGTH)
    df["ema150"] = ema(df["close"], 150)
    df["ema200"] = ema(df["close"], 200)
    df["ema250"] = ema(df["close"], 250)
    df["ema500"] = ema(df["close"], 500)
    df["adx"], df["plus_di"], df["minus_di"] = adx_calc(df["high"], df["low"], df["close"], ADX_LENGTH)
    return df


def get_daily_trend_series(df_daily):
    """Return series of daily trend: 1=BULLISH, -1=BEARISH, 0=NEUTRAL"""
    e50 = ema(df_daily["close"], DAILY_EMA_FAST)
    e200 = ema(df_daily["close"], DAILY_EMA_SLOW)
    price = df_daily["close"]

    trend = pd.Series(0, index=df_daily.index)
    trend[(e50 > e200) & (price > e50)] = 1   # BULLISH
    trend[(e50 < e200) & (price < e50)] = -1  # BEARISH
    return trend  # 1=BULL, -1=BEAR, 0=NEUTRAL


def run_backtest_symbol(exchange, symbol):
    # Fetch data
    df_1h_full = fetch_data(exchange, symbol, TIMEFRAME)
    df_daily_full = fetch_data(exchange, symbol, "1d")

    if df_1h_full is None or df_daily_full is None or len(df_1h_full) < 600:
        return []

    # Prepare indicators
    df = prepare_1h(df_1h_full)
    daily_trend = get_daily_trend_series(df_daily_full)

    # Map daily trend to 1H bars using reindex + ffill
    daily_1h = daily_trend.reindex(df.index, method="ffill").fillna(0)

    # Get signals
    vol_buy, vol_sell = vol_system(df)
    rsi_buy, rsi_sell = rsi_confluence(df)
    sr_buy, sr_sell = sr_confluence(df)
    prs_buy, prs_sell = pressure_system(df)

    # Trade window
    cutoff = df.index[-1] - pd.DateOffset(months=BACKTEST_MONTHS)

    # Build trades
    trades = []
    for i in range(len(df)):
        idx = df.index[i]
        if idx < cutoff:
            continue

        dt = daily_1h.iloc[i]
        # Determine minimum confluence and trend alignment
        if dt == 0:
            min_conf = 1  # NEUTRAL: 1+ confluence
            allow_long = True
            allow_short = True
        elif dt == 1:  # BULLISH
            min_conf = 1
            allow_long = True
            allow_short = False
        else:  # BEARISH
            min_conf = 1
            allow_long = False
            allow_short = True

        # ADX filter
        adx_val = df["adx"].iloc[i]
        if pd.isna(adx_val) or adx_val < ADX_MIN:
            continue

        # Check LONG
        if vol_buy.iloc[i] and allow_long:
            confluence = []
            if rsi_buy.iloc[i]: confluence.append("RSI")
            if sr_buy.iloc[i]: confluence.append("S/R")
            if prs_buy.iloc[i]: confluence.append("Pressure")
            if len(confluence) >= min_conf:
                trades.append({
                    "time": idx, "symbol": symbol, "type": "LONG",
                    "entry": df["close"].iloc[i],
                    "confluence": confluence, "adx": adx_val,
                    "strength": len(confluence),
                })

        # Check SHORT
        if vol_sell.iloc[i] and allow_short:
            confluence = []
            if rsi_sell.iloc[i]: confluence.append("RSI")
            if sr_sell.iloc[i]: confluence.append("S/R")
            if prs_sell.iloc[i]: confluence.append("Pressure")
            if len(confluence) >= min_conf:
                trades.append({
                    "time": idx, "symbol": symbol, "type": "SHORT",
                    "entry": df["close"].iloc[i],
                    "confluence": confluence, "adx": adx_val,
                    "strength": len(confluence),
                })

    return trades, df


def simulate_trades(trades, df):
    tp_levels = [(TP1_PCT, 0.50), (TP2_PCT, 0.40), (TP3_PCT, 0.10)]
    results = []

    for t in trades:
        entry = t["entry"]
        is_long = t["type"] == "LONG"

        try:
            bar_idx = df.index.get_loc(t["time"])
        except KeyError:
            continue

        remaining_weight = 1.0
        weighted_exit = 0.0
        tp_idx = 0
        closed = False

        for j in range(bar_idx + 1, min(bar_idx + 200, len(df))):
            h, l = df.iloc[j]["high"], df.iloc[j]["low"]

            if is_long:
                if l <= entry * (1 - SL_PCT / 100):
                    weighted_exit += remaining_weight * entry * (1 - SL_PCT / 100)
                    closed = True; break
                if tp_idx < len(tp_levels):
                    tp_pct, weight = tp_levels[tp_idx]
                    if h >= entry * (1 + tp_pct / 100):
                        weighted_exit += weight * entry * (1 + tp_pct / 100)
                        remaining_weight -= weight
                        tp_idx += 1
                        if remaining_weight <= 0.01:
                            closed = True; break
            else:
                if h >= entry * (1 + SL_PCT / 100):
                    weighted_exit += remaining_weight * entry * (1 + SL_PCT / 100)
                    closed = True; break
                if tp_idx < len(tp_levels):
                    tp_pct, weight = tp_levels[tp_idx]
                    if l <= entry * (1 - tp_pct / 100):
                        weighted_exit += weight * entry * (1 - tp_pct / 100)
                        remaining_weight -= weight
                        tp_idx += 1
                        if remaining_weight <= 0.01:
                            closed = True; break

        if not closed and remaining_weight > 0.01:
            last_j = min(bar_idx + 200, len(df)) - 1
            weighted_exit += remaining_weight * df.iloc[last_j]["close"]
            exit_reason = "TIMEOUT"
            exit_bar = last_j
        else:
            exit_reason = "SL" if not closed and remaining_weight <= 0.01 else ("TP3" if remaining_weight <= 0.01 else "SL")
            if closed:
                # Determine if last action was SL or TP
                if is_long:
                    if l <= entry * (1 - SL_PCT / 100):
                        exit_reason = "SL"
                    else:
                        exit_reason = f"TP{tp_idx}" if tp_idx > 0 else "TP1"
                else:
                    if h >= entry * (1 + SL_PCT / 100):
                        exit_reason = "SL"
                    else:
                        exit_reason = f"TP{tp_idx}" if tp_idx > 0 else "TP1"
            exit_bar = j if closed else last_j

        if is_long:
            pnl_pct = (weighted_exit - entry) / entry * 100
        else:
            pnl_pct = (entry - weighted_exit) / entry * 100

        risk_amount = INITIAL_BALANCE * (RISK_PER_TRADE_PCT / 100)
        pos_size = risk_amount / (SL_PCT / 100)
        pnl_dollar = pos_size * (pnl_pct / 100)

        results.append({
            "time": t["time"], "symbol": t["symbol"],
            "type": t["type"], "entry": entry,
            "confluence": t["confluence"], "strength": t["strength"],
            "pnl_pct": round(pnl_pct, 3), "pnl_dollar": round(pnl_dollar, 2),
            "exit_reason": exit_reason,
        })

    return results


def calculate_stats(trades_df):
    if trades_df.empty:
        return None

    total = len(trades_df)
    wins = trades_df[trades_df["pnl_dollar"] > 0]
    losses = trades_df[trades_df["pnl_dollar"] <= 0]
    gross_profit = wins["pnl_dollar"].sum()
    gross_loss = abs(losses["pnl_dollar"].sum())
    total_pnl = trades_df["pnl_dollar"].sum()

    running = INITIAL_BALANCE + trades_df["pnl_dollar"].cumsum()
    peak = running.cummax()
    max_dd = abs(((running - peak) / peak * 100).min())

    # Per strength
    strength_stats = {}
    for s in sorted(trades_df["strength"].unique()):
        sub = trades_df[trades_df["strength"] == s]
        sw = sub[sub["pnl_dollar"] > 0]
        strength_stats[s] = {
            "trades": len(sub),
            "wr": round(len(sw) / len(sub) * 100, 1) if len(sub) > 0 else 0,
            "pnl": round(sub["pnl_dollar"].sum(), 2),
        }

    # Per pair
    pair_stats = {}
    for sym in trades_df["symbol"].unique():
        sub = trades_df[trades_df["symbol"] == sym]
        sw = sub[sub["pnl_dollar"] > 0]
        pair_stats[sym] = {
            "trades": len(sub),
            "wr": round(len(sw) / len(sub) * 100, 1) if len(sub) > 0 else 0,
            "pnl": round(sub["pnl_dollar"].sum(), 2),
        }

    # Per confluence combo
    conf_stats = {}
    for _, row in trades_df.iterrows():
        key = " + ".join(sorted(row["confluence"]))
        if key not in conf_stats:
            conf_stats[key] = {"trades": 0, "wins": 0, "pnl": 0.0}
        conf_stats[key]["trades"] += 1
        if row["pnl_dollar"] > 0:
            conf_stats[key]["wins"] += 1
        conf_stats[key]["pnl"] += row["pnl_dollar"]
    for k in conf_stats:
        conf_stats[k]["wr"] = round(conf_stats[k]["wins"] / conf_stats[k]["trades"] * 100, 1)
        conf_stats[k]["pnl"] = round(conf_stats[k]["pnl"], 2)

    return {
        "total_trades": total,
        "win_rate": round(len(wins) / total * 100, 1) if total > 0 else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999,
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl / INITIAL_BALANCE * 100, 2),
        "max_dd": round(max_dd, 2),
        "final_balance": round(INITIAL_BALANCE + total_pnl, 2),
        "avg_win": round(wins["pnl_dollar"].mean(), 2) if len(wins) > 0 else 0,
        "avg_loss": round(losses["pnl_dollar"].mean(), 2) if len(losses) > 0 else 0,
        "strength_stats": strength_stats,
        "pair_stats": pair_stats,
        "conf_stats": conf_stats,
    }


def send_telegram(text):
    if not BOT_TOKEN or not CHANNEL_ID:
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHANNEL_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
        return r.json().get("ok", False)
    except:
        return False


def build_message(stats, period):
    msg = (
        f"📊 <b>OTS Precision V2.0 — Backtest Report</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 Period: {period}\n"
        f"Pairs: {', '.join(SYMBOLS)}\n"
        f"Timeframe: 1H | Filters: Daily Trend + ADX>{ADX_MIN} + Confluence\n"
        f"Capital: ${INITIAL_BALANCE:,.0f} | Risk: {RISK_PER_TRADE_PCT}%/trade\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    msg += (
        f"📈 <b>OVERALL</b>\n"
        f"Total Trades: <b>{stats['total_trades']}</b>\n"
        f"Win Rate: <b>{stats['win_rate']}%</b>\n"
        f"Profit Factor: <b>{stats['profit_factor']}</b>\n"
        f"Total PnL: <b>${stats['total_pnl']:,.2f}</b>\n"
        f"Return: <b>{stats['return_pct']}%</b>\n"
        f"Max Drawdown: <b>{stats['max_dd']}%</b>\n"
        f"Final Balance: <b>${stats['final_balance']:,.2f}</b>\n\n"
    )

    msg += f"💪 <b>STRENGTH (Confluence Count)</b>\n"
    for s, ss in sorted(stats["strength_stats"].items()):
        emoji = "🟢" if ss["pnl"] > 0 else "🔻"
        msg += f"  {emoji} {s}/3 confluence: {ss['trades']} trades | WR {ss['wr']}% | PnL ${ss['pnl']:,.2f}\n"

    msg += f"\n📊 <b>PAIR BREAKDOWN</b>\n"
    for sym, ps in sorted(stats["pair_stats"].items(), key=lambda x: x[1]["pnl"], reverse=True):
        emoji = "🟢" if ps["pnl"] > 0 else "🔻"
        msg += f"  {emoji} {sym}: {ps['trades']} trades | WR {ps['wr']}% | PnL ${ps['pnl']:,.2f}\n"

    msg += f"\n🧩 <b>CONFLUENCE COMBOS</b>\n"
    for combo, cs in sorted(stats["conf_stats"].items(), key=lambda x: x[1]["pnl"], reverse=True):
        emoji = "🟢" if cs["pnl"] > 0 else "🔻"
        msg += f"  {emoji} {combo}: {cs['trades']} trades | WR {cs['wr']}% | PnL ${cs['pnl']:,.2f}\n"

    return msg


def main():
    start = time.time()
    logger.info(f"=== OTS Precision V2.0 Backtest — {BACKTEST_MONTHS} months ===")

    exchange = ccxt.__dict__.get(EXCHANGE_NAME, ccxt.mexc)()
    exchange.enableRateLimit = True

    all_results = []
    for symbol in SYMBOLS:
        logger.info(f"\n--- {symbol} ---")
        result = run_backtest_symbol(exchange, symbol)
        if result is None:
            continue
        trades, df = result
        logger.info(f"  Precision signals: {len(trades)}")
        simulated = simulate_trades(trades, df)
        all_results.extend(simulated)
        logger.info(f"  Simulated: {len(simulated)}")

    if not all_results:
        logger.error("No trades!")
        send_telegram("⚠️ Precision V2.0 Backtest: No trades generated.")
        return

    df_trades = pd.DataFrame(all_results).sort_values("time").reset_index(drop=True)
    stats = calculate_stats(df_trades)

    period = f"{df_trades['time'].min().strftime('%Y-%m-%d')} → {df_trades['time'].max().strftime('%Y-%m-%d')}"

    logger.info(f"\n{'='*50}")
    logger.info(f"PRECISION V2.0 RESULTS — {period}")
    logger.info(f"{'='*50}")
    logger.info(f"Trades: {stats['total_trades']} | WR: {stats['win_rate']}% | PF: {stats['profit_factor']}")
    logger.info(f"PnL: ${stats['total_pnl']:,.2f} ({stats['return_pct']}%) | DD: {stats['max_dd']}%")
    for s, ss in sorted(stats["strength_stats"].items()):
        logger.info(f"  Strength {s}/3: {ss['trades']} trades, WR {ss['wr']}%, PnL ${ss['pnl']:,.2f}")
    for combo, cs in sorted(stats["conf_stats"].items(), key=lambda x: x[1]["pnl"], reverse=True):
        logger.info(f"  {combo}: {cs['trades']} trades, WR {cs['wr']}%, PnL ${cs['pnl']:,.2f}")

    msg = build_message(stats, period)
    msg += f"\n⏱ {time.time() - start:.0f}s"

    if send_telegram(msg):
        logger.info("Results sent to Telegram!")
    else:
        logger.warning("Telegram not sent.")
        print("\n" + msg)


if __name__ == "__main__":
    main()
