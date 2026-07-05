import os
import ssl
import requests
import time
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

ssl._create_default_https_context = ssl._create_unverified_context

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHANNEL_ID = os.environ.get("CHANNEL_ID")

PAIRS = [
    ("BTC-USD",  "ETH-USD",  "BTCUSDT",  "ETHUSDT"),
    ("SOL-USD",  "AVAX-USD", "SOLUSDT",  "AVAXUSDT"),
    ("ADA-USD",  "DOT-USD",  "ADAUSDT",  "DOTUSDT"),
    ("DOGE-USD", "SHIB-USD", "DOGEUSDT", "SHIBUSDT"),
    ("LINK-USD", "AAVE-USD", "LINKUSDT", "AAVEUSDT"),
]

LEVERAGE = "10"
TIMEFRAME = "4H"

Z_ENTRY = 1.2
TP1_PCT = 1.0
TP2_PCT = 2.0
SL_PCT = 2.5
COINT_PVALUE_MAX = 0.10
ROLLING_WINDOW = 20
OLS_WINDOW = 80

COIN_NAMES = {
    "BTCUSDT": "Bitcoin", "ETHUSDT": "Ethereum",
    "SOLUSDT": "Solana", "AVAXUSDT": "Avalanche",
    "ADAUSDT": "Cardano", "DOTUSDT": "Polkadot",
    "DOGEUSDT": "Dogecoin", "SHIBUSDT": "Shiba Inu",
    "LINKUSDT": "Chainlink", "AAVEUSDT": "Aave",
}

def get_decimals(price):
    if price > 100: return 2
    elif price > 1: return 3
    elif price > 0.01: return 5
    else: return 8

def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHANNEL_ID, "text": text, "disable_web_page_preview": True}
    try:
        requests.post(url, json=payload)
        time.sleep(1)
    except Exception as e:
        print(f"Error sending message: {e}")

def calc_targets(current_p, is_long, decimals):
    if is_long:
        tp1 = round(current_p * (1 + TP1_PCT / 100), decimals)
        tp2 = round(current_p * (1 + TP2_PCT / 100), decimals)
        sl = round(current_p * (1 - SL_PCT / 100), decimals)
    else:
        tp1 = round(current_p * (1 - TP1_PCT / 100), decimals)
        tp2 = round(current_p * (1 - TP2_PCT / 100), decimals)
        sl = round(current_p * (1 + SL_PCT / 100), decimals)
    return tp1, tp2, sl

def fetch_4h_data(ticker):
    df = yf.download(ticker, period="60d", interval="1h", progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.resample("4h").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()
    df.index = df.index.tz_localize(None)
    return df["Close"]

def analyze_pairs():
    print("Fetching 4H data and calculating Z-Scores...")

    for yf_ticker1, yf_ticker2, name1, name2 in PAIRS:
        try:
            c1 = fetch_4h_data(yf_ticker1)
            c2 = fetch_4h_data(yf_ticker2)

            if c1.empty or c2.empty:
                print(f"{name1}/{name2} no data. Skipping.")
                continue

            common = c1.index.intersection(c2.index)
            p1 = c1.loc[common]
            p2 = c2.loc[common]

            if len(p1) < 60:
                print(f"{name1}/{name2} not enough data. Skipping.")
                continue

            X = sm.add_constant(p2)
            model = sm.OLS(p1, X).fit()
            hedge_ratio = model.params.iloc[1]
            spread = p1 - (hedge_ratio * p2)

            spread_mean = spread.rolling(window=ROLLING_WINDOW).mean()
            spread_std = spread.rolling(window=ROLLING_WINDOW).std()
            z_score = (spread - spread_mean) / spread_std
            current_z = z_score.iloc[-1]

            if pd.isna(current_z) or spread_std.iloc[-1] < 1e-8:
                continue

            _, p_value, _ = sm.tsa.stattools.coint(p1, p2)
            if p_value > COINT_PVALUE_MAX:
                print(f"{name1}/{name2} p-value: {p_value:.3f}. Skipping.")
                continue

            print(f"{name1}/{name2} Z: {current_z:.2f} | p: {p_value:.4f} | H: {hedge_ratio:.4f}")

            if abs(current_z) > Z_ENTRY:
                current_p1 = p1.iloc[-1]
                current_p2 = p2.iloc[-1]
                dec1 = get_decimals(current_p1)
                dec2 = get_decimals(current_p2)

                zone1_low = round(current_p1 * 0.999, dec1)
                zone1_high = round(current_p1 * 1.001, dec1)
                zone2_low = round(current_p2 * 0.999, dec2)
                zone2_high = round(current_p2 * 1.001, dec2)

                if current_z > 0:
                    dir1, emoji1 = "Short", "📉"
                    dir2, emoji2 = "Long", "📈"
                    p1_long, p2_long = False, True
                else:
                    dir1, emoji1 = "Long", "📈"
                    dir2, emoji2 = "Short", "📉"
                    p1_long, p2_long = True, False

                tp1_p1, tp2_p1, sl_p1 = calc_targets(current_p1, p1_long, dec1)
                tp1_p2, tp2_p2, sl_p2 = calc_targets(current_p2, p2_long, dec2)

                text1 = (f"📩 #{name1} {TIMEFRAME} | Pairs Trade\n"
                         f"{emoji1} {dir1} Entry Zone: {zone1_low}-{zone1_high}\n"
                         f"⚡ Leverage: {LEVERAGE}x\n\n"
                         f"🎯 Statistical Arbitrage (Z: {current_z:.2f})\n"
                         f"📊 Coint: {p_value:.4f} | Hedge: {hedge_ratio:.4f}\n\n"
                         f"⏳ Target 1 ({TP1_PCT}%): {tp1_p1}\n"
                         f"Target 2 ({TP2_PCT}%): {tp2_p1}\n\n"
                         f"🔻 Stop-Loss ({SL_PCT}%): {sl_p1}\n"
                         f"💡 Paired with {COIN_NAMES.get(name2, name2)}\n"
                         f"🔁 Close both when Z crosses 0")

                text2 = (f"📩 #{name2} {TIMEFRAME} | Pairs Trade\n"
                         f"{emoji2} {dir2} Entry Zone: {zone2_low}-{zone2_high}\n"
                         f"⚡ Leverage: {LEVERAGE}x\n\n"
                         f"🎯 Statistical Arbitrage (Z: {current_z:.2f})\n"
                         f"📊 Coint: {p_value:.4f} | Hedge: {hedge_ratio:.4f}\n\n"
                         f"⏳ Target 1 ({TP1_PCT}%): {tp1_p2}\n"
                         f"Target 2 ({TP2_PCT}%): {tp2_p2}\n\n"
                         f"🔻 Stop-Loss ({SL_PCT}%): {sl_p2}\n"
                         f"💡 Paired with {COIN_NAMES.get(name1, name1)}\n"
                         f"🔁 Close both when Z crosses 0")

                print(f"Sending: {name1}({dir1}) / {name2}({dir2})")
                send_message(text1)
                send_message(text2)

        except Exception as e:
            print(f"Error {yf_ticker1}/{yf_ticker2}: {e}")

if __name__ == "__main__":
    print("Pairs Bot V3 (4H) started...")
    analyze_pairs()
    print("Scan finished.")
