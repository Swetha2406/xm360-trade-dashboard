"""
XM360 Trade Dashboard — MT5 Live Data Backend
==============================================
Pulls data DIRECTLY from XM's servers via MetaTrader5 Python API.
Same prices, same spread, same candles as what you see in XM 360 app.

SETUP:
  pip install MetaTrader5 pandas numpy flask flask-cors

REQUIREMENTS:
  - MetaTrader5 (XM) must be installed and logged in on your PC
  - MT5 must be running in the background
  - Only works on Windows (MT5 Python API is Windows-only)

Run: python backend/server_mt5.py
Open: http://localhost:5000
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json, os, time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Try MT5 first, fall back to yfinance ─────────────────────────────────────
MT5_OK = False
YF_OK  = False

try:
    import MetaTrader5 as mt5
    if mt5.initialize():
        MT5_OK = True
        info = mt5.terminal_info()
        print(f"\n  ✓ MetaTrader5 connected!")
        print(f"  Broker: {mt5.account_info().company if mt5.account_info() else 'unknown'}")
        print(f"  Account: {mt5.account_info().login if mt5.account_info() else 'unknown'}")
        print(f"  Balance: ${mt5.account_info().balance:.2f}" if mt5.account_info() else "")
    else:
        print(f"\n  ✗ MT5 init failed: {mt5.last_error()}")
        print("  Make sure MetaTrader5 (XM) is open and logged in")
except ImportError:
    print("\n  ✗ MetaTrader5 not installed")
    print("  Run: pip install MetaTrader5")
except Exception as e:
    print(f"\n  ✗ MT5 error: {e}")

if not MT5_OK:
    try:
        import yfinance as yf
        YF_OK = True
        print("  ✓ Falling back to yfinance (15min delayed)")
    except ImportError:
        print("  ✗ yfinance not found either")
        print("  Run: pip install MetaTrader5 yfinance")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(BASE_DIR, "frontend")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND)
CORS(app)

# ── Instrument Config ─────────────────────────────────────────────────────────
# MT5 symbol names may vary by broker — XM uses these
INSTRUMENTS = {
    "GOLD":   {"mt5_sym":"XAUUSD",   "yf_ticker":"GC=F",     "label":"GOLD.i#",  "pip":0.01,   "digits":2, "pip_val":1.0,  "spread_pips":3},
    "EURUSD": {"mt5_sym":"EURUSD",   "yf_ticker":"EURUSD=X", "label":"EURUSD#",  "pip":0.0001, "digits":5, "pip_val":10.0, "spread_pips":1},
    "GBPUSD": {"mt5_sym":"GBPUSD",   "yf_ticker":"GBPUSD=X", "label":"GBPUSD#",  "pip":0.0001, "digits":5, "pip_val":10.0, "spread_pips":1},
    "USDJPY": {"mt5_sym":"USDJPY",   "yf_ticker":"USDJPY=X", "label":"USDJPY#",  "pip":0.01,   "digits":3, "pip_val":9.0,  "spread_pips":1},
    "GBPJPY": {"mt5_sym":"GBPJPY",   "yf_ticker":"GBPJPY=X", "label":"GBPJPY#",  "pip":0.01,   "digits":3, "pip_val":9.0,  "spread_pips":2},
    "EURJPY": {"mt5_sym":"EURJPY",   "yf_ticker":"EURJPY=X", "label":"EURJPY#",  "pip":0.01,   "digits":3, "pip_val":9.0,  "spread_pips":2},
}

# MT5 Timeframe mapping
MT5_TF = {
    "1m":  mt5.TIMEFRAME_M1  if MT5_OK else None,
    "5m":  mt5.TIMEFRAME_M5  if MT5_OK else None,
    "15m": mt5.TIMEFRAME_M15 if MT5_OK else None,
    "1h":  mt5.TIMEFRAME_H1  if MT5_OK else None,
    "4h":  mt5.TIMEFRAME_H4  if MT5_OK else None,
    "1d":  mt5.TIMEFRAME_D1  if MT5_OK else None,
}

_cache = {}
CACHE_TTL_MT5 = 10   # 10 seconds for MT5 (near real-time)
CACHE_TTL_YF  = 60   # 60 seconds for yfinance


# ── FETCH FROM MT5 (XM direct) ────────────────────────────────────────────────
def fetch_mt5(symbol: str, tf: str = "1h", n_candles: int = 500) -> tuple:
    """
    Pull candles directly from XM's servers via MT5.
    Returns EXACTLY what you see in MetaTrader5 — same price, same spread.
    """
    if not MT5_OK:
        return None, "mt5_not_connected"

    inst    = INSTRUMENTS[symbol]
    mt5_sym = inst["mt5_sym"]
    mt5_tf  = MT5_TF.get(tf)

    if mt5_tf is None:
        return None, "invalid_tf"

    try:
        # Get candles from MT5
        rates = mt5.copy_rates_from_pos(mt5_sym, mt5_tf, 0, n_candles)

        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            print(f"  ✗ MT5 no data for {mt5_sym}: {err}")
            # Try enabling symbol
            mt5.symbol_select(mt5_sym, True)
            rates = mt5.copy_rates_from_pos(mt5_sym, mt5_tf, 0, n_candles)

        if rates is None or len(rates) == 0:
            return None, "no_data"

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
        df = df.rename(columns={
            "open":   "Open",
            "high":   "High",
            "low":    "Low",
            "close":  "Close",
            "tick_volume": "Volume",
        })
        df = df[["Open","High","Low","Close","Volume"]].copy()

        # Get live tick (current price - most accurate)
        tick = mt5.symbol_info_tick(mt5_sym)
        if tick:
            # Update the last candle close with live bid
            df.iloc[-1, df.columns.get_loc("Close")] = tick.bid

        print(f"  ✓ MT5 {symbol} {tf}: {len(df)} candles, last={df['Close'].iloc[-1]:.{inst['digits']}f}")
        return df, "live (XM direct)"

    except Exception as e:
        print(f"  ✗ MT5 fetch error for {symbol}: {e}")
        return None, f"mt5_error: {e}"


# ── FETCH FROM YFINANCE (fallback) ────────────────────────────────────────────
def fetch_yfinance(symbol: str, tf: str = "1h") -> tuple:
    if not YF_OK:
        return None, "no_yfinance"

    inst = INSTRUMENTS[symbol]
    interval_map = {
        "1m":  ("1d",  "1m"),
        "5m":  ("5d",  "5m"),
        "15m": ("5d",  "15m"),
        "1h":  ("30d", "1h"),
        "4h":  ("60d", "1h"),
        "1d":  ("1y",  "1d"),
    }
    period, interval = interval_map.get(tf, ("30d","1h"))

    try:
        df = yf.download(inst["yf_ticker"], period=period,
                         interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        if tf == "4h":
            df = df.resample("4h").agg({"Open":"first","High":"max",
                                        "Low":"min","Close":"last","Volume":"sum"}).dropna()
        if len(df) < 10:
            return None, "empty"
        return df, "delayed (Yahoo Finance, ~15min)"
    except Exception as e:
        return None, f"yf_error: {e}"


# ── SYNTHETIC FALLBACK ────────────────────────────────────────────────────────
def generate_synthetic(symbol: str, n: int = 300) -> pd.DataFrame:
    cfg = INSTRUMENTS[symbol]
    bases = {"GOLD":4820,"EURUSD":1.1794,"GBPUSD":1.3567,
             "USDJPY":158.92,"GBPJPY":215.48,"EURJPY":187.37}
    p = bases.get(symbol, 1.0)
    rng = np.random.default_rng(seed=abs(hash(symbol)) % (2**31))
    rows, times = [], []
    now = datetime.utcnow()
    for i in range(n):
        trend = float(rng.choice([-3e-4, 0, 3e-4]))
        vol   = p * float(rng.uniform(0.0008, 0.002))
        o=p; c=o*(1+float(rng.normal(trend,vol/p)))
        hi=max(o,c)*(1+abs(float(rng.normal(0,3e-4))))
        lo=min(o,c)*(1-abs(float(rng.normal(0,3e-4))))
        rows.append({"Open":round(o,cfg["digits"]),"High":round(hi,cfg["digits"]),
                     "Low":round(lo,cfg["digits"]),"Close":round(c,cfg["digits"]),"Volume":1000})
        times.append(now - timedelta(hours=n-i))
        p = c
    return pd.DataFrame(rows, index=pd.DatetimeIndex(times))


def get_data(symbol: str, tf: str = "1h", n_candles: int = 500) -> tuple:
    """Smart data fetcher with caching"""
    key = f"{symbol}_{tf}"
    now = time.time()
    ttl = CACHE_TTL_MT5 if MT5_OK else CACHE_TTL_YF

    if key in _cache and (now - _cache[key]["ts"]) < ttl:
        return _cache[key]["df"], _cache[key]["src"]

    # Try MT5 first (XM direct)
    df, src = fetch_mt5(symbol, tf, n_candles)

    # Fall back to Yahoo Finance
    if df is None:
        df, src = fetch_yfinance(symbol, tf)

    # Final fallback to synthetic
    if df is None:
        print(f"  ! Using synthetic data for {symbol} — install MT5 or yfinance")
        df  = generate_synthetic(symbol, 300)
        src = "synthetic"

    _cache[key] = {"df": df, "ts": now, "src": src}
    return df, src


# ── GET LIVE ACCOUNT INFO FROM MT5 ────────────────────────────────────────────
def get_account_info() -> dict:
    if not MT5_OK:
        return {}
    try:
        acc = mt5.account_info()
        if acc:
            return {
                "balance":   round(acc.balance, 2),
                "equity":    round(acc.equity, 2),
                "margin":    round(acc.margin, 2),
                "free_margin": round(acc.margin_free, 2),
                "profit":    round(acc.profit, 2),
                "leverage":  acc.leverage,
                "currency":  acc.currency,
                "broker":    acc.company,
                "login":     acc.login,
            }
    except Exception:
        pass
    return {}


# ── GET LIVE POSITIONS FROM MT5 ───────────────────────────────────────────────
def get_open_positions() -> list:
    if not MT5_OK:
        return []
    try:
        positions = mt5.positions_get()
        if positions:
            result = []
            for p in positions:
                inst = next((v for v in INSTRUMENTS.values() if v["mt5_sym"]==p.symbol), None)
                result.append({
                    "ticket":   p.ticket,
                    "symbol":   p.symbol,
                    "type":     "BUY" if p.type == 0 else "SELL",
                    "volume":   p.volume,
                    "price_open": round(p.price_open, inst["digits"] if inst else 5),
                    "sl":       round(p.sl, inst["digits"] if inst else 5),
                    "tp":       round(p.tp, inst["digits"] if inst else 5),
                    "profit":   round(p.profit, 2),
                    "pips":     round((p.price_current - p.price_open) / (inst["pip"] if inst else 0.0001) *
                                     (1 if p.type==0 else -1), 1),
                    "price_current": round(p.price_current, inst["digits"] if inst else 5),
                })
            return result
    except Exception:
        pass
    return []


# ── INDICATORS ────────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, hi, lo = df["Close"], df["High"], df["Low"]
    d = df.copy()
    d["ma9"]  = c.rolling(9).mean()
    d["ma20"] = c.rolling(20).mean()
    d["ma50"] = c.rolling(50).mean()
    delta = c.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - (100 / (1 + g / l.replace(0, np.nan)))
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    d["macd"]      = e12 - e26
    d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]
    tr = pd.concat([hi-lo,(hi-c.shift()).abs(),(lo-c.shift()).abs()],axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["bb_mid"] = c.rolling(20).mean()
    std = c.rolling(20).std()
    d["bb_up"] = d["bb_mid"] + 2*std
    d["bb_lo"] = d["bb_mid"] - 2*std
    d["stoch"] = (c - lo.rolling(14).min()) * 100 / (
        hi.rolling(14).max() - lo.rolling(14).min() + 1e-10)
    d["momentum"] = c - c.shift(10)
    return d.dropna()


# ── SIGNAL ENGINE ─────────────────────────────────────────────────────────────
def signal_at(i: int, df: pd.DataFrame):
    if i < 2: return None
    cur, prv = df.iloc[i], df.iloc[i-1]
    bull = bear = 0
    reasons = []

    if cur["ma20"] > cur["ma50"]:   bull+=2; reasons.append(("bull","MA20 above MA50 — uptrend"))
    else:                            bear+=2; reasons.append(("bear","MA20 below MA50 — downtrend"))
    if cur["Close"]>cur["ma20"]>cur["ma50"]: bull+=1; reasons.append(("bull","Price above both MAs"))
    elif cur["Close"]<cur["ma20"]<cur["ma50"]: bear+=1; reasons.append(("bear","Price below both MAs"))
    if cur["rsi"]<35:  bull+=2; reasons.append(("bull",f"RSI {cur['rsi']:.1f} — oversold"))
    elif cur["rsi"]>65: bear+=2; reasons.append(("bear",f"RSI {cur['rsi']:.1f} — overbought"))
    if cur["macd_hist"]>0 and prv["macd_hist"]<=0:   bull+=3; reasons.append(("bull","MACD crossed UP"))
    elif cur["macd_hist"]<0 and prv["macd_hist"]>=0: bear+=3; reasons.append(("bear","MACD crossed DOWN"))
    elif cur["macd_hist"]>0: bull+=1; reasons.append(("bull","MACD positive"))
    else:                    bear+=1; reasons.append(("bear","MACD negative"))
    if cur["ma9"]>cur["ma20"] and prv["ma9"]<=prv["ma20"]:  bull+=2; reasons.append(("bull","MA9 crossed above MA20"))
    elif cur["ma9"]<cur["ma20"] and prv["ma9"]>=prv["ma20"]: bear+=2; reasons.append(("bear","MA9 crossed below MA20"))
    if cur["Close"]<=cur["bb_lo"]:   bull+=1; reasons.append(("bull","Price at lower Bollinger Band"))
    elif cur["Close"]>=cur["bb_up"]: bear+=1; reasons.append(("bear","Price at upper Bollinger Band"))
    if cur["stoch"]<25 and cur["stoch"]>prv["stoch"]:   bull+=1; reasons.append(("bull","Stochastic oversold + rising"))
    elif cur["stoch"]>75 and cur["stoch"]<prv["stoch"]: bear+=1; reasons.append(("bear","Stochastic overbought + falling"))
    if cur["momentum"]>0 and cur["ma20"]>cur["ma50"]:   bull+=1; reasons.append(("bull","Positive momentum in uptrend"))
    elif cur["momentum"]<0 and cur["ma20"]<cur["ma50"]: bear+=1; reasons.append(("bear","Negative momentum in downtrend"))

    total = bull + bear
    if total == 0: return None
    if bull>=6 and bull>bear:
        return {"dir":"BUY", "conf":min(95,round(bull/total*100)),"bull":bull,"bear":bear,
                "reasons":[r[1] for r in reasons if r[0]=="bull"]}
    if bear>=6 and bear>bull:
        return {"dir":"SELL","conf":min(95,round(bear/total*100)),"bull":bull,"bear":bear,
                "reasons":[r[1] for r in reasons if r[0]=="bear"]}
    return {"dir":"WAIT","conf":0,"bull":bull,"bear":bear,"reasons":[r[1] for r in reasons]}


# ── BACKTEST ──────────────────────────────────────────────────────────────────
def backtest(df, sym, sl_m=1.5, tp_m=2.5, bal=10000, risk=2, min_conf=60):
    inst=INSTRUMENTS[sym]; trades=[]; equity=[bal]; balance=bal
    in_trade=False; trade=None
    for i in range(60, len(df)):
        row=df.iloc[i]
        if in_trade:
            t=trade; hit_sl=hit_tp=False
            if t["dir"]=="BUY":
                if row["Low"]<=t["sl"]: hit_sl=True
                if row["High"]>=t["tp"]: hit_tp=True
            else:
                if row["High"]>=t["sl"]: hit_sl=True
                if row["Low"]<=t["tp"]: hit_tp=True
            if hit_tp:   pnl=t["risk"]*tp_m/sl_m; outcome="WIN"
            elif hit_sl: pnl=-t["risk"];           outcome="LOSS"
            else:
                t["bars"]+=1
                if t["bars"]>=20:
                    diff=float(row["Close"])-t["entry"]
                    pnl=diff/inst["pip"]*inst["pip_val"]*t["lots"]
                    if t["dir"]=="SELL": pnl=-pnl
                    outcome="WIN" if pnl>0 else "LOSS"
                else: equity.append(balance); continue
            balance+=pnl
            trade.update({"exit":round(float(row["Close"]),inst["digits"]),"outcome":outcome,
                          "pnl":round(pnl,2),"exit_dt":df.index[i].strftime("%Y-%m-%d %H:%M")})
            trades.append(trade); equity.append(balance); in_trade=False; trade=None; continue
        sig=signal_at(i,df)
        if not sig or sig["conf"]<min_conf or sig["dir"]=="WAIT": equity.append(balance); continue
        atr=float(row["atr"]); sl_d=atr*sl_m; tp_d=atr*tp_m
        entry=round(float(row["Close"]),inst["digits"])
        sl_pips=sl_d/inst["pip"]; risk_amt=balance*(risk/100)
        lots=max(0.01,round(risk_amt/(sl_pips*inst["pip_val"]),2))
        sl=round(entry-sl_d,inst["digits"]) if sig["dir"]=="BUY" else round(entry+sl_d,inst["digits"])
        tp=round(entry+tp_d,inst["digits"]) if sig["dir"]=="BUY" else round(entry-tp_d,inst["digits"])
        trade={"dir":sig["dir"],"conf":sig["conf"],"entry":entry,"sl":sl,"tp":tp,
               "sl_pips":round(sl_pips),"tp_pips":round(tp_d/inst["pip"]),
               "risk":round(risk_amt,2),"lots":lots,"bars":0,
               "entry_dt":df.index[i].strftime("%Y-%m-%d %H:%M"),"sym":sym}
        in_trade=True; equity.append(balance)
    wins=[t for t in trades if t["outcome"]=="WIN"]
    losses=[t for t in trades if t["outcome"]=="LOSS"]
    closed=wins+losses; wr=round(len(wins)/len(closed)*100,1) if closed else 0
    gp=sum(t["pnl"] for t in wins); gl=abs(sum(t["pnl"] for t in losses))
    pf=round(gp/gl,2) if gl>0 else 0
    peak=equity[0]; max_dd=0.0
    for eq in equity:
        if eq>peak: peak=eq
        dd=(peak-eq)/peak*100
        if dd>max_dd: max_dd=dd
    pnls=[t["pnl"] for t in trades]
    sharpe=round(np.mean(pnls)/np.std(pnls)*np.sqrt(len(pnls)),2) if len(pnls)>1 and np.std(pnls)>0 else 0
    avg_w=round(np.mean([t["pnl"] for t in wins]),2) if wins else 0
    avg_l=round(np.mean([t["pnl"] for t in losses]),2) if losses else 0
    return {"trades":trades,"equity":equity,"win_rate":wr,"profit_factor":pf,
            "max_dd":round(max_dd,2),"total":len(trades),"wins":len(wins),"losses":len(losses),
            "net_pnl":round(balance-bal,2),"sharpe":sharpe,"final_bal":round(balance,2),
            "avg_win":avg_w,"avg_loss":avg_l,"expectancy":round(wr/100*avg_w+(1-wr/100)*avg_l,2)}

def grade(r):
    s=sum([r["win_rate"]>=55,r["profit_factor"]>=1.5,r["max_dd"]<=15,r["net_pnl"]>0,r["sharpe"]>1])
    return{5:"A+",4:"A",3:"B",2:"C",1:"D",0:"F"}[s]

def load_json(n,d): p=os.path.join(DATA_DIR,n); return json.load(open(p)) if os.path.exists(p) else d
def save_json(n,d): json.dump(d,open(os.path.join(DATA_DIR,n),"w"),indent=2)


# ═══════════════════════════════════════ ROUTES ═══════════════════════════════

@app.route("/")
def index(): return send_from_directory(FRONTEND,"index.html")

@app.route("/<path:path>")
def static_files(path): return send_from_directory(FRONTEND,path)

@app.route("/api/health")
def health():
    acc = get_account_info()
    return jsonify({
        "status":    "ok",
        "mt5_live":  MT5_OK,
        "yf_ok":     YF_OK,
        "account":   acc,
        "data_source": "XM (MT5 direct)" if MT5_OK else "Yahoo Finance (delayed)" if YF_OK else "synthetic",
        "ts":        datetime.utcnow().isoformat()
    })

@app.route("/api/account")
def account():
    """Live account info from XM"""
    return jsonify(get_account_info())

@app.route("/api/positions")
def live_positions():
    """Live open positions from XM"""
    return jsonify(get_open_positions())

@app.route("/api/quote/<sym>")
def quote(sym):
    if sym not in INSTRUMENTS: return jsonify({"error":"unknown"}),400
    tf  = request.args.get("tf","1h")
    n_c = int(request.args.get("candles","300"))

    df, src = get_data(sym, tf, n_c)
    inst    = INSTRUMENTS[sym]
    df2     = add_indicators(df)
    last    = df2.iloc[-1]
    prev    = df2.iloc[-2]
    sig     = signal_at(len(df2)-1, df2)
    atr     = float(last["atr"])
    d       = inst["digits"]

    # Get live bid/ask spread from MT5
    bid = ask = spread_val = None
    if MT5_OK:
        tick = mt5.symbol_info_tick(inst["mt5_sym"])
        if tick:
            bid        = round(tick.bid, d)
            ask        = round(tick.ask, d)
            spread_val = round((ask - bid) / inst["pip"], 1)

    entry=sl=tp1=tp2=sl_pips=tp_pips=rr=None
    if sig and sig["dir"] not in ("WAIT",None):
        entry = round(float(last["Close"]),d)
        if sig["dir"]=="BUY":
            sl=round(entry-atr*1.5,d); tp1=round(entry+atr*2.5,d); tp2=round(entry+atr*4,d)
        else:
            sl=round(entry+atr*1.5,d); tp1=round(entry-atr*2.5,d); tp2=round(entry-atr*4,d)
        sl_pips=round(abs(entry-sl)/inst["pip"])
        tp_pips=round(abs(tp1-entry)/inst["pip"])
        rr=round(tp_pips/sl_pips,1) if sl_pips else 0

    tail = df2.tail(n_c)
    candles = []
    for idx, r in tail.iterrows():
        candles.append({
            "t":    idx.strftime("%Y-%m-%d %H:%M"),
            "o":    round(float(r["Open"]),d),
            "h":    round(float(r["High"]),d),
            "l":    round(float(r["Low"]),d),
            "c":    round(float(r["Close"]),d),
            "v":    int(r["Volume"]) if not np.isnan(r["Volume"]) else 0,
            "ma9":  round(float(r["ma9"]),d)  if not np.isnan(r["ma9"])  else None,
            "ma20": round(float(r["ma20"]),d) if not np.isnan(r["ma20"]) else None,
            "ma50": round(float(r["ma50"]),d) if not np.isnan(r["ma50"]) else None,
            "rsi":  round(float(r["rsi"]),1)  if not np.isnan(r["rsi"])  else None,
            "macd_hist": round(float(r["macd_hist"]),6) if not np.isnan(r["macd_hist"]) else None,
            "bb_up":round(float(r["bb_up"]),d) if not np.isnan(r["bb_up"]) else None,
            "bb_lo":round(float(r["bb_lo"]),d) if not np.isnan(r["bb_lo"]) else None,
            "stoch":round(float(r["stoch"]),1) if not np.isnan(r["stoch"]) else None,
        })

    chg = round((float(last["Close"])-float(prev["Close"]))/float(prev["Close"])*100,3)

    return jsonify({
        "sym":sym,"label":inst["label"],"digits":d,
        "price": bid or round(float(last["Close"]),d),  # Use live bid if available
        "ask":   ask,
        "bid":   bid,
        "open":  round(float(last["Open"]),d),
        "high":  round(float(last["High"]),d),
        "low":   round(float(last["Low"]),d),
        "change":chg,
        "rsi":   round(float(last["rsi"]),1),
        "atr":   round(atr,d),
        "stoch": round(float(last["stoch"]),1),
        "signal":sig,"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,
        "sl_pips":sl_pips,"tp_pips":tp_pips,"rr":rr,
        "ma20":  round(float(last["ma20"]),d),
        "ma50":  round(float(last["ma50"]),d),
        "bb_up": round(float(last["bb_up"]),d),
        "bb_lo": round(float(last["bb_lo"]),d),
        "spread":spread_val or inst["spread_pips"],
        "candles":candles,"src":src,
        "ts":datetime.utcnow().strftime("%H:%M:%S UTC"),
    })


@app.route("/api/backtest/<sym>")
def do_backtest(sym):
    if sym not in INSTRUMENTS: return jsonify({"error":"unknown"}),400
    tf=request.args.get("tf","1h")
    bal=float(request.args.get("balance","10000"))
    risk=float(request.args.get("risk","2"))
    sl_m=float(request.args.get("sl","1.5"))
    tp_m=float(request.args.get("tp","2.5"))
    min_conf=int(request.args.get("conf","60"))
    df,src=get_data(sym,tf); df2=add_indicators(df)
    r=backtest(df2,sym,sl_m,tp_m,bal,risk,min_conf)
    r.update({"src":src,"sym":sym,"label":INSTRUMENTS[sym]["label"],"grade":grade(r),
               "date_range":f"{df2.index[0].strftime('%d %b %Y')} → {df2.index[-1].strftime('%d %b %Y')}"})
    return jsonify(r)


@app.route("/api/sweep")
def sweep():
    tf=request.args.get("tf","1h"); bal=float(request.args.get("balance","10000"))
    risk=float(request.args.get("risk","2")); results=[]
    for sym in INSTRUMENTS:
        df,src=get_data(sym,tf); df2=add_indicators(df)
        r=backtest(df2,sym,1.5,2.5,bal,risk,60)
        sig=signal_at(len(df2)-1,df2)
        results.append({"sym":sym,"label":INSTRUMENTS[sym]["label"],"tf":tf,
                        "win_rate":r["win_rate"],"profit_factor":r["profit_factor"],
                        "max_dd":r["max_dd"],"net_pnl":r["net_pnl"],"sharpe":r["sharpe"],
                        "total":r["total"],"grade":grade(r),
                        "score":sum([r["win_rate"]>=55,r["profit_factor"]>=1.5,
                                     r["max_dd"]<=15,r["net_pnl"]>0,r["sharpe"]>1]),
                        "signal":sig["dir"] if sig else "WAIT",
                        "conf":sig["conf"] if sig else 0,"src":src})
    results.sort(key=lambda x:(x["score"],x["profit_factor"]),reverse=True)
    return jsonify(results)


@app.route("/api/scan")
def scan():
    tf=request.args.get("tf","1h"); results=[]
    for sym in INSTRUMENTS:
        df,src=get_data(sym,tf); df2=add_indicators(df)
        last=df2.iloc[-1]; prev=df2.iloc[-2]
        sig=signal_at(len(df2)-1,df2)
        inst=INSTRUMENTS[sym]; dg=inst["digits"]
        price=round(float(last["Close"]),dg)
        # Use live price from MT5 if available
        if MT5_OK:
            tick=mt5.symbol_info_tick(inst["mt5_sym"])
            if tick: price=round(tick.bid,dg)
        chg=round((price-float(prev["Close"]))/float(prev["Close"])*100,3)
        atr=float(last["atr"]); entry=sl=tp1=tp2=sl_pips=rr=None
        if sig and sig["dir"] not in ("WAIT",None):
            entry=price
            if sig["dir"]=="BUY": sl=round(entry-atr*1.5,dg);tp1=round(entry+atr*2.5,dg);tp2=round(entry+atr*4,dg)
            else:                  sl=round(entry+atr*1.5,dg);tp1=round(entry-atr*2.5,dg);tp2=round(entry-atr*4,dg)
            sl_pips=round(abs(entry-sl)/inst["pip"])
            rr=round(abs(tp1-entry)/abs(entry-sl),1) if sl else 0
        results.append({"sym":sym,"label":inst["label"],"price":price,"change":chg,
                        "rsi":round(float(last["rsi"]),1),
                        "signal":sig["dir"] if sig else "WAIT",
                        "conf":sig["conf"] if sig else 0,
                        "entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,
                        "sl_pips":sl_pips,"rr":rr,"src":src})
    results.sort(key=lambda x:x["conf"],reverse=True)
    return jsonify(results)


@app.route("/api/alerts",methods=["GET","POST","DELETE"])
def alerts():
    data=load_json("alerts.json",[])
    if request.method=="GET":
        triggered=[]; remaining=[]
        for a in data:
            sym=a.get("sym"); level=float(a.get("level",0)); above=a.get("above",True)
            if sym in INSTRUMENTS:
                df,_=get_data(sym,"1h"); price=float(df["Close"].iloc[-1])
                if MT5_OK:
                    tick=mt5.symbol_info_tick(INSTRUMENTS[sym]["mt5_sym"])
                    if tick: price=tick.bid
                if (above and price>=level) or (not above and price<=level):
                    triggered.append({**a,"price":price}); continue
            remaining.append(a)
        if triggered: save_json("alerts.json",remaining)
        return jsonify({"alerts":remaining,"triggered":triggered})
    if request.method=="POST":
        a=request.json; a["id"]=int(time.time()*1000)
        a["created"]=datetime.utcnow().isoformat(); data.append(a)
        save_json("alerts.json",data); return jsonify(a)
    if request.method=="DELETE":
        pid=request.args.get("id"); data=[a for a in data if str(a.get("id"))!=str(pid)]
        save_json("alerts.json",data); return jsonify({"ok":True})


@app.route("/api/journal",methods=["GET","POST","PUT","DELETE"])
def journal():
    data=load_json("journal.json",[])
    if request.method=="GET":
        closed=[t for t in data if t.get("status")=="closed"]
        wins=[t for t in closed if (t.get("pnl") or 0)>0]
        losses=[t for t in closed if (t.get("pnl") or 0)<=0]
        return jsonify({"trades":data,"stats":{"total":len(data),"closed":len(closed),
            "open":len([t for t in data if t.get("status")=="open"]),"wins":len(wins),
            "losses":len(losses),"win_rate":round(len(wins)/len(closed)*100,1) if closed else 0,
            "total_pnl":round(sum(t.get("pnl",0) for t in closed),2)}})
    if request.method=="POST":
        t=request.json; t["id"]=int(time.time()*1000)
        t["opened"]=datetime.utcnow().isoformat(); t["status"]="open"; t["pnl"]=None
        data.append(t); save_json("journal.json",data); return jsonify(t)
    if request.method=="PUT":
        tid=request.json.get("id"); exit_px=float(request.json.get("exit_price",0))
        updated=[]
        for t in data:
            if str(t.get("id"))==str(tid):
                inst=INSTRUMENTS.get(t["sym"],{}); pip=inst.get("pip",1e-4)
                pipval=inst.get("pip_val",10); d_=inst.get("digits",5)
                lots=float(t.get("lots",0.01)); entry=float(t.get("entry",0))
                diff=exit_px-entry if t["dir"]=="BUY" else entry-exit_px
                pnl=round(diff/pip*pipval*lots,2)
                t.update({"status":"closed","exit_price":round(exit_px,d_),
                          "closed":datetime.utcnow().isoformat(),"pnl":pnl})
            updated.append(t)
        save_json("journal.json",updated); return jsonify({"ok":True})
    if request.method=="DELETE":
        tid=request.args.get("id"); data=[t for t in data if str(t.get("id"))!=str(tid)]
        save_json("journal.json",data); return jsonify({"ok":True})


@app.route("/api/stats")
def stats():
    j=load_json("journal.json",[]); closed=[t for t in j if t.get("status")=="closed"]
    wins=[t for t in closed if (t.get("pnl") or 0)>0]
    losses=[t for t in closed if (t.get("pnl") or 0)<=0]
    pnls=[t.get("pnl",0) for t in closed]
    return jsonify({"total_trades":len(closed),"wins":len(wins),"losses":len(losses),
                    "win_rate":round(len(wins)/len(closed)*100,1) if closed else 0,
                    "total_pnl":round(sum(pnls),2),
                    "best_trade":max(pnls) if pnls else 0,"worst_trade":min(pnls) if pnls else 0})


if __name__=="__main__":
    src = "XM (MetaTrader5 direct)" if MT5_OK else "Yahoo Finance (delayed)" if YF_OK else "synthetic only"
    print("\n"+"="*55)
    print("  XM360 Trade Dashboard — MT5 Live Edition")
    print(f"  Data: {src}")
    print("  URL:  http://localhost:5000")
    print("="*55+"\n")
    app.run(debug=False, port=5000, host="0.0.0.0", threaded=True)
