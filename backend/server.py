"""
XM360 Trade Dashboard - Backend Server v4
==========================================
REAL LIVE DATA from Yahoo Finance (yfinance)
Updates every 30 seconds - real candles, real prices

Run: python backend/server.py
Open: http://localhost:5000
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json, os, time, threading
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Try importing data libraries ──────────────────────────────────────────────
try:
    import yfinance as yf
    HAS_YF = True
    print("  ✓ yfinance loaded - LIVE data enabled")
except ImportError:
    HAS_YF = False
    print("  ✗ yfinance not found - run: pip install yfinance")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(BASE_DIR, "frontend")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND)
CORS(app)

# ── Instrument Config ─────────────────────────────────────────────────────────
INSTRUMENTS = {
    "GOLD":   {"ticker":"GC=F",     "label":"GOLD.i#",  "pip":0.1,    "digits":2, "pip_val":1.0,  "spread":0.35},
    "EURUSD": {"ticker":"EURUSD=X", "label":"EURUSD#",  "pip":0.0001, "digits":5, "pip_val":10.0, "spread":0.00012},
    "GBPUSD": {"ticker":"GBPUSD=X", "label":"GBPUSD#",  "pip":0.0001, "digits":5, "pip_val":10.0, "spread":0.00014},
    "USDJPY": {"ticker":"USDJPY=X", "label":"USDJPY#",  "pip":0.01,   "digits":3, "pip_val":9.0,  "spread":0.013},
    "GBPJPY": {"ticker":"GBPJPY=X", "label":"GBPJPY#",  "pip":0.01,   "digits":3, "pip_val":9.0,  "spread":0.022},
    "EURJPY": {"ticker":"EURJPY=X", "label":"EURJPY#",  "pip":0.01,   "digits":3, "pip_val":9.0,  "spread":0.018},
}

# ── Cache - stores real fetched data ─────────────────────────────────────────
_cache = {}
CACHE_TTL = 30   # seconds - refresh every 30s for near-real-time

# ── Fetch REAL data from Yahoo Finance ────────────────────────────────────────
def fetch_live(symbol: str, tf: str = "1h", period: str = "7d") -> tuple:
    """
    Fetch real OHLCV candle data from Yahoo Finance.
    Returns (DataFrame, source_string)
    """
    if not HAS_YF:
        return None, "no_yfinance"

    inst = INSTRUMENTS[symbol]
    ticker = inst["ticker"]

    # Map timeframe to yfinance interval
    interval_map = {
        "1m":  ("1d",   "1m"),   # 1 minute - last 1 day
        "5m":  ("5d",   "5m"),   # 5 minute - last 5 days
        "15m": ("5d",   "15m"),  # 15 minute - last 5 days
        "1h":  ("30d",  "1h"),   # 1 hour - last 30 days
        "4h":  ("60d",  "1h"),   # 4 hour (resampled from 1h) - last 60 days
        "1d":  ("1y",   "1d"),   # daily - last 1 year
    }
    period_str, interval = interval_map.get(tf, ("30d", "1h"))

    try:
        df = yf.download(
            ticker,
            period=period_str,
            interval=interval,
            progress=False,
            auto_adjust=True,
            prepost=False,
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open","High","Low","Close","Volume"]].dropna()

        if df.empty or len(df) < 10:
            return None, "empty"

        # Resample 4H from 1H data
        if tf == "4h":
            df = df.resample("4h").agg({
                "Open":"first","High":"max","Low":"min",
                "Close":"last","Volume":"sum"
            }).dropna()

        print(f"  ✓ {symbol} {tf}: {len(df)} candles, last={df['Close'].iloc[-1]:.{inst['digits']}f}")
        return df, "live"

    except Exception as e:
        print(f"  ✗ {symbol} fetch error: {e}")
        return None, f"error: {e}"


def generate_synthetic(symbol: str, n: int = 500) -> pd.DataFrame:
    """Fallback synthetic data - only used if live fetch fails"""
    cfg = INSTRUMENTS[symbol]
    bases = {"GOLD":4820,"EURUSD":1.1794,"GBPUSD":1.3567,
              "USDJPY":158.92,"GBPJPY":215.48,"EURJPY":187.37}
    p = bases.get(symbol, 1.0)
    rng = np.random.default_rng(seed=abs(hash(symbol+str(int(time.time()/3600)))) % (2**31))
    rows, times = [], []
    now = datetime.utcnow()
    trend = 0.0; vol = p * 0.0015
    for i in range(n):
        if i % 30 == 0:
            trend = float(rng.choice([-5e-4,-2e-4,0,2e-4,5e-4]))
            vol = p * float(rng.uniform(0.0008, 0.002))
        o = p; c = o*(1+float(rng.normal(trend, vol/p)))
        hi = max(o,c)*(1+abs(float(rng.normal(0,3e-4))))
        lo = min(o,c)*(1-abs(float(rng.normal(0,3e-4))))
        rows.append({"Open":round(o,cfg["digits"]),"High":round(hi,cfg["digits"]),
                     "Low":round(lo,cfg["digits"]),"Close":round(c,cfg["digits"]),"Volume":1000})
        times.append(now - timedelta(hours=n-i))
        p = c
    return pd.DataFrame(rows, index=pd.DatetimeIndex(times))


def get_data(symbol: str, tf: str = "1h") -> tuple:
    """Get data with caching"""
    key = f"{symbol}_{tf}"
    now = time.time()

    if key in _cache and (now - _cache[key]["ts"]) < CACHE_TTL:
        return _cache[key]["df"], _cache[key]["src"]

    # Try live data first
    df, src = fetch_live(symbol, tf)

    if df is None or len(df) < 20:
        print(f"  ! Using synthetic data for {symbol}")
        df = generate_synthetic(symbol, 300)
        src = "synthetic (live fetch failed)"

    _cache[key] = {"df": df, "ts": now, "src": src}
    return df, src


# ── Technical Indicators ──────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, hi, lo = df["Close"], df["High"], df["Low"]
    d = df.copy()

    d["ma9"]  = c.rolling(9).mean()
    d["ma20"] = c.rolling(20).mean()
    d["ma50"] = c.rolling(50).mean()

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    d["macd"]      = e12 - e26
    d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]

    # ATR
    tr = pd.concat([hi-lo,(hi-c.shift()).abs(),(lo-c.shift()).abs()],axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()

    # Bollinger Bands
    d["bb_mid"] = c.rolling(20).mean()
    std = c.rolling(20).std()
    d["bb_up"] = d["bb_mid"] + 2*std
    d["bb_lo"] = d["bb_mid"] - 2*std

    # Stochastic
    d["stoch"] = (c - lo.rolling(14).min()) * 100 / (
        hi.rolling(14).max() - lo.rolling(14).min() + 1e-10)

    # Momentum
    d["momentum"] = c - c.shift(10)

    return d.dropna()


# ── Signal Engine ─────────────────────────────────────────────────────────────
def signal_at(i: int, df: pd.DataFrame):
    if i < 2: return None
    cur, prv = df.iloc[i], df.iloc[i-1]
    bull = bear = 0
    reasons = []

    # 1. MA Trend
    if cur["ma20"] > cur["ma50"]:   bull += 2; reasons.append(("bull","MA20 above MA50 — uptrend"))
    else:                            bear += 2; reasons.append(("bear","MA20 below MA50 — downtrend"))

    # 2. Price vs MAs
    if cur["Close"] > cur["ma20"] > cur["ma50"]:   bull += 1; reasons.append(("bull","Price above both MAs"))
    elif cur["Close"] < cur["ma20"] < cur["ma50"]: bear += 1; reasons.append(("bear","Price below both MAs"))

    # 3. RSI
    if cur["rsi"] < 35:   bull += 2; reasons.append(("bull",f"RSI {cur['rsi']:.1f} — oversold"))
    elif cur["rsi"] > 65: bear += 2; reasons.append(("bear",f"RSI {cur['rsi']:.1f} — overbought"))

    # 4. MACD
    if cur["macd_hist"] > 0 and prv["macd_hist"] <= 0:   bull += 3; reasons.append(("bull","MACD crossed UP — bullish"))
    elif cur["macd_hist"] < 0 and prv["macd_hist"] >= 0: bear += 3; reasons.append(("bear","MACD crossed DOWN — bearish"))
    elif cur["macd_hist"] > 0: bull += 1; reasons.append(("bull","MACD histogram positive"))
    else:                       bear += 1; reasons.append(("bear","MACD histogram negative"))

    # 5. MA9 crossover
    if cur["ma9"] > cur["ma20"] and prv["ma9"] <= prv["ma20"]:  bull += 2; reasons.append(("bull","MA9 crossed above MA20"))
    elif cur["ma9"] < cur["ma20"] and prv["ma9"] >= prv["ma20"]: bear += 2; reasons.append(("bear","MA9 crossed below MA20"))

    # 6. Bollinger Band
    if cur["Close"] <= cur["bb_lo"]:   bull += 1; reasons.append(("bull","Price at lower BB — bounce likely"))
    elif cur["Close"] >= cur["bb_up"]: bear += 1; reasons.append(("bear","Price at upper BB — reversal likely"))

    # 7. Stochastic
    if cur["stoch"] < 25 and cur["stoch"] > prv["stoch"]:   bull += 1; reasons.append(("bull","Stochastic oversold + rising"))
    elif cur["stoch"] > 75 and cur["stoch"] < prv["stoch"]: bear += 1; reasons.append(("bear","Stochastic overbought + falling"))

    # 8. Momentum
    if cur["momentum"] > 0 and cur["ma20"] > cur["ma50"]:   bull += 1; reasons.append(("bull","Positive momentum in uptrend"))
    elif cur["momentum"] < 0 and cur["ma20"] < cur["ma50"]: bear += 1; reasons.append(("bear","Negative momentum in downtrend"))

    total = bull + bear
    if total == 0: return None

    if bull >= 6 and bull > bear:
        return {"dir":"BUY",  "conf":min(95,round(bull/total*100)), "bull":bull,"bear":bear,
                "reasons":[r[1] for r in reasons if r[0]=="bull"]}
    if bear >= 6 and bear > bull:
        return {"dir":"SELL", "conf":min(95,round(bear/total*100)), "bull":bull,"bear":bear,
                "reasons":[r[1] for r in reasons if r[0]=="bear"]}

    return {"dir":"WAIT","conf":0,"bull":bull,"bear":bear,
            "reasons":[r[1] for r in reasons]}


# ── Backtest ──────────────────────────────────────────────────────────────────
def backtest(df, sym, sl_m=1.5, tp_m=2.5, bal=10000, risk=2, min_conf=60):
    inst = INSTRUMENTS[sym]
    trades=[];equity=[bal];balance=bal;in_trade=False;trade=None

    for i in range(60, len(df)):
        row = df.iloc[i]
        if in_trade:
            t=trade;hit_sl=hit_tp=False
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
                          "pnl":round(pnl,2),"bal":round(balance,2),
                          "exit_dt":df.index[i].strftime("%Y-%m-%d %H:%M")})
            trades.append(trade);equity.append(balance);in_trade=False;trade=None;continue

        sig=signal_at(i,df)
        if not sig or sig["conf"]<min_conf or sig["dir"]=="WAIT": equity.append(balance); continue

        atr=float(row["atr"]);sl_d=atr*sl_m;tp_d=atr*tp_m
        entry=round(float(row["Close"]),inst["digits"])
        sl_pips=sl_d/inst["pip"];risk_amt=balance*(risk/100)
        lots=max(0.01,round(risk_amt/(sl_pips*inst["pip_val"]),2))
        sl=round(entry-sl_d,inst["digits"]) if sig["dir"]=="BUY" else round(entry+sl_d,inst["digits"])
        tp=round(entry+tp_d,inst["digits"]) if sig["dir"]=="BUY" else round(entry-tp_d,inst["digits"])
        trade={"dir":sig["dir"],"conf":sig["conf"],"entry":entry,"sl":sl,"tp":tp,
               "sl_pips":round(sl_pips),"tp_pips":round(tp_d/inst["pip"]),
               "risk":round(risk_amt,2),"lots":lots,"bars":0,
               "entry_dt":df.index[i].strftime("%Y-%m-%d %H:%M"),"sym":sym}
        in_trade=True;equity.append(balance)

    wins=[t for t in trades if t["outcome"]=="WIN"]
    losses=[t for t in trades if t["outcome"]=="LOSS"]
    closed=wins+losses;wr=round(len(wins)/len(closed)*100,1) if closed else 0
    gp=sum(t["pnl"] for t in wins);gl=abs(sum(t["pnl"] for t in losses))
    pf=round(gp/gl,2) if gl>0 else 0
    peak=equity[0];max_dd=0.0
    for eq in equity:
        if eq>peak: peak=eq
        dd=(peak-eq)/peak*100
        if dd>max_dd: max_dd=dd
    pnls=[t["pnl"] for t in trades]
    sharpe=round(np.mean(pnls)/np.std(pnls)*np.sqrt(len(pnls)),2) if len(pnls)>1 and np.std(pnls)>0 else 0
    avg_win=round(np.mean([t["pnl"] for t in wins]),2) if wins else 0
    avg_loss=round(np.mean([t["pnl"] for t in losses]),2) if losses else 0
    expect=round(wr/100*avg_win+(1-wr/100)*avg_loss,2)

    return {"trades":trades,"equity":equity,"win_rate":wr,"profit_factor":pf,
            "max_dd":round(max_dd,2),"total":len(trades),"wins":len(wins),"losses":len(losses),
            "net_pnl":round(balance-bal,2),"sharpe":sharpe,"final_bal":round(balance,2),
            "avg_win":avg_win,"avg_loss":avg_loss,"expectancy":expect}


def grade(r):
    s=sum([r["win_rate"]>=55,r["profit_factor"]>=1.5,r["max_dd"]<=15,r["net_pnl"]>0,r["sharpe"]>1])
    return{5:"A+",4:"A",3:"B",2:"C",1:"D",0:"F"}[s]

def load_json(name,default):
    p=os.path.join(DATA_DIR,name)
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return default

def save_json(name,data):
    with open(os.path.join(DATA_DIR,name),"w") as f: json.dump(data,f,indent=2)


# ═══════════════════════════════════════ ROUTES ═══════════════════════════════

@app.route("/")
def index(): return send_from_directory(FRONTEND,"index.html")

@app.route("/<path:path>")
def static_files(path): return send_from_directory(FRONTEND,path)

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","live_data":HAS_YF,"ts":datetime.utcnow().isoformat()})


@app.route("/api/quote/<sym>")
def quote(sym):
    if sym not in INSTRUMENTS: return jsonify({"error":"unknown"}),400
    tf   = request.args.get("tf","1h")
    n_c  = int(request.args.get("candles","200"))

    df, src = get_data(sym, tf)
    inst    = INSTRUMENTS[sym]
    df2     = add_indicators(df)
    last    = df2.iloc[-1]
    prev    = df2.iloc[-2]
    sig     = signal_at(len(df2)-1, df2)
    atr     = float(last["atr"])
    d       = inst["digits"]
    sl_m, tp_m = 1.5, 2.5

    entry=sl=tp1=tp2=sl_pips=tp_pips=rr=None
    if sig and sig["dir"] not in ("WAIT",None):
        entry = round(float(last["Close"]),d)
        if sig["dir"]=="BUY":
            sl=round(entry-atr*sl_m,d); tp1=round(entry+atr*tp_m,d); tp2=round(entry+atr*4,d)
        else:
            sl=round(entry+atr*sl_m,d); tp1=round(entry-atr*tp_m,d); tp2=round(entry-atr*4,d)
        sl_pips=round(abs(entry-sl)/inst["pip"])
        tp_pips=round(abs(tp1-entry)/inst["pip"])
        rr=round(tp_pips/sl_pips,1) if sl_pips else 0

    tail = df2.tail(n_c)
    candles = []
    for idx, r in tail.iterrows():
        candles.append({
            "t":   idx.strftime("%Y-%m-%d %H:%M"),
            "o":   round(float(r["Open"]),d),
            "h":   round(float(r["High"]),d),
            "l":   round(float(r["Low"]),d),
            "c":   round(float(r["Close"]),d),
            "v":   int(r["Volume"]) if not np.isnan(r["Volume"]) else 0,
            "ma9": round(float(r["ma9"]),d)  if not np.isnan(r["ma9"])  else None,
            "ma20":round(float(r["ma20"]),d) if not np.isnan(r["ma20"]) else None,
            "ma50":round(float(r["ma50"]),d) if not np.isnan(r["ma50"]) else None,
            "rsi": round(float(r["rsi"]),1)  if not np.isnan(r["rsi"])  else None,
            "macd_hist":round(float(r["macd_hist"]),6) if not np.isnan(r["macd_hist"]) else None,
            "bb_up":round(float(r["bb_up"]),d) if not np.isnan(r["bb_up"]) else None,
            "bb_lo":round(float(r["bb_lo"]),d) if not np.isnan(r["bb_lo"]) else None,
            "stoch":round(float(r["stoch"]),1) if not np.isnan(r["stoch"]) else None,
        })

    chg = round((float(last["Close"])-float(prev["Close"]))/float(prev["Close"])*100,3)

    return jsonify({
        "sym":sym,"label":inst["label"],"digits":d,
        "price":round(float(last["Close"]),d),
        "open": round(float(last["Open"]),d),
        "high": round(float(last["High"]),d),
        "low":  round(float(last["Low"]),d),
        "change":chg,"rsi":round(float(last["rsi"]),1),
        "atr":round(atr,d),"stoch":round(float(last["stoch"]),1),
        "signal":sig,"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,
        "sl_pips":sl_pips,"tp_pips":tp_pips,"rr":rr,
        "ma20":round(float(last["ma20"]),d),
        "ma50":round(float(last["ma50"]),d),
        "bb_up":round(float(last["bb_up"]),d),
        "bb_lo":round(float(last["bb_lo"]),d),
        "spread":inst["spread"],"candles":candles,"src":src,
        "ts":datetime.utcnow().strftime("%H:%M:%S UTC"),
        "candle_count":len(candles),
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
    tf=request.args.get("tf","1h")
    bal=float(request.args.get("balance","10000"))
    risk=float(request.args.get("risk","2"))
    results=[]
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
        inst=INSTRUMENTS[sym]; d=inst["digits"]
        price=round(float(last["Close"]),d)
        chg=round((price-float(prev["Close"]))/float(prev["Close"])*100,3)
        atr=float(last["atr"]); entry=sl=tp1=tp2=sl_pips=rr=None
        if sig and sig["dir"] not in ("WAIT",None):
            entry=price
            if sig["dir"]=="BUY": sl=round(entry-atr*1.5,d);tp1=round(entry+atr*2.5,d);tp2=round(entry+atr*4,d)
            else:                  sl=round(entry+atr*1.5,d);tp1=round(entry-atr*2.5,d);tp2=round(entry-atr*4,d)
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
    print("\n"+"="*55)
    print("  XM360 Trade Dashboard v4")
    print(f"  Data: {'LIVE from Yahoo Finance ✓' if HAS_YF else 'NO yfinance - install it!'}")
    print("  URL:  http://localhost:5000")
    print("  Refresh: every 30 seconds")
    print("="*55+"\n")
    if not HAS_YF:
        print("  Run first: pip install yfinance pandas numpy flask flask-cors\n")
    app.run(debug=False, port=5000, host="0.0.0.0", threaded=True)
