"""
XM360 Trade Dashboard — Backend API Server v2
=============================================
Run:  python backend/server.py
Open: http://localhost:5000

Features:
 - Candlestick OHLC data for every instrument
 - Price alerts with browser notification support
 - Full trade journal with real P&L calculation
 - Live scan ranking all instruments by signal strength
 - Session statistics
 - Backtest with grade, equity curve, full trade log
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json, os, time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(BASE_DIR, "frontend")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND)
CORS(app)

INSTRUMENTS = {
    "GOLD":   {"ticker":"GC=F",     "label":"GOLD.i#",  "pip":0.1,    "digits":2, "pip_val":1.0,  "spread":0.35},
    "EURUSD": {"ticker":"EURUSD=X", "label":"EURUSD#",  "pip":0.0001, "digits":5, "pip_val":10.0, "spread":0.00012},
    "GBPUSD": {"ticker":"GBPUSD=X", "label":"GBPUSD#",  "pip":0.0001, "digits":5, "pip_val":10.0, "spread":0.00014},
    "USDJPY": {"ticker":"USDJPY=X", "label":"USDJPY#",  "pip":0.01,   "digits":3, "pip_val":9.0,  "spread":0.013},
    "GBPJPY": {"ticker":"GBPJPY=X", "label":"GBPJPY#",  "pip":0.01,   "digits":3, "pip_val":9.0,  "spread":0.022},
    "EURJPY": {"ticker":"EURJPY=X", "label":"EURJPY#",  "pip":0.01,   "digits":3, "pip_val":9.0,  "spread":0.018},
}

_cache = {}
CACHE_TTL = 60

# ── Data ──────────────────────────────────────────────────────────────────────
def synth(sym, n=600):
    cfg   = INSTRUMENTS[sym]
    bases = {"GOLD":4400,"EURUSD":1.12,"GBPUSD":1.28,
             "USDJPY":152,"GBPJPY":195,"EURJPY":170}
    p     = bases[sym]
    rng   = np.random.default_rng(seed=abs(hash(sym)) % (2**31))
    rows, times = [], []
    now = datetime(2025, 1, 1)
    trend = 0.0003; vol = p * 0.0018
    for i in range(n):
        if i % 40 == 0:
            trend = float(rng.choice([-6e-4,-3e-4,0,3e-4,6e-4]))
            vol   = p * float(rng.uniform(0.001, 0.003))
        o = p; c = o*(1+float(rng.normal(trend,vol/p)))
        hi = max(o,c)*(1+abs(float(rng.normal(0,4e-4))))
        lo = min(o,c)*(1-abs(float(rng.normal(0,4e-4))))
        rows.append({"Open":round(o,cfg["digits"]),"High":round(hi,cfg["digits"]),
                     "Low":round(lo,cfg["digits"]),"Close":round(c,cfg["digits"]),
                     "Volume":int(rng.uniform(500,5000))})
        times.append(now+timedelta(hours=i)); p=c
    return pd.DataFrame(rows, index=pd.DatetimeIndex(times))

def get_df(sym, tf="1h", period="60d"):
    key = f"{sym}_{tf}_{period}"
    if key in _cache and time.time()-_cache[key]["ts"] < CACHE_TTL:
        return _cache[key]["df"], _cache[key]["src"]
    df=None; src="synthetic"
    if HAS_YF:
        try:
            iv = "1h" if tf=="4h" else tf
            raw = yf.download(INSTRUMENTS[sym]["ticker"],period=period,
                              interval=iv,progress=False,auto_adjust=True)
            if isinstance(raw.columns,pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw[["Open","High","Low","Close","Volume"]].dropna()
            if tf=="4h":
                raw=raw.resample("4h").agg({"Open":"first","High":"max",
                                            "Low":"min","Close":"last","Volume":"sum"}).dropna()
            if len(raw)>=80: df=raw; src="live"
        except: pass
    if df is None: df=synth(sym,600); src="synthetic"
    _cache[key]={"df":df,"ts":time.time(),"src":src}
    return df, src

# ── Indicators ────────────────────────────────────────────────────────────────
def add_indicators(df):
    c,hi,lo = df["Close"],df["High"],df["Low"]; d=df.copy()
    d["ma9"] =c.rolling(9).mean(); d["ma20"]=c.rolling(20).mean(); d["ma50"]=c.rolling(50).mean()
    delta=c.diff(); g=delta.clip(lower=0).rolling(14).mean(); l=(-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"]=100-(100/(1+g/l.replace(0,np.nan)))
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    d["macd"]=e12-e26; d["macd_sig"]=d["macd"].ewm(span=9,adjust=False).mean()
    d["macd_hist"]=d["macd"]-d["macd_sig"]
    tr=pd.concat([hi-lo,(hi-c.shift()).abs(),(lo-c.shift()).abs()],axis=1).max(axis=1)
    d["atr"]=tr.rolling(14).mean()
    d["bb_mid"]=c.rolling(20).mean(); std=c.rolling(20).std()
    d["bb_up"]=d["bb_mid"]+2*std; d["bb_lo"]=d["bb_mid"]-2*std
    d["stoch"]=(c-lo.rolling(14).min())*100/(hi.rolling(14).max()-lo.rolling(14).min()+1e-10)
    d["momentum"]=c-c.shift(10)
    return d.dropna()

# ── Signal ────────────────────────────────────────────────────────────────────
def signal_at(i, df):
    if i<2: return None
    cur,prv = df.iloc[i],df.iloc[i-1]
    bull=bear=0; rb=[]; rs=[]
    if cur["ma20"]>cur["ma50"]:   bull+=2; rb.append("MA20 above MA50 — uptrend")
    else:                          bear+=2; rs.append("MA20 below MA50 — downtrend")
    if cur["Close"]>cur["ma20"]>cur["ma50"]:  bull+=1; rb.append("Price above both MAs")
    elif cur["Close"]<cur["ma20"]<cur["ma50"]: bear+=1; rs.append("Price below both MAs")
    if cur["rsi"]<35:              bull+=2; rb.append(f"RSI {cur['rsi']:.1f} oversold")
    elif cur["rsi"]>65:            bear+=2; rs.append(f"RSI {cur['rsi']:.1f} overbought")
    if cur["macd_hist"]>0 and prv["macd_hist"]<=0:   bull+=3; rb.append("MACD histogram crossed up")
    elif cur["macd_hist"]<0 and prv["macd_hist"]>=0: bear+=3; rs.append("MACD histogram crossed down")
    elif cur["macd_hist"]>0:       bull+=1; rb.append("MACD histogram positive")
    else:                          bear+=1; rs.append("MACD histogram negative")
    if cur["ma9"]>cur["ma20"] and prv["ma9"]<=prv["ma20"]:  bull+=2; rb.append("MA9 crossed above MA20")
    elif cur["ma9"]<cur["ma20"] and prv["ma9"]>=prv["ma20"]: bear+=2; rs.append("MA9 crossed below MA20")
    if cur["Close"]<=cur["bb_lo"]: bull+=1; rb.append("Price at lower Bollinger Band")
    elif cur["Close"]>=cur["bb_up"]: bear+=1; rs.append("Price at upper Bollinger Band")
    if cur["stoch"]<25 and cur["stoch"]>prv["stoch"]: bull+=1; rb.append("Stochastic oversold + rising")
    elif cur["stoch"]>75 and cur["stoch"]<prv["stoch"]: bear+=1; rs.append("Stochastic overbought + falling")
    if cur["momentum"]>0 and cur["ma20"]>cur["ma50"]: bull+=1; rb.append("Positive momentum in uptrend")
    elif cur["momentum"]<0 and cur["ma20"]<cur["ma50"]: bear+=1; rs.append("Negative momentum in downtrend")
    total=bull+bear
    if total==0: return None
    if bull>=6 and bull>bear: return {"dir":"BUY", "conf":min(95,round(bull/total*100)),"bull":bull,"bear":bear,"reasons":rb}
    if bear>=6 and bear>bull: return {"dir":"SELL","conf":min(95,round(bear/total*100)),"bull":bull,"bear":bear,"reasons":rs}
    return {"dir":"WAIT","conf":0,"bull":bull,"bear":bear,"reasons":rb+rs}

# ── Backtest ──────────────────────────────────────────────────────────────────
def backtest(df,sym,sl_m=1.5,tp_m=2.5,bal=10000,risk=2,min_conf=60):
    inst=INSTRUMENTS[sym]; trades=[]; equity=[bal]; balance=bal
    in_trade=False; trade=None
    for i in range(60,len(df)):
        row=df.iloc[i]
        if in_trade:
            t=trade; hit_sl=hit_tp=False
            if t["dir"]=="BUY":
                if row["Low"] <=t["sl"]: hit_sl=True
                if row["High"]>=t["tp"]: hit_tp=True
            else:
                if row["High"]>=t["sl"]: hit_sl=True
                if row["Low"] <=t["tp"]: hit_tp=True
            if hit_tp:   pnl=t["risk"]*tp_m/sl_m; outcome="WIN"
            elif hit_sl: pnl=-t["risk"];            outcome="LOSS"
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
                          "exit_dt":df.index[i].strftime("%Y-%m-%d %H:%M"),"bars_held":t["bars"]})
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
    avg_win =round(np.mean([t["pnl"] for t in wins]), 2) if wins   else 0
    avg_loss=round(np.mean([t["pnl"] for t in losses]),2) if losses else 0
    expect  =round(wr/100*avg_win+(1-wr/100)*avg_loss,2)
    return {"trades":trades,"equity":equity,"win_rate":wr,"profit_factor":pf,
            "max_dd":round(max_dd,2),"total":len(trades),"wins":len(wins),"losses":len(losses),
            "net_pnl":round(balance-bal,2),"sharpe":sharpe,"final_bal":round(balance,2),
            "avg_win":avg_win,"avg_loss":avg_loss,"expectancy":expect}

def grade(r):
    s=sum([r["win_rate"]>=55,r["profit_factor"]>=1.5,r["max_dd"]<=15,r["net_pnl"]>0,r["sharpe"]>1])
    return {5:"A+",4:"A",3:"B",2:"C",1:"D",0:"F"}[s]

def load_json(name,default): 
    p=os.path.join(DATA_DIR,name)
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return default

def save_json(name,data):
    with open(os.path.join(DATA_DIR,name),"w") as f: json.dump(data,f,indent=2)

# ═══════════════════════ ROUTES ═══════════════════════════════════════════════

@app.route("/")
def index(): return send_from_directory(FRONTEND,"index.html")

@app.route("/<path:path>")
def static_files(path): return send_from_directory(FRONTEND,path)

@app.route("/api/health")
def health(): return jsonify({"status":"ok","live_data":HAS_YF,"ts":datetime.utcnow().isoformat()})

@app.route("/api/instruments")
def instruments_list():
    return jsonify([{"id":k,"label":v["label"],"spread":v["spread"]} for k,v in INSTRUMENTS.items()])

@app.route("/api/quote/<sym>")
def quote(sym):
    if sym not in INSTRUMENTS: return jsonify({"error":"unknown"}),400
    tf=request.args.get("tf","1h"); period=request.args.get("period","60d")
    n_c=int(request.args.get("candles",150))
    df,src=get_df(sym,tf,period); inst=INSTRUMENTS[sym]; df2=add_indicators(df)
    last=df2.iloc[-1]; prev=df2.iloc[-2]; sig=signal_at(len(df2)-1,df2)
    atr=float(last["atr"]); d=inst["digits"]; sl_m,tp_m=1.5,2.5
    entry=sl=tp1=tp2=sl_pips=tp_pips=rr=None
    if sig and sig["dir"] not in ("WAIT",None):
        entry=round(float(last["Close"]),d)
        if sig["dir"]=="BUY": sl=round(entry-atr*sl_m,d);tp1=round(entry+atr*tp_m,d);tp2=round(entry+atr*4,d)
        else:                  sl=round(entry+atr*sl_m,d);tp1=round(entry-atr*tp_m,d);tp2=round(entry-atr*4,d)
        sl_pips=round(abs(entry-sl)/inst["pip"]); tp_pips=round(abs(tp1-entry)/inst["pip"])
        rr=round(tp_pips/sl_pips,1) if sl_pips else 0
    tail=df2.tail(n_c); candles=[]
    for idx,r in tail.iterrows():
        candles.append({"t":str(idx),"o":round(float(r["Open"]),d),"h":round(float(r["High"]),d),
                        "l":round(float(r["Low"]),d),"c":round(float(r["Close"]),d),"v":int(r["Volume"]),
                        "ma9":round(float(r["ma9"]),d),"ma20":round(float(r["ma20"]),d),
                        "ma50":round(float(r["ma50"]),d),"rsi":round(float(r["rsi"]),1),
                        "macd_hist":round(float(r["macd_hist"]),6),"macd_line":round(float(r["macd"]),6),
                        "macd_sig":round(float(r["macd_sig"]),6),"bb_up":round(float(r["bb_up"]),d),
                        "bb_lo":round(float(r["bb_lo"]),d),"stoch":round(float(r["stoch"]),1)})
    chg=round((float(last["Close"])-float(prev["Close"]))/float(prev["Close"])*100,3)
    return jsonify({"sym":sym,"label":inst["label"],"digits":d,
                    "price":round(float(last["Close"]),d),"open":round(float(last["Open"]),d),
                    "high":round(float(last["High"]),d),"low":round(float(last["Low"]),d),
                    "change":chg,"rsi":round(float(last["rsi"]),1),"atr":round(atr,d),
                    "stoch":round(float(last["stoch"]),1),"signal":sig,
                    "entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,
                    "sl_pips":sl_pips,"tp_pips":tp_pips,"rr":rr,
                    "ma20":round(float(last["ma20"]),d),"ma50":round(float(last["ma50"]),d),
                    "bb_up":round(float(last["bb_up"]),d),"bb_lo":round(float(last["bb_lo"]),d),
                    "spread":inst["spread"],"candles":candles,"src":src,
                    "ts":datetime.utcnow().strftime("%H:%M:%S UTC")})

@app.route("/api/backtest/<sym>")
def do_backtest(sym):
    if sym not in INSTRUMENTS: return jsonify({"error":"unknown"}),400
    tf=request.args.get("tf","1h"); period=request.args.get("period","120d")
    bal=float(request.args.get("balance","10000")); risk=float(request.args.get("risk","2"))
    sl_m=float(request.args.get("sl","1.5")); tp_m=float(request.args.get("tp","2.5"))
    min_conf=int(request.args.get("conf","60"))
    df,src=get_df(sym,tf,period); df2=add_indicators(df); r=backtest(df2,sym,sl_m,tp_m,bal,risk,min_conf)
    r.update({"src":src,"sym":sym,"label":INSTRUMENTS[sym]["label"],"grade":grade(r),
               "date_range":f"{df2.index[0].strftime('%d %b %Y')} → {df2.index[-1].strftime('%d %b %Y')}"})
    return jsonify(r)

@app.route("/api/sweep")
def sweep():
    tf=request.args.get("tf","1h"); bal=float(request.args.get("balance","10000"))
    risk=float(request.args.get("risk","2")); results=[]
    for sym in INSTRUMENTS:
        df,src=get_df(sym,tf); df2=add_indicators(df)
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
        df,src=get_df(sym,tf); df2=add_indicators(df)
        last=df2.iloc[-1]; prev=df2.iloc[-2]; sig=signal_at(len(df2)-1,df2)
        inst=INSTRUMENTS[sym]; d=inst["digits"]
        price=round(float(last["Close"]),d); chg=round((price-float(prev["Close"]))/float(prev["Close"])*100,3)
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
                df,_=get_df(sym,"1h"); price=float(df["Close"].iloc[-1])
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
        note=request.json.get("note",""); updated=[]
        for t in data:
            if str(t.get("id"))==str(tid):
                inst=INSTRUMENTS.get(t["sym"],{}); pip=inst.get("pip",1e-4)
                pipval=inst.get("pip_val",10); d_=inst.get("digits",5)
                lots=float(t.get("lots",0.01)); entry=float(t.get("entry",0))
                diff=exit_px-entry if t["dir"]=="BUY" else entry-exit_px
                pnl=round(diff/pip*pipval*lots,2)
                t.update({"status":"closed","exit_price":round(exit_px,d_),
                          "closed":datetime.utcnow().isoformat(),"pnl":pnl,"note":note})
            updated.append(t)
        save_json("journal.json",updated); return jsonify({"ok":True})
    if request.method=="DELETE":
        tid=request.args.get("id"); data=[t for t in data if str(t.get("id"))!=str(tid)]
        save_json("journal.json",data); return jsonify({"ok":True})

@app.route("/api/stats")
def stats():
    j=load_json("journal.json",[]); closed=[t for t in j if t.get("status")=="closed"]
    wins=[t for t in closed if (t.get("pnl") or 0)>0]
    losses=[t for t in closed if (t.get("pnl") or 0)<=0]; pnls=[t.get("pnl",0) for t in closed]
    return jsonify({"total_trades":len(closed),"wins":len(wins),"losses":len(losses),
                    "win_rate":round(len(wins)/len(closed)*100,1) if closed else 0,
                    "total_pnl":round(sum(pnls),2),
                    "avg_win":round(np.mean([t["pnl"] for t in wins]),2) if wins else 0,
                    "avg_loss":round(np.mean([t["pnl"] for t in losses]),2) if losses else 0,
                    "best_trade":max(pnls) if pnls else 0,"worst_trade":min(pnls) if pnls else 0})

if __name__=="__main__":
    print("\n"+"="*55)
    print("  XM360 Trade Dashboard v2")
    print(f"  Data: {'LIVE — Yahoo Finance' if HAS_YF else 'SYNTHETIC'}")
    print("  URL:  http://localhost:5000")
    print("="*55+"\n")
    app.run(debug=False,port=5000,host="0.0.0.0")
